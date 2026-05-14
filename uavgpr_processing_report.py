# Copyright (C) 2015-2023: The University of Edinburgh
#                 Authors: Craig Warren and Antonis Giannopoulos
#
# This file is part of gprMax.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gprMax is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gprMax.  If not, see <http://www.gnu.org/licenses/>.

"""MyGPR processing report bridge for UavGPR gprMax GUI outputs."""

from __future__ import annotations

import html
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import h5py
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


C0 = 299792458.0
DEFAULT_MYGPR_ROOT = r"D:\MyGPR"
EPS = 1.0e-12


@dataclass
class ProcessingCandidate:
    background_id: str
    background_params: Dict[str, Any] = field(default_factory=dict)
    gain_id: str = "none"
    gain_params: Dict[str, Any] = field(default_factory=dict)
    rerun_reason: str = ""

    @property
    def candidate_id(self) -> str:
        return "{0}__{1}".format(self.background_id, self.gain_id)


def run_processing_report(
    manifest_path: str,
    mygpr_root: Optional[str] = None,
    output_dir: Optional[str] = None,
    log_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Run MyGPR processing combinations and write a standalone HTML report."""

    manifest_file = Path(manifest_path)
    with manifest_file.open("r", encoding="utf-8") as fobj:
        manifest = json.load(fobj)

    root = Path(mygpr_root or os.environ.get("MYGPR_ROOT") or DEFAULT_MYGPR_ROOT)
    run_processing_method = _load_mygpr_processing_engine(root)
    data, dt = _load_bscan(manifest)
    report_dir = Path(output_dir) if output_dir else manifest_file.parent / "processing_report"
    report_dir.mkdir(parents=True, exist_ok=True)

    raw_png = report_dir / "raw_bscan.png"
    _save_bscan_png(data, dt, raw_png, "Raw Ez B-scan")

    target_roi = _estimate_target_roi(manifest, data.shape, dt)
    candidates = _build_candidates(data.shape[1])
    results = []
    raw_metrics = _metric_context(data, target_roi)

    for candidate in candidates:
        _log(log_callback, "Processing {0}".format(candidate.candidate_id))
        result = _run_candidate(
            data,
            candidate,
            run_processing_method,
            dt=dt,
            total_time_ns=float(data.shape[0]) * float(dt) * 1.0e9,
        )
        score = _score_result(data, result["data"], target_roi, raw_metrics)
        adjusted = _maybe_adjust_candidate(candidate, score)
        if adjusted is not None:
            _log(
                log_callback,
                "Feedback rerun {0}: {1}".format(
                    adjusted.candidate_id, adjusted.rerun_reason
                ),
            )
            result = _run_candidate(
                data,
                adjusted,
                run_processing_method,
                dt=dt,
                total_time_ns=float(data.shape[0]) * float(dt) * 1.0e9,
            )
            score = _score_result(data, result["data"], target_roi, raw_metrics)
            candidate = adjusted

        image_name = "{0}.png".format(candidate.candidate_id)
        image_path = report_dir / image_name
        _save_bscan_png(result["data"], dt, image_path, candidate.candidate_id)
        record = {
            "candidate_id": candidate.candidate_id,
            "background": {
                "method": candidate.background_id,
                "params": _json_safe(candidate.background_params),
            },
            "gain": {
                "method": candidate.gain_id,
                "params": _json_safe(candidate.gain_params),
            },
            "rerun_reason": candidate.rerun_reason,
            "score": _json_safe(score),
            "image": image_name,
            "method_metadata": _json_safe(result["metadata"]),
        }
        results.append(record)

    best = _choose_best(results)
    summary = {
        "schema": "uavgpr_processing_report_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "manifest_path": str(manifest_file),
        "mygpr_root": str(root),
        "primary_out_file": manifest.get("primary_out_file"),
        "raw_is_unchanged": True,
        "preview_processing_only": True,
        "data_shape": [int(data.shape[0]), int(data.shape[1])],
        "dt_s": float(dt),
        "target_roi": _json_safe(target_roi),
        "raw_image": raw_png.name,
        "candidates": results,
        "recommended": best,
    }

    summary_path = report_dir / "processing_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    html_path = report_dir / "index.html"
    html_path.write_text(_build_html_report(summary), encoding="utf-8")
    summary["report_html"] = str(html_path)
    summary["summary_json"] = str(summary_path)
    return summary


def _load_mygpr_processing_engine(root: Path) -> Callable[[np.ndarray, str, Dict[str, Any]], Tuple[np.ndarray, Dict[str, Any]]]:
    if not root.exists():
        raise FileNotFoundError("MyGPR root not found: {0}".format(root))
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    from core.processing_engine import run_processing_method

    return run_processing_method


def _load_bscan(manifest: Dict[str, Any]) -> Tuple[np.ndarray, float]:
    primary = manifest.get("primary_out_file") or manifest.get("merged_out_file")
    if not primary:
        raise ValueError("manifest does not contain primary_out_file")
    with h5py.File(str(primary), "r") as fobj:
        dt = float(fobj.attrs["dt"])
        component = str(manifest.get("component") or "Ez")
        data = np.asarray(fobj["/rxs/rx1/{0}".format(component)][:], dtype=np.float32)
    if data.ndim == 1:
        data = data[:, np.newaxis]
    return data, dt


def _build_candidates(trace_count: int) -> List[ProcessingCandidate]:
    background_window = _adapt_trace_window(51, trace_count)
    wider_background_window = _adapt_trace_window(81, trace_count)
    agc_window = 151
    backgrounds = [
        ProcessingCandidate("none", {}, "none", {}),
        ProcessingCandidate(
            "subtracting_average_2D",
            {"ntraces": background_window},
            "none",
            {},
        ),
        ProcessingCandidate(
            "median_background_2D",
            {"ntraces": background_window},
            "none",
            {},
        ),
    ]
    gains = [
        ("none", {}),
        ("sec_gain", {"gain_min": 1.0, "gain_max": 4.0, "power": 1.2}),
        ("agcGain", {"window": agc_window, "_low_energy_guard": True}),
        (
            "energy_decay_gain",
            {"strength": 0.8, "smoothing_samples": 61, "max_gain": 6.0},
        ),
        ("time_power_gain", {"gain_min": 1.0, "gain_max": 5.0, "power": 1.4}),
    ]

    candidates = []
    for background in backgrounds:
        for gain_id, gain_params in gains:
            candidates.append(
                ProcessingCandidate(
                    background.background_id,
                    dict(background.background_params),
                    gain_id,
                    dict(gain_params),
                )
            )
    for candidate in candidates:
        candidate.background_params.setdefault("_fallback_ntraces", wider_background_window)
    return candidates


def _adapt_trace_window(window: int, trace_count: int) -> int:
    count = max(1, int(trace_count))
    resolved = min(int(window), count)
    if resolved > 1 and resolved % 2 == 0:
        resolved -= 1
    return max(1, resolved)


def _run_candidate(
    data: np.ndarray,
    candidate: ProcessingCandidate,
    run_processing_method: Callable[[np.ndarray, str, Dict[str, Any]], Tuple[np.ndarray, Dict[str, Any]]],
    *,
    dt: float,
    total_time_ns: float,
) -> Dict[str, Any]:
    current = np.array(data, copy=True)
    metadata = []
    if candidate.background_id != "none":
        params = {
            key: value
            for key, value in candidate.background_params.items()
            if not str(key).startswith("_")
        }
        params.setdefault("time_window_ns", total_time_ns)
        current, meta = run_processing_method(current, candidate.background_id, params)
        metadata.append(meta)
    if candidate.gain_id != "none":
        params = dict(candidate.gain_params)
        params.setdefault("time_step_s", dt)
        current, meta = run_processing_method(current, candidate.gain_id, params)
        metadata.append(meta)
    return {"data": current, "metadata": metadata}


def _maybe_adjust_candidate(
    candidate: ProcessingCandidate, score: Dict[str, float]
) -> Optional[ProcessingCandidate]:
    adjusted = ProcessingCandidate(
        candidate.background_id,
        dict(candidate.background_params),
        candidate.gain_id,
        dict(candidate.gain_params),
    )
    reasons = []
    if score["target_preservation"] < 0.45 and adjusted.background_id != "none":
        fallback = int(adjusted.background_params.get("_fallback_ntraces", 81))
        if int(adjusted.background_params.get("ntraces", fallback)) != fallback:
            adjusted.background_params["ntraces"] = fallback
            reasons.append("target_preservation_low_background_window_to_{0}".format(fallback))

    if score["hot_pixel_ratio"] > 0.01 or score["peak_to_p99"] > 10.0:
        if "gain_max" in adjusted.gain_params:
            adjusted.gain_params["gain_max"] = float(adjusted.gain_params["gain_max"]) * 0.75
            reasons.append("overhot_gain_max_reduced")
        elif adjusted.gain_id == "energy_decay_gain":
            adjusted.gain_params["max_gain"] = float(adjusted.gain_params["max_gain"]) * 0.75
            reasons.append("overhot_max_gain_reduced")
        elif adjusted.gain_id == "agcGain":
            adjusted.gain_params["window"] = int(adjusted.gain_params.get("window", 151)) * 2
            reasons.append("overhot_agc_window_widened")

    if not reasons:
        return None
    adjusted.rerun_reason = ", ".join(reasons)
    return adjusted


def _estimate_target_roi(
    manifest: Dict[str, Any], shape: Tuple[int, int], dt: float
) -> Dict[str, int]:
    samples, traces = shape
    targets = manifest.get("simple_targets") or []
    target = targets[0] if targets else {}
    scan = manifest.get("scan_geometry") or {}
    medium = manifest.get("medium") or {}
    target_x = float(target.get("center_x_m", scan.get("scan_mid_x_m", 0.5)))
    target_y = float(target.get("center_y_m", 0.25))
    source_start_x = float(scan.get("source_start_x_m", 0.0))
    receiver_offset = float(scan.get("receiver_offset_m", 0.0))
    step = float(scan.get("effective_scan_step_m", manifest.get("effective_scan_step_m", 1.0)))
    ground_y = float(scan.get("ground_surface_y_m", 0.0))
    lift_off = float(scan.get("uav_lift_off_m", manifest.get("uav_lift_off_m", 0.0)))
    eps_r = max(1.0, float(medium.get("host_eps_r", 9.0)))

    midpoints = source_start_x + 0.5 * receiver_offset + np.arange(traces) * step
    trace_center = int(np.argmin(np.abs(midpoints - target_x))) if traces > 1 else 0
    depth = max(0.0, ground_y - target_y)
    soil_velocity = C0 / math_sqrt(eps_r)
    twt_s = 2.0 * lift_off / C0 + 2.0 * depth / soil_velocity
    sample_center = int(round(twt_s / max(float(dt), EPS)))
    sample_center = max(0, min(samples - 1, sample_center))

    sample_radius = max(8, int(round(0.08 * samples)))
    trace_radius = max(3, int(round(0.12 * traces)))
    return {
        "sample_start": max(0, sample_center - sample_radius),
        "sample_end": min(samples, sample_center + sample_radius + 1),
        "trace_start": max(0, trace_center - trace_radius),
        "trace_end": min(traces, trace_center + trace_radius + 1),
        "sample_center": sample_center,
        "trace_center": trace_center,
    }


def math_sqrt(value: float) -> float:
    return float(np.sqrt(float(value)))


def _metric_context(data: np.ndarray, target_roi: Dict[str, int]) -> Dict[str, float]:
    target = _target_slice(data, target_roi)
    background = _background_values(data, target_roi)
    return {
        "target_rms": _rms(target),
        "background_rms": _rms(background),
        "p99_abs": float(np.percentile(np.abs(data), 99.0)) if data.size else 0.0,
    }


def _score_result(
    raw: np.ndarray,
    processed: np.ndarray,
    target_roi: Dict[str, int],
    raw_metrics: Dict[str, float],
) -> Dict[str, float]:
    target = _target_slice(processed, target_roi)
    background = _background_values(processed, target_roi)
    target_rms = _rms(target)
    background_rms = _rms(background)
    saliency = target_rms / max(background_rms, EPS)
    raw_saliency = raw_metrics["target_rms"] / max(raw_metrics["background_rms"], EPS)
    target_preservation = target_rms / max(raw_metrics["target_rms"], EPS)
    background_reduction = 1.0 - background_rms / max(raw_metrics["background_rms"], EPS)
    abs_processed = np.abs(processed)
    p99 = float(np.percentile(abs_processed, 99.0)) if processed.size else 0.0
    peak = float(np.max(abs_processed)) if processed.size else 0.0
    hot_threshold = max(raw_metrics["p99_abs"] * 8.0, EPS)
    hot_pixel_ratio = float(np.mean(abs_processed > hot_threshold)) if processed.size else 0.0
    peak_to_p99 = peak / max(p99, EPS)
    score = (
        0.40 * min(saliency / max(raw_saliency, EPS), 4.0)
        + 0.30 * max(min(background_reduction, 1.0), -1.0)
        + 0.20 * min(target_preservation, 2.0)
        - 0.30 * min(hot_pixel_ratio * 20.0, 1.0)
        - 0.10 * max(0.0, min((peak_to_p99 - 10.0) / 10.0, 1.0))
    )
    return {
        "score": float(score),
        "target_rms": float(target_rms),
        "background_rms": float(background_rms),
        "saliency": float(saliency),
        "saliency_gain": float(saliency / max(raw_saliency, EPS)),
        "target_preservation": float(target_preservation),
        "background_reduction": float(background_reduction),
        "hot_pixel_ratio": float(hot_pixel_ratio),
        "peak_to_p99": float(peak_to_p99),
    }


def _target_slice(data: np.ndarray, roi: Dict[str, int]) -> np.ndarray:
    return data[
        int(roi["sample_start"]): int(roi["sample_end"]),
        int(roi["trace_start"]): int(roi["trace_end"]),
    ]


def _background_values(data: np.ndarray, roi: Dict[str, int]) -> np.ndarray:
    mask = np.ones(data.shape, dtype=bool)
    mask[
        int(roi["sample_start"]): int(roi["sample_end"]),
        int(roi["trace_start"]): int(roi["trace_end"]),
    ] = False
    values = data[mask]
    return values if values.size else data.reshape(-1)


def _rms(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(arr * arr)))


def _choose_best(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {}
    ranked = sorted(
        results,
        key=lambda item: float(item["score"].get("score", -1.0e9)),
        reverse=True,
    )
    best = dict(ranked[0])
    best["reason"] = (
        "最高综合评分，兼顾目标 ROI 保留、背景压制和过曝风险。"
    )
    return best


def _save_bscan_png(data: np.ndarray, dt: float, path: Path, title: str) -> None:
    vmax = float(np.percentile(np.abs(data), 99.5)) if data.size else 1.0
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    extent = [0, data.shape[1], data.shape[0] * float(dt) * 1.0e9, 0]
    fig = Figure(figsize=(8.5, 4.8), dpi=130)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    image = ax.imshow(
        data,
        aspect="auto",
        cmap="seismic",
        vmin=-vmax,
        vmax=vmax,
        extent=extent,
        interpolation="nearest",
    )
    ax.set_title(title)
    ax.set_xlabel("Trace")
    ax.set_ylabel("Time [ns]")
    fig.colorbar(image, ax=ax, shrink=0.85, label="Ez")
    fig.tight_layout()
    fig.savefig(path)


def _build_html_report(summary: Dict[str, Any]) -> str:
    rows = []
    for item in summary["candidates"]:
        score = item["score"]
        rows.append(
            """
            <article class="card">
              <h3>{candidate}</h3>
              <img src="{image}" alt="{candidate}">
              <p><b>Background:</b> {background} | <b>Gain:</b> {gain}</p>
              <p><b>Score:</b> {score:.3f} | saliency gain {saliency:.2f} | target keep {keep:.2f} | bg reduction {bg:.2f}</p>
              <p class="muted">{rerun}</p>
            </article>
            """.format(
                candidate=html.escape(item["candidate_id"]),
                image=html.escape(item["image"]),
                background=html.escape(item["background"]["method"]),
                gain=html.escape(item["gain"]["method"]),
                score=float(score["score"]),
                saliency=float(score["saliency_gain"]),
                keep=float(score["target_preservation"]),
                bg=float(score["background_reduction"]),
                rerun=html.escape(item.get("rerun_reason") or "no feedback rerun"),
            )
        )
    best = summary.get("recommended") or {}
    return """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>UavGPR MyGPR Processing Report</title>
  <style>
    body {{ margin:0; font-family:Segoe UI, Microsoft YaHei, sans-serif; background:#f8fafc; color:#111827; }}
    header {{ padding:28px 34px; background:#0f172a; color:#e5e7eb; }}
    main {{ padding:24px 34px; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(360px, 1fr)); gap:18px; }}
    .card {{ background:white; border:1px solid #d1d5db; border-radius:8px; padding:14px; box-shadow:0 1px 3px rgba(15,23,42,.08); }}
    img {{ width:100%; border:1px solid #e5e7eb; border-radius:6px; }}
    .muted {{ color:#64748b; }}
    code {{ background:#e5e7eb; padding:2px 4px; border-radius:4px; }}
  </style>
</head>
<body>
  <header>
    <p>gprMax GUI / MyGPR processing bridge</p>
    <h1>UavGPR 背景抑制与增益组合报告</h1>
    <p>Raw .out/_merged.out 未修改；本报告只生成派生 PNG/JSON/HTML。</p>
  </header>
  <main>
    <section class="card">
      <h2>推荐组合</h2>
      <p><b>{best_id}</b></p>
      <p>{best_reason}</p>
      <p>数据 shape: <code>{shape}</code>, dt: <code>{dt}</code></p>
      <p>主输出: <code>{primary}</code></p>
    </section>
    <section class="card">
      <h2>Raw B-scan</h2>
      <img src="{raw_image}" alt="Raw B-scan">
    </section>
    <section>
      <h2>候选组合</h2>
      <div class="grid">
        {rows}
      </div>
    </section>
  </main>
</body>
</html>
""".format(
        best_id=html.escape(str(best.get("candidate_id", ""))),
        best_reason=html.escape(str(best.get("reason", ""))),
        shape=html.escape(str(summary.get("data_shape"))),
        dt=html.escape(str(summary.get("dt_s"))),
        primary=html.escape(str(summary.get("primary_out_file"))),
        raw_image=html.escape(str(summary.get("raw_image"))),
        rows="\n".join(rows),
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _log(callback: Optional[Callable[[str], None]], message: str) -> None:
    if callback is not None:
        callback(message)
