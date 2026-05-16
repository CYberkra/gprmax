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

"""Validate UavGPR simulation output folders before handoff to MyGPR."""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np


@dataclass
class ValidationMessage(object):
    level: str
    text: str


@dataclass
class ValidationResult(object):
    output_dir: str
    manifest_path: str
    messages: List[ValidationMessage]

    def has_errors(self) -> bool:
        return any(message.level == "error" for message in self.messages)

    def errors(self) -> List[ValidationMessage]:
        return [message for message in self.messages if message.level == "error"]

    def warnings(self) -> List[ValidationMessage]:
        return [message for message in self.messages if message.level == "warning"]


def _add(messages: List[ValidationMessage], level: str, text: str) -> None:
    messages.append(ValidationMessage(level, text))


def _as_float_list(value: object) -> List[float]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return [float(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    return [float(value)]


def _json_attr(value: object) -> object:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _almost_equal(left: object, right: object, atol: float = 1e-9) -> bool:
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if left_array.shape != right_array.shape:
        return False
    if np.issubdtype(left_array.dtype, np.number) and np.issubdtype(
        right_array.dtype, np.number
    ):
        return bool(np.allclose(left_array, right_array, rtol=0.0, atol=atol))
    return left_array.tolist() == right_array.tolist()


def _resolve_path(base_dir: str, path_value: object) -> str:
    if not path_value:
        return ""
    path = os.path.expanduser(str(path_value))
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(base_dir, path))


def _load_json(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as fobj:
        return json.load(fobj)


def _find_manifest(output_dir: str, manifest_path: Optional[str]) -> str:
    if manifest_path:
        return _resolve_path(output_dir, manifest_path)
    matches = sorted(glob.glob(os.path.join(output_dir, "*_manifest.json")))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            "found multiple manifests; pass --manifest explicitly: {0}".format(
                ", ".join(matches)
            )
        )
    raise ValueError("no *_manifest.json found in {0}".format(output_dir))


def _target_bounds_from_config(config: Dict[str, object]) -> Optional[Tuple[float, float]]:
    if not config:
        return None

    if bool(config.get("use_curved_crack", False)):
        path_text = str(config.get("crack_path_text", "")).strip()
        points = []
        for line in path_text.splitlines():
            parts = line.split()
            if len(parts) >= 2:
                points.append((float(parts[0]), float(parts[1])))
        if points:
            width = float(config.get("target_height", 0.0))
            xs = [point[0] for point in points]
            return min(xs) - width, max(xs) + width

    center_x = float(config.get("target_center_x", 0.0))
    shape = str(config.get("target_shape", ""))
    if shape == "cylinder":
        radius = float(config.get("target_radius", 0.0))
        return center_x - radius, center_x + radius
    if shape == "crack":
        half_width = 0.5 * float(config.get("target_width", 0.0))
        half_height = 0.5 * float(config.get("target_height", 0.0))
        angle = 0.0
        if str(config.get("target_orientation", "")) == "vertical":
            angle = 90.0
        elif str(config.get("target_orientation", "")) == "angled":
            angle = float(config.get("target_angle_deg", 0.0))
        angle_rad = math.radians(angle)
        projected_half_width = abs(half_width * math.cos(angle_rad)) + abs(
            half_height * math.sin(angle_rad)
        )
        return center_x - projected_half_width, center_x + projected_half_width
    return center_x, center_x


def _coverage_from_config(
    config: Dict[str, object], positions: Optional[np.ndarray]
) -> Optional[Tuple[float, float]]:
    if positions is not None and positions.size:
        receiver_offset = float(config.get("receiver_offset", 0.0))
        midpoints = positions[:, 0] - 0.5 * receiver_offset
        return float(np.min(midpoints)), float(np.max(midpoints))

    if not config:
        return None
    source_start_x = float(config.get("source_start_x", 0.0))
    receiver_offset = float(config.get("receiver_offset", 0.0))
    n_traces = int(config.get("n_traces", 1))
    scan_step = float(
        config.get("effective_scan_step", config.get("scan_step", 0.0))
    )
    start = source_start_x + 0.5 * receiver_offset
    end = start + max(0, n_traces - 1) * scan_step
    return min(start, end), max(start, end)


def _read_metadata(manifest: Dict[str, object], output_dir: str) -> Dict[str, object]:
    metadata_path = _resolve_path(output_dir, manifest.get("metadata_file"))
    if metadata_path and os.path.exists(metadata_path):
        return _load_json(metadata_path)
    matches = sorted(glob.glob(os.path.join(output_dir, "*_metadata.json")))
    if len(matches) == 1:
        return _load_json(matches[0])
    return {}


def _check_required_paths(
    messages: List[ValidationMessage], manifest: Dict[str, object], output_dir: str
) -> Tuple[str, str]:
    input_path = _resolve_path(output_dir, manifest.get("input_file"))
    if not input_path:
        matches = sorted(glob.glob(os.path.join(output_dir, "*.in")))
        input_path = matches[0] if matches else ""
    if input_path and os.path.exists(input_path):
        _add(messages, "ok", ".in exists: {0}".format(input_path))
    else:
        _add(messages, "error", ".in file is missing")

    primary_out_path = _resolve_path(output_dir, manifest.get("primary_out_file"))
    if primary_out_path and os.path.exists(primary_out_path):
        _add(messages, "ok", "primary .out exists: {0}".format(primary_out_path))
    else:
        _add(messages, "error", "primary_out_file is missing or does not exist")
    return input_path, primary_out_path


def _check_manifest_summary(
    messages: List[ValidationMessage],
    manifest: Dict[str, object],
    attrs: Dict[str, object],
    datasets: Dict[str, Sequence[int]],
    rx_name: str,
    component: str,
) -> None:
    summary = manifest.get("primary_out_summary", {})
    if not isinstance(summary, dict):
        _add(messages, "warning", "manifest has no primary_out_summary object")
        return

    summary_attrs = summary.get("attrs", {})
    if isinstance(summary_attrs, dict):
        for key in ("Iterations", "dt", "nrx", "nsrc", "dx_dy_dz", "rxsteps"):
            if key in summary_attrs and key in attrs:
                if not _almost_equal(summary_attrs[key], attrs[key]):
                    _add(
                        messages,
                        "error",
                        "manifest attr {0} does not match HDF5".format(key),
                    )

    receivers = summary.get("receivers", {})
    if isinstance(receivers, dict) and rx_name in receivers:
        receiver_summary = receivers[rx_name]
        if isinstance(receiver_summary, dict):
            summary_datasets = receiver_summary.get("datasets", {})
            if isinstance(summary_datasets, dict):
                for dataset_name in (component, "Positions"):
                    if dataset_name in summary_datasets and dataset_name in datasets:
                        if list(summary_datasets[dataset_name]) != list(
                            datasets[dataset_name]
                        ):
                            _add(
                                messages,
                                "error",
                                "manifest dataset {0} shape does not match HDF5".format(
                                    dataset_name
                                ),
                            )
    _add(messages, "ok", "manifest summary matches checked HDF5 fields")


def _check_hdf5(
    messages: List[ValidationMessage],
    manifest: Dict[str, object],
    metadata: Dict[str, object],
    primary_out_path: str,
    component: str,
    rx_name: str,
) -> None:
    if not primary_out_path or not os.path.exists(primary_out_path):
        return

    dataset_path = "/rxs/{0}/{1}".format(rx_name, component)
    positions = None
    with h5py.File(primary_out_path, "r") as fobj:
        attrs = {key: _json_attr(fobj.attrs[key]) for key in fobj.attrs}
        if dataset_path not in fobj:
            _add(messages, "error", "{0} is missing".format(dataset_path))
            return

        data = fobj[dataset_path]
        datasets = {component: list(data.shape)}
        iterations = int(attrs.get("Iterations", data.shape[0]))
        if data.ndim not in (1, 2):
            _add(messages, "error", "{0} must be 1D or 2D".format(dataset_path))
        elif data.shape[0] != iterations:
            _add(
                messages,
                "error",
                "{0} samples do not match Iterations attr".format(dataset_path),
            )
        else:
            _add(messages, "ok", "{0} shape is valid: {1}".format(dataset_path, data.shape))

        trace_count = 1 if data.ndim == 1 else int(data.shape[1])
        merged_count = attrs.get("MergedModelCount")
        if merged_count is not None and int(merged_count) != trace_count:
            _add(messages, "error", "MergedModelCount does not match Ez trace count")

        positions_path = "/rxs/{0}/Positions".format(rx_name)
        if positions_path in fobj:
            positions = np.asarray(fobj[positions_path][:], dtype=np.float64)
            datasets["Positions"] = list(positions.shape)
            if positions.shape != (trace_count, 3):
                _add(messages, "error", "Positions shape must be traces x 3")
            elif trace_count > 1:
                dx_dy_dz = _as_float_list(attrs.get("dx_dy_dz"))
                rxsteps = _as_float_list(attrs.get("rxsteps"))
                expected_step = np.asarray(rxsteps[:3]) * np.asarray(dx_dy_dz[:3])
                actual_steps = np.diff(positions, axis=0)
                if not np.allclose(actual_steps, expected_step, rtol=0.0, atol=1e-9):
                    _add(messages, "error", "Positions do not match rxsteps * dx_dy_dz")
                else:
                    _add(messages, "ok", "Positions match rxsteps and grid spacing")
        elif trace_count > 1:
            _add(messages, "error", "merged B-scan is missing /rxs/{0}/Positions".format(rx_name))

        _check_manifest_summary(messages, manifest, attrs, datasets, rx_name, component)

    config = metadata.get("config", {}) if isinstance(metadata, dict) else {}
    if isinstance(config, dict):
        coverage = _coverage_from_config(config, positions)
        target_bounds = _target_bounds_from_config(config)
        if coverage and target_bounds:
            target_center = 0.5 * (target_bounds[0] + target_bounds[1])
            if target_center < coverage[0] or target_center > coverage[1]:
                _add(
                    messages,
                    "error",
                    "target center x={0:.6f} m is outside scan midpoint coverage {1:.6f}..{2:.6f} m".format(
                        target_center, coverage[0], coverage[1]
                    ),
                )
            elif target_bounds[0] < coverage[0] or target_bounds[1] > coverage[1]:
                _add(
                    messages,
                    "warning",
                    "target bounds {0:.6f}..{1:.6f} m are not fully covered by scan midpoint range {2:.6f}..{3:.6f} m".format(
                        target_bounds[0], target_bounds[1], coverage[0], coverage[1]
                    ),
                )
            else:
                _add(messages, "ok", "target is inside scan midpoint coverage")


def validate_dataset(
    output_dir: str,
    manifest_path: Optional[str] = None,
    component: str = "Ez",
    rx_name: str = "rx1",
) -> ValidationResult:
    output_dir = os.path.abspath(output_dir)
    messages = []
    manifest_path = _find_manifest(output_dir, manifest_path)
    if not os.path.exists(manifest_path):
        raise ValueError("manifest does not exist: {0}".format(manifest_path))

    manifest = _load_json(manifest_path)
    _add(messages, "ok", "manifest exists: {0}".format(manifest_path))
    if manifest.get("schema") == "uavgpr_manifest_v1":
        _add(messages, "ok", "manifest schema is uavgpr_manifest_v1")
    else:
        _add(messages, "error", "manifest schema must be uavgpr_manifest_v1")

    readiness = manifest.get("dataset_readiness")
    if not isinstance(readiness, dict):
        _add(messages, "error", "manifest dataset_readiness object is missing")
    elif not readiness.get("primary_out_file"):
        _add(messages, "error", "dataset_readiness.primary_out_file is not true")
    else:
        _add(messages, "ok", "dataset_readiness.primary_out_file is true")

    metadata = _read_metadata(manifest, output_dir)
    if metadata:
        _add(messages, "ok", "metadata JSON is readable")
    else:
        _add(messages, "warning", "metadata JSON was not found; target coverage is limited")

    component = str(manifest.get("component", component) or component)
    _, primary_out_path = _check_required_paths(messages, manifest, output_dir)
    _check_hdf5(messages, manifest, metadata, primary_out_path, component, rx_name)

    return ValidationResult(output_dir, manifest_path, messages)


def _print_result(result: ValidationResult) -> None:
    labels = {"ok": "OK", "warning": "WARNING", "error": "ERROR"}
    for message in result.messages:
        print("[{0}] {1}".format(labels.get(message.level, message.level), message.text))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate a UavGPR gprMax output directory."
    )
    parser.add_argument("output_dir", help="directory containing *_manifest.json")
    parser.add_argument("--manifest", help="manifest JSON path when multiple exist")
    parser.add_argument("--component", default="Ez", help="receiver dataset component")
    parser.add_argument("--rx", default="rx1", help="receiver group name")
    args = parser.parse_args(argv)

    try:
        result = validate_dataset(args.output_dir, args.manifest, args.component, args.rx)
    except Exception as exc:
        print("[ERROR] {0}".format(exc))
        return 1
    _print_result(result)
    return 1 if result.has_errors() else 0


if __name__ == "__main__":
    sys.exit(main())
