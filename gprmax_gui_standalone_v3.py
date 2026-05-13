#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
独立版可视化 gprMax 模拟数据生成器 GUI（增强版）

升级点：
1. 完全独立，不依赖旧的 landslide/crack 脚本
2. 内置两类模型：滑坡/空洞、线性裂缝
3. 深色界面、进度条、实时参数校验与网格规模估计
4. 支持单次生成、完整正演流程、批量数据集生成
5. 支持保存/加载 JSON 参数模板
6. 可选调用 gprMax 并自动绘制 B-scan
"""

from __future__ import annotations

import json
import math
import os
import queue
import subprocess
import sys
import threading
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox, ttk

_MISSING_DEPS = []
try:
    import h5py
except Exception:
    h5py = None
    _MISSING_DEPS.append("h5py")

try:
    import numpy as np
except Exception:
    np = None
    _MISSING_DEPS.append("numpy")

import matplotlib

matplotlib.use("Agg")
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None
    _MISSING_DEPS.append("matplotlib")

try:
    from matplotlib.path import Path as MplPath
except Exception:
    MplPath = None
    if "matplotlib" not in _MISSING_DEPS:
        _MISSING_DEPS.append("matplotlib")

APP_TITLE = "独立版 GPRMax 模拟数据生成器 GUI · 增强版"
WINDOW_SIZE = "1400x850"  # 减小窗口大小以适应更多屏幕


@dataclass
class GenerationArtifacts:
    output_dir: str = ""
    hdf5_path: str = ""
    materials_path: str = ""
    in_path: str = ""
    preview_path: str = ""
    bscan_path: str = ""
    merged_out_path: str = ""
    model_info: Dict[str, Any] = field(default_factory=dict)


class StandaloneModelBuilder:
    MAT_IDS = {
        "free_space": 0,
        "dry_soil": 1,
        "saturated_soil": 2,
        "rock": 3,
        "water_void": 4,
        "air_crack": 5,
        "water_crack": 6,
    }

    MATERIALS = {
        "free_space": (1.0, 0.0, 1.0, 0.0),
        "dry_soil": (6.0, 0.0, 1.0, 0.0),
        "saturated_soil": (25.0, 0.0, 1.0, 0.0),
        "rock": (5.0, 0.0, 1.0, 0.0),
        "water_void": (81.0, 0.0, 1.0, 0.0),
        "air_crack": (1.0, 0.0, 1.0, 0.0),
        "water_crack": (81.0, 0.0, 1.0, 0.0),
    }

    COLORS = {
        0: (248, 248, 248),
        1: (230, 196, 65),
        2: (184, 140, 55),
        3: (126, 42, 18),
        4: (35, 72, 196),
        5: (16, 16, 16),
        6: (0, 122, 255),
    }

    def __init__(
        self,
        logger: Optional[Callable[[str], None]] = None,
        progress: Optional[Callable[[float, str], None]] = None,
    ):
        self.logger = logger or (lambda msg: None)
        self.progress = progress or (lambda value, msg: None)

    def log(self, msg: str):
        self.logger(msg)

    def set_progress(self, value: float, msg: str):
        self.progress(max(0.0, min(1.0, value)), msg)

    @staticmethod
    def _timestamp_dir(base_dir: str, prefix: str, use_timestamp: bool = True) -> str:
        if use_timestamp:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = os.path.join(base_dir, f"{prefix}_{stamp}")
        else:
            out_dir = os.path.join(base_dir, prefix)
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    @staticmethod
    def _polygon_mask(points_xy: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
        ny, nx = shape
        yy, xx = np.mgrid[0:ny, 0:nx]
        coords = np.column_stack((xx.ravel(), yy.ravel()))
        path = MplPath(points_xy)
        return path.contains_points(coords).reshape(ny, nx)

    @staticmethod
    def _segment_distance_mask(
        nx: int, ny: int, x1: float, y1: float, x2: float, y2: float, radius: float
    ) -> np.ndarray:
        yy, xx = np.mgrid[0:ny, 0:nx]
        vx = x2 - x1
        vy = y2 - y1
        seg_len2 = vx * vx + vy * vy
        if seg_len2 < 1e-12:
            return (xx - x1) ** 2 + (yy - y1) ** 2 <= radius**2
        t = ((xx - x1) * vx + (yy - y1) * vy) / seg_len2
        t = np.clip(t, 0.0, 1.0)
        proj_x = x1 + t * vx
        proj_y = y1 + t * vy
        dist2 = (xx - proj_x) ** 2 + (yy - proj_y) ** 2
        return dist2 <= radius**2

    def build_landslide_model(
        self,
        output_dir: str,
        output_name: str,
        model_x: float,
        model_y: float,
        dx: float,
        dy: float,
        water_table_depth: float,
        src_start_x: float,
        surface_amplitude: float,
        surface_frequency: float,
        dry_rock_ratio: float,
        saturated_rock_ratio: float,
        rock_min_radius: float,
        rock_max_radius: float,
        void_center_x: float,
        void_center_y: float,
        void_radius_x: float,
        void_radius_y: float,
        n_traces: int,
        center_freq_mhz: float,
        time_window_ns: float,
        seed: Optional[int],
        use_timestamp_dir: bool,
    ) -> GenerationArtifacts:
        if seed is not None:
            np.random.seed(seed)
        self.set_progress(0.05, "初始化滑坡模型")

        out_dir = self._timestamp_dir(output_dir, output_name, use_timestamp_dir)
        nx = max(10, int(round(model_x / dx)))
        ny = max(10, int(round(model_y / dy)))
        model = np.zeros((ny, nx), dtype=np.int16)
        self.log(f"开始生成滑坡/空洞模型: 网格 {nx} x {ny}")

        self.set_progress(0.15, "生成地表与土层")
        x_indices = np.arange(nx)
        surface_base = int(ny * 0.85)
        surface_line = surface_base + (surface_amplitude / dy) * np.sin(
            2 * np.pi * surface_frequency * x_indices / max(nx, 1)
        )
        surface_line += np.random.normal(0, max(0.5, 0.01 / max(dy, 1e-9)), size=nx)
        surface_line = np.clip(surface_line, int(ny * 0.65), ny - 1).astype(int)

        water_table_grid = int(round(ny - water_table_depth / dy))
        water_table_grid = max(0, min(ny - 1, water_table_grid))
        for ix in range(nx):
            surface_idx = int(surface_line[ix])
            dry_soil_end = min(surface_idx, water_table_grid)
            if dry_soil_end > 0:
                model[0:dry_soil_end, ix] = self.MAT_IDS["saturated_soil"]
            if surface_idx > dry_soil_end:
                model[dry_soil_end:surface_idx, ix] = self.MAT_IDS["dry_soil"]

        self.set_progress(0.35, "填充岩石")

        def populate_rocks(
            y_min: int, y_max: int, target_ratio: float, part_weight: float
        ):
            if y_max <= y_min:
                return
            layer_area = nx * (y_max - y_min)
            target_area = int(layer_area * max(0.0, target_ratio))
            painted = 0
            attempts = 0
            max_attempts = max(240, target_area * 4)
            while painted < target_area and attempts < max_attempts:
                attempts += 1
                if attempts % 30 == 0:
                    frac = painted / max(target_area, 1)
                    self.set_progress(
                        0.35 + part_weight * frac, f"岩石填充 {painted}/{target_area}"
                    )
                radius_x = np.random.uniform(rock_min_radius / dx, rock_max_radius / dx)
                radius_y = np.random.uniform(rock_min_radius / dy, rock_max_radius / dy)
                margin_x = int(radius_x) + 2
                margin_y = int(radius_y) + 2
                if (nx - 2 * margin_x <= 0) or (y_max - y_min - 2 * margin_y <= 0):
                    break
                cx = np.random.randint(margin_x, nx - margin_x)
                cy = np.random.randint(y_min + margin_y, y_max - margin_y)
                pts = []
                n_pts = np.random.randint(6, 11)
                for ang in np.sort(np.random.uniform(0, 2 * np.pi, size=n_pts)):
                    scale = np.random.uniform(0.6, 1.0)
                    pts.append(
                        (
                            cx + radius_x * scale * np.cos(ang),
                            cy + radius_y * scale * np.sin(ang),
                        )
                    )
                polygon = np.asarray(pts, dtype=float)
                mask = self._polygon_mask(polygon, (ny, nx))
                mask[:y_min, :] = False
                mask[y_max:, :] = False
                if not np.any(mask):
                    continue
                touched = model[mask]
                if np.any(touched == self.MAT_IDS["free_space"]) or np.any(
                    touched == self.MAT_IDS["rock"]
                ):
                    continue
                model[mask] = self.MAT_IDS["rock"]
                painted += int(mask.sum())

        populate_rocks(0, water_table_grid, saturated_rock_ratio, 0.20)
        populate_rocks(
            water_table_grid, int(np.min(surface_line)), dry_rock_ratio, 0.20
        )

        self.set_progress(0.75, "生成空洞")
        cx = void_center_x / dx
        cy = void_center_y / dy
        rx = max(1.0, void_radius_x / dx)
        ry = max(1.0, void_radius_y / dy)
        yy, xx = np.mgrid[0:ny, 0:nx]
        void_mask = ((xx - cx) ** 2 / (rx**2) + (yy - cy) ** 2 / (ry**2)) <= 1.0
        void_mask &= model != self.MAT_IDS["free_space"]
        model[void_mask] = self.MAT_IDS["water_void"]

        used_materials = ["dry_soil", "saturated_soil", "rock", "water_void"]
        self.set_progress(0.88, "导出模型文件")
        artifacts = self._export_all(
            out_dir=out_dir,
            output_name=output_name,
            model=model,
            model_x=model_x,
            model_y=model_y,
            dx=dx,
            dy=dy,
            water_table_depth=water_table_depth,
            n_traces=n_traces,
            center_freq_mhz=center_freq_mhz,
            time_window_ns=time_window_ns,
            src_start_x=src_start_x,
            source_materials=used_materials,
            preview_title="GPR Landslide / Void Model",
            extra_lines=[
                (
                    "water_table",
                    model_y - water_table_depth,
                    "cyan",
                    "--",
                    "Water Table",
                )
            ],
            model_info={
                "type": "landslide",
                "nx": nx,
                "ny": ny,
                "water_table_depth": water_table_depth,
                "seed": seed,
                "cells": int(nx * ny),
            },
        )
        self.set_progress(1.0, "滑坡模型生成完成")
        self.log(f"滑坡模型生成完成: {artifacts.output_dir}")
        return artifacts

    def build_crack_model(
        self,
        output_dir: str,
        output_name: str,
        model_x: float,
        model_y: float,
        dx: float,
        dy: float,
        water_table_depth: float,
        src_start_x: float,
        crack_type: str = "linear",
        crack_start_x: float = 1.0,
        crack_start_y: float = 0.5,
        crack_end_x: float = 4.0,
        crack_end_y: float = 1.5,
        crack_width: float = 0.03,
        crack_fill: str = "air",
        crack_angle: float = 45,
        crack_count: int = 3,
        crack_curvature: float = 0.5,
        n_traces: int = 5,
        center_freq_mhz: float = 400,
        time_window_ns: float = 40,
        seed: Optional[int] = None,
        use_timestamp_dir: bool = True,
    ) -> GenerationArtifacts:
        if seed is not None:
            np.random.seed(seed)
        self.set_progress(0.05, "初始化裂缝模型")

        out_dir = self._timestamp_dir(output_dir, output_name, use_timestamp_dir)
        nx = max(10, int(round(model_x / dx)))
        ny = max(10, int(round(model_y / dy)))
        model = np.zeros((ny, nx), dtype=np.int16)
        self.log(f"开始生成裂缝模型: 网格 {nx} x {ny}, 类型: {crack_type}")
        self.log(
            f"裂缝参数: start=({crack_start_x}, {crack_start_y}) end=({crack_end_x}, {crack_end_y})"
        )

        self.set_progress(0.20, "填充背景土层")
        surface_line = np.full(nx, int(ny * 0.85), dtype=int)
        water_table_grid = int(round(ny - water_table_depth / dy))
        water_table_grid = max(0, min(ny - 1, water_table_grid))
        for ix in range(nx):
            s = int(surface_line[ix])
            mid = min(s, water_table_grid)
            if mid > 0:
                model[0:mid, ix] = self.MAT_IDS["saturated_soil"]
            if s > mid:
                model[mid:s, ix] = self.MAT_IDS["dry_soil"]

        self.set_progress(0.55, f"生成{crack_type}裂缝")
        crack_id = (
            self.MAT_IDS["air_crack"]
            if crack_fill == "air"
            else self.MAT_IDS["water_crack"]
        )

        # 根据裂缝类型生成不同的裂缝形态
        if crack_type == "linear":
            # 线性裂缝
            gx1, gy1 = crack_start_x / dx, crack_start_y / dy
            gx2, gy2 = crack_end_x / dx, crack_end_y / dy
            radius = max(1.0, crack_width / max(dx, dy) / 2.0)
            crack_mask = self._segment_distance_mask(nx, ny, gx1, gy1, gx2, gy2, radius)
            model[crack_mask & (model != self.MAT_IDS["free_space"])] = crack_id

        elif crack_type == "oblique":
            # 斜裂缝 - 带角度的裂缝
            gx1, gy1 = crack_start_x / dx, crack_start_y / dy
            gx2, gy2 = crack_end_x / dx, crack_end_y / dy
            radius = max(1.0, crack_width / max(dx, dy) / 2.0)
            crack_mask = self._segment_distance_mask(nx, ny, gx1, gy1, gx2, gy2, radius)
            model[crack_mask & (model != self.MAT_IDS["free_space"])] = crack_id

        elif crack_type == "curved":
            # 弧形裂缝
            center_x = (crack_start_x + crack_end_x) / 2 / dx
            center_y = (crack_start_y + crack_end_y) / 2 / dy
            radius = crack_curvature * min(nx, ny) / 2
            yy, xx = np.mgrid[0:ny, 0:nx]
            crack_mask = ((xx - center_x) ** 2 + (yy - center_y) ** 2) <= radius**2
            crack_mask &= model != self.MAT_IDS["free_space"]
            model[crack_mask] = crack_id

        elif crack_type == "network":
            # 网状裂缝 - 多条交叉裂缝
            for i in range(crack_count):
                offset = (i - crack_count // 2) * (model_x / crack_count) * 0.3
                gx1 = crack_start_x / dx
                gy1 = crack_start_y / dy + offset / dy
                gx2 = crack_end_x / dx + offset / dx
                gy2 = crack_end_y / dy + offset / dy
                radius = max(1.0, crack_width / max(dx, dy) / 2.0)
                crack_mask = self._segment_distance_mask(
                    nx, ny, gx1, gy1, gx2, gy2, radius
                )
                model[crack_mask & (model != self.MAT_IDS["free_space"])] = crack_id

        elif crack_type == "multiple":
            # 多条平行裂缝
            spacing = model_y / (crack_count + 1)
            for i in range(crack_count):
                offset_y = (i + 1) * spacing
                gx1, gy1 = crack_start_x / dx, (crack_start_y + offset_y) / dy
                gx2, gy2 = crack_end_x / dx, (crack_end_y + offset_y) / dy
                radius = max(1.0, crack_width / max(dx, dy) / 2.0)
                crack_mask = self._segment_distance_mask(
                    nx, ny, gx1, gy1, gx2, gy2, radius
                )
                # 限制在模型范围内
                gx1 = max(0, min(nx - 1, gx1))
                gx2 = max(0, min(nx - 1, gx2))
                gy1 = max(0, min(ny - 1, gy1))
                gy2 = max(0, min(ny - 1, gy2))
                crack_mask = self._segment_distance_mask(
                    nx, ny, gx1, gy1, gx2, gy2, radius
                )
                model[crack_mask & (model != self.MAT_IDS["free_space"])] = crack_id
        else:
            # 默认线性裂缝
            gx1, gy1 = crack_start_x / dx, crack_start_y / dy
            gx2, gy2 = crack_end_x / dx, crack_end_y / dy
            radius = max(1.0, crack_width / max(dx, dy) / 2.0)
            crack_mask = self._segment_distance_mask(nx, ny, gx1, gy1, gx2, gy2, radius)
            model[crack_mask & (model != self.MAT_IDS["free_space"])] = crack_id

        used_materials = [
            "dry_soil",
            "saturated_soil",
            "air_crack" if crack_fill == "air" else "water_crack",
        ]
        self.set_progress(0.88, "导出模型文件")
        artifacts = self._export_all(
            out_dir=out_dir,
            output_name=output_name,
            model=model,
            model_x=model_x,
            model_y=model_y,
            dx=dx,
            dy=dy,
            water_table_depth=water_table_depth,
            n_traces=n_traces,
            center_freq_mhz=center_freq_mhz,
            time_window_ns=time_window_ns,
            src_start_x=src_start_x,
            source_materials=used_materials,
            preview_title=f"GPR Crack Model ({crack_fill})",
            crack_line=((crack_start_x, crack_start_y), (crack_end_x, crack_end_y)),
            extra_lines=[
                (
                    "water_table",
                    model_y - water_table_depth,
                    "cyan",
                    "--",
                    "Water Table",
                )
            ],
            model_info={
                "type": "crack",
                "nx": nx,
                "ny": ny,
                "crack_fill": crack_fill,
                "seed": seed,
                "cells": int(nx * ny),
            },
        )
        self.set_progress(1.0, "裂缝模型生成完成")
        self.log(f"裂缝模型生成完成: {artifacts.output_dir}")
        return artifacts

    def _export_all(
        self,
        out_dir: str,
        output_name: str,
        model: np.ndarray,
        model_x: float,
        model_y: float,
        dx: float,
        dy: float,
        water_table_depth: float,
        n_traces: int,
        center_freq_mhz: float,
        time_window_ns: float,
        src_start_x: float,
        source_materials: list[str],
        preview_title: str,
        crack_line: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
        extra_lines: Optional[list[Tuple[str, float, str, str, str]]] = None,
        model_info: Optional[Dict[str, Any]] = None,
    ) -> GenerationArtifacts:
        h5_path = os.path.join(out_dir, f"{output_name}.h5")
        materials_path = os.path.join(out_dir, f"{output_name}_materials.txt")
        in_path = os.path.join(out_dir, f"{output_name}.in")
        preview_path = os.path.join(out_dir, f"{output_name}.png")

        with h5py.File(h5_path, "w") as f:
            f.create_dataset(
                "data", data=model[np.newaxis, :, :].astype(np.int16), dtype=np.int16
            )
            f.attrs["dx_dy_dz"] = (dx, dy, dx)
            f.attrs["created_by"] = "gprmax_gui_standalone_v2"
            f.attrs["created_at"] = datetime.now().isoformat()

        with open(materials_path, "w", encoding="utf-8") as f:
            # 确保材料从free_space开始按ID顺序定义
            all_materials = [
                "free_space",
                "dry_soil",
                "saturated_soil",
                "rock",
                "water_void",
                "air_crack",
                "water_crack",
            ]
            for name in all_materials:
                if name in self.MATERIALS:
                    eps_r, sigma, mu_r, sigma_star = self.MATERIALS[name]
                    f.write(f"#material: {eps_r} {sigma} {mu_r} {sigma_star} {name}\n")

        src_steps = self._calc_src_steps(model_x, n_traces, dx, src_start_x)
        center_freq_hz = center_freq_mhz * 1e6
        time_window_s = time_window_ns * 1e-9

        input_lines = [
            f"#title: {preview_title}",
            f"#domain: {model_x} {model_y} {dx}",
            f"#dx_dy_dz: {dx} {dy} {dx}",
            f"#time_window: {time_window_s}",
            "",
        ]
        # free_space 是 gprMax 内置材料，不要在 .in 文件中重复定义
        all_materials = [
            "dry_soil",
            "saturated_soil",
            "rock",
            "water_void",
            "air_crack",
            "water_crack",
        ]
        for name in all_materials:
            if name in self.MATERIALS:
                eps_r, sigma, mu_r, sigma_star = self.MATERIALS[name]
                input_lines.append(
                    f"#material: {eps_r} {sigma} {mu_r} {sigma_star} {name}"
                )

        # 源/接收器位置：在地表（HDF5 中地表在 ny*0.85 行）
        surface_y = model_y * 0.85
        rx_offset = 0.04  # 收发间距 4cm

        input_lines += [
            "",
            f"#geometry_objects_read: 0 0 0 {os.path.basename(h5_path)} {os.path.basename(materials_path)}",
            "",
            f"#waveform: ricker 1 {center_freq_hz} wave1",
            f"#hertzian_dipole: z {src_start_x} {surface_y} 0 wave1",
            f"#rx: {src_start_x + rx_offset} {surface_y} 0",
            f"#src_steps: {src_steps} 0 0",
            f"#rx_steps: {src_steps} 0 0",
        ]
        with open(in_path, "w", encoding="utf-8") as f:
            f.write("\n".join(input_lines) + "\n")

        self._save_preview_png(
            model=model,
            out_path=preview_path,
            model_x=model_x,
            model_y=model_y,
            title=preview_title,
            crack_line=crack_line,
            extra_lines=extra_lines or [],
        )
        return GenerationArtifacts(
            output_dir=out_dir,
            hdf5_path=h5_path,
            materials_path=materials_path,
            in_path=in_path,
            preview_path=preview_path,
            model_info=model_info or {},
        )

    def _save_preview_png(
        self,
        model: np.ndarray,
        out_path: str,
        model_x: float,
        model_y: float,
        title: str,
        crack_line: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
        extra_lines: Optional[list[Tuple[str, float, str, str, str]]] = None,
    ) -> None:
        ny, nx = model.shape
        rgb = np.zeros((ny, nx, 3), dtype=np.uint8)
        for mid, color in self.COLORS.items():
            rgb[model == mid] = color

        fig, ax = plt.subplots(figsize=(10, 4.2))
        ax.imshow(rgb, origin="lower", aspect="auto", extent=[0, model_x, 0, model_y])
        if extra_lines:
            for _, y, color, linestyle, label in extra_lines:
                ax.axhline(
                    y=y, color=color, linestyle=linestyle, linewidth=1.5, label=label
                )
        if crack_line is not None:
            (x1, y1), (x2, y2) = crack_line
            ax.plot([x1, x2], [y1, y2], color="red", linewidth=2.3, label="Crack")
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            unique = dict(zip(labels, handles))
            ax.legend(unique.values(), unique.keys(), loc="upper right")
        ax.set_title(title)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y / Depth (m)")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close(fig)

    @staticmethod
    def _calc_src_steps(
        model_x: float, n_traces: int, dx: float = 0.005, src_start_x: float = 0.05
    ) -> float:
        """计算源步长，确保不越界

        Args:
            model_x: 模型 X 方向尺寸 (m)
            n_traces: 总道数
            dx: 网格尺寸 (m)
            src_start_x: 源起始 X 位置 (m)

        Returns:
            步长 (m)
        """
        if n_traces <= 1:
            return 0.1

        # 安全边界 0.10m
        # 最大安全终点 = model_x - 0.10
        # 可用距离 = model_x - src_start_x - 0.10
        available = model_x - src_start_x - 0.10

        if available <= 0:
            return 0.001

        # 计算理想步长
        ideal_step = available / (n_traces - 1)

        # 尝试对齐到网格（向下取整到 dx 的整数倍）
        grids = max(1, int(ideal_step / dx))
        aligned_step = grids * dx

        # 验证对齐后的步长是否安全
        end_pos = src_start_x + (n_traces - 1) * aligned_step
        if end_pos <= model_x - 0.10:
            return aligned_step  # 安全，返回对齐步长

        # 对齐步长不安全，使用理想步长（不强求对齐）
        return ideal_step

    def plot_gprmax_bscan(self, data_file_path: str, output_png_path: str) -> str:
        folder = os.path.dirname(data_file_path)
        out_files = sorted(
            [f for f in os.listdir(folder) if f.endswith(".out") and "merged" not in f]
        )

        if out_files:
            traces = []
            dt = None
            for fname in out_files:
                fpath = os.path.join(folder, fname)
                with h5py.File(fpath, "r") as f:
                    traces.append(f["rxs"]["rx1"]["Ez"][:])
                    if dt is None:
                        dt = float(f.attrs.get("dt", 1.0))
            data = np.stack(traces, axis=1)
        else:
            with h5py.File(data_file_path, "r") as f:
                dt = float(f.attrs.get("dt", 1.0))
                data = np.array(f["rxs"]["rx1"]["Ez"][:], dtype=np.float32)
                if data.ndim == 1:
                    data = data[:, np.newaxis]

        samples, traces_n = data.shape
        time_axis = np.arange(samples, dtype=float) * dt
        vmax = max(1e-12, np.max(np.abs(data)) * 0.30)

        fig, ax = plt.subplots(figsize=(11, 7))
        im = ax.imshow(
            data,
            cmap="seismic",
            aspect="auto",
            vmin=-vmax,
            vmax=vmax,
            origin="upper",
            extent=[0, traces_n, samples * dt * 1e9, 0],
        )
        ax.set_title("GPR B-scan")
        ax.set_xlabel("Trace Number")
        ax.set_ylabel("Time (ns)")
        plt.colorbar(im, ax=ax, label="Amplitude")
        plt.tight_layout()
        plt.savefig(output_png_path, dpi=150)
        plt.close(fig)
        return output_png_path


class GPRMaxStandaloneGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry(WINDOW_SIZE)
        self.minsize(1200, 700)  # 降低最小尺寸要求
        self.configure(bg="#16181d")

        # 居中显示窗口
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f"{width}x{height}+{x}+{y}")

        self.log_queue: queue.Queue[str] = queue.Queue()
        self.image_cache = None
        self.last_artifacts = GenerationArtifacts()
        self.worker: Optional[threading.Thread] = None
        self._live_update_job: Optional[str] = None

        self._build_variables()
        self.builder = StandaloneModelBuilder(
            logger=self.log, progress=self._threadsafe_progress
        )
        self._build_style()
        self._build_layout()
        self._bind_live_updates()
        self._poll_log_queue()
        self._refresh_live_info()
        self.log("已启动增强版 GUI。当前为独立版，不依赖旧脚本。")

    def _build_variables(self):
        self.model_type_var = tk.StringVar(value="landslide")
        self.output_dir_var = tk.StringVar(value=r"D:\ClawX-Data\sim\gprmax_outcsv")
        self.output_name_var = tk.StringVar(value="gpr_model")
        self.seed_var = tk.StringVar(value="42")
        self.n_traces_var = tk.StringVar(value="70")
        self.center_freq_var = tk.StringVar(value="1500")
        self.time_window_var = tk.StringVar(value="3")
        # 使用 gprMax 虚拟环境中的 Python
        self.python_var = tk.StringVar(
            value=r"E:\gprMax\gprMax-v.3.1.7\.venv\Scripts\python.exe"
        )
        self.use_timestamp_var = tk.BooleanVar(value=True)
        self.auto_run_var = tk.BooleanVar(value=False)
        self.theme_var = tk.StringVar(value="dark")
        self.preset_var = tk.StringVar(value="balanced")
        self.progress_var = tk.DoubleVar(value=0.0)
        self.status_var = tk.StringVar(value="就绪")
        self.estimate_var = tk.StringVar(value="")
        self.validation_var = tk.StringVar(value="")
        self.batch_count_var = tk.StringVar(value="10")
        self.batch_seed_step_var = tk.StringVar(value="1")
        self.batch_start_idx_var = tk.StringVar(value="0")
        self.batch_run_var = tk.BooleanVar(value=False)
        self.use_gpu_var = tk.BooleanVar(value=False)

        self.common_vars = {
            "model_x": tk.StringVar(value="0.24"),
            "model_y": tk.StringVar(value="0.21"),
            "dx": tk.StringVar(value="0.002"),
            "dy": tk.StringVar(value="0.002"),
            "water_table_depth": tk.StringVar(value="0.17"),
            "src_start_x": tk.StringVar(value="0.04"),
        }

        self.landslide_vars = {
            "surface_amplitude": tk.StringVar(value="0.03"),
            "surface_frequency": tk.StringVar(value="1.5"),
            "dry_rock_ratio": tk.StringVar(value="0.03"),
            "saturated_rock_ratio": tk.StringVar(value="0.10"),
            "rock_min_radius": tk.StringVar(value="0.05"),
            "rock_max_radius": tk.StringVar(value="0.25"),
            "void_center_x": tk.StringVar(value="0.12"),
            "void_center_y": tk.StringVar(value="0.08"),
            "void_radius_x": tk.StringVar(value="0.01"),
            "void_radius_y": tk.StringVar(value="0.01"),
        }

        self.landslide_vars = {
            "surface_amplitude": tk.StringVar(value="0.03"),
            "surface_frequency": tk.StringVar(value="1.5"),
            "dry_rock_ratio": tk.StringVar(value="0.03"),
            "saturated_rock_ratio": tk.StringVar(value="0.10"),
            "rock_min_radius": tk.StringVar(value="0.05"),
            "rock_max_radius": tk.StringVar(value="0.25"),
            "void_center_x": tk.StringVar(value="2.5"),
            "void_center_y": tk.StringVar(value="1.2"),
            "void_radius_x": tk.StringVar(value="0.20"),
            "void_radius_y": tk.StringVar(value="0.15"),
        }

        self.crack_vars = {
            "crack_type": tk.StringVar(value="linear"),  # 新增：裂缝类型
            "crack_start_x": tk.StringVar(value="1.0"),
            "crack_start_y": tk.StringVar(value="0.5"),
            "crack_end_x": tk.StringVar(value="4.0"),
            "crack_end_y": tk.StringVar(value="1.5"),
            "crack_width": tk.StringVar(value="0.03"),
            "crack_fill": tk.StringVar(value="air"),
            # 扩展参数
            "crack_angle": tk.StringVar(value="45"),  # 斜裂缝角度
            "crack_count": tk.StringVar(value="3"),  # 网状/多条裂缝数量
            "crack_curvature": tk.StringVar(value="0.5"),  # 弧形曲率
        }

    def _pick_font_family(self) -> str:
        preferred = ["Microsoft YaHei UI", "Microsoft YaHei", "Segoe UI", "Arial"]
        try:
            families = set(tkfont.families(self))
        except Exception:
            families = set()
        for name in preferred:
            if name in families:
                return name
        return "TkDefaultFont"

    def _build_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass

        bg = "#16181d"
        bg2 = "#20242b"
        bg3 = "#2b313a"
        fg = "#e7ebf0"
        mutefg = "#aab3bf"
        accent = "#4c8dff"

        self._theme_colors = {
            "bg": bg,
            "bg2": bg2,
            "bg3": bg3,
            "fg": fg,
            "mutefg": mutefg,
            "accent": accent,
        }

        family = self._pick_font_family()
        try:
            default_font = tkfont.nametofont("TkDefaultFont")
            default_font.configure(family=family, size=10)
            text_font = tkfont.nametofont("TkTextFont")
            text_font.configure(family=family, size=10)
            fixed_font = tkfont.nametofont("TkFixedFont")
            fixed_font.configure(size=10)
        except Exception:
            pass

        title_font = tkfont.Font(self, family=family, size=15, weight="bold")
        section_font = tkfont.Font(self, family=family, size=10, weight="bold")

        self.option_add("*TCombobox*Listbox.background", bg2)
        self.option_add("*TCombobox*Listbox.foreground", fg)

        style.configure("TFrame", background=bg)
        style.configure("TLabel", background=bg, foreground=fg)
        style.configure("Muted.TLabel", background=bg, foreground=mutefg)
        style.configure("Title.TLabel", background=bg, foreground=fg, font=title_font)
        style.configure(
            "Section.TLabelframe", background=bg, foreground=fg, borderwidth=1
        )
        style.configure(
            "Section.TLabelframe.Label", background=bg, foreground=fg, font=section_font
        )
        style.configure("TCheckbutton", background=bg, foreground=fg)
        style.configure("TRadiobutton", background=bg, foreground=fg)
        style.configure("TNotebook", background=bg)
        style.configure("TNotebook.Tab", background=bg3, foreground=fg, padding=(10, 5))
        style.map("TNotebook.Tab", background=[("selected", bg2)])
        style.configure("TEntry", fieldbackground=bg2, foreground=fg, insertcolor=fg)
        style.configure("TCombobox", fieldbackground=bg2, foreground=fg, background=bg3)
        style.configure(
            "TButton", background=bg3, foreground=fg, borderwidth=0, padding=(10, 6)
        )
        style.map("TButton", background=[("active", accent), ("pressed", accent)])
        style.configure(
            "Primary.TButton", background=accent, foreground="#ffffff", padding=(10, 7)
        )
        style.map(
            "Primary.TButton",
            background=[("active", "#6ca1ff"), ("pressed", "#3f7ff0")],
        )
        style.configure(
            "TProgressbar",
            troughcolor=bg3,
            background=accent,
            bordercolor=bg3,
            lightcolor=accent,
            darkcolor=accent,
        )

    def _build_layout(self):
        root = ttk.Frame(self, padding=8)
        root.pack(fill="both", expand=True)

        # 使用三栏布局: 左(快捷操作) + 中(参数设置) + 右(预览和日志)
        root.columnconfigure(0, weight=0, minsize=260)  # 左侧：快捷操作
        root.columnconfigure(1, weight=1, minsize=380)  # 中间：参数设置
        root.columnconfigure(2, weight=1, minsize=400)  # 右侧：预览+日志
        root.rowconfigure(0, weight=1)

        # 左侧面板 - 快捷操作和基本设置
        left_frame = ttk.Frame(root, padding=8)
        left_frame.grid(row=0, column=0, sticky="ns", padx=(0, 8))
        left_frame.rowconfigure(0, weight=0)
        left_frame.rowconfigure(1, weight=0)
        left_frame.rowconfigure(2, weight=1)
        left_frame.rowconfigure(3, weight=0)

        # 中间面板 - 详细参数设置
        mid_frame = ttk.Frame(root, padding=8)
        mid_frame.grid(row=0, column=1, sticky="nsew", padx=(0, 8))
        self._build_mid_panel(mid_frame)

        # 右侧面板 - 预览和日志
        right_frame = ttk.Frame(root, padding=8)
        right_frame.grid(row=0, column=2, sticky="nsew")
        self._build_right_panel(right_frame)

        self._build_left_panel(left_frame)

    def _build_mid_panel(self, parent):
        """中间面板 - 详细参数设置（带滚动条）"""
        canvas = tk.Canvas(parent, bg="#16181d", highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas, padding=4)

        scrollable_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # 使用 scrollable_frame 作为实际容器
        cont = scrollable_frame
        cont.columnconfigure(0, weight=1)

        # 模型基本参数
        model_box = ttk.LabelFrame(
            cont, text="📐 模型参数", style="Section.TLabelframe", padding=8
        )
        model_box.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        model_box.columnconfigure(1, weight=1)

        rows_params = [
            ("模型宽度 (m)", "model_x", "10.0"),
            ("模型高度 (m)", "model_y", "4.0"),
            ("网格 dx (m)", "dx", "0.005"),
            ("网格 dy (m)", "dy", "0.005"),
            ("潜水面深度 (m)", "water_table_depth", "1.5"),
        ]
        for i, (label, key, default) in enumerate(rows_params):
            ttk.Label(model_box, text=label, width=14).grid(
                row=i, column=0, sticky="w", padx=(0, 8)
            )
            ent = ttk.Entry(model_box, textvariable=self.common_vars[key], width=14)
            ent.grid(row=i, column=1, sticky="ew", pady=2)

        # 扫描参数
        scan_box = ttk.LabelFrame(
            cont, text="📡 扫描参数", style="Section.TLabelframe", padding=8
        )
        scan_box.grid(row=1, column=0, sticky="ew", pady=(0, 6))
        scan_box.columnconfigure(1, weight=1)

        scan_params = [
            ("扫描道数", "n_traces", "5"),
            ("中心频率 (MHz)", "center_freq_mhz", "400"),
            ("时间窗 (ns)", "time_window_ns", "40"),
            ("起始 X 位置 (m)", "src_start_x", "0.05"),
        ]
        for i, (label, key, default) in enumerate(scan_params):
            var_key = key.replace("_mhz", "_var").replace("_ns", "_var")
            if key == "n_traces":
                var = self.n_traces_var
            elif key == "center_freq_mhz":
                var = self.center_freq_var
            elif key == "time_window_ns":
                var = self.time_window_var
            elif key == "src_start_x":
                var = self.common_vars["src_start_x"]
            else:
                var = tk.StringVar(value=default)

            ttk.Label(scan_box, text=label, width=14).grid(
                row=i, column=0, sticky="w", padx=(0, 8)
            )
            ent = ttk.Entry(scan_box, textvariable=var, width=14)
            ent.grid(row=i, column=1, sticky="ew", pady=2)

        # 滑坡模型参数
        landslide_box = ttk.LabelFrame(
            cont, text="⛰️ 滑坡/空洞参数", style="Section.TLabelframe", padding=8
        )
        landslide_box.grid(row=2, column=0, sticky="ew", pady=(0, 6))
        landslide_box.columnconfigure(1, weight=1)

        ls_params = [
            ("地表起伏幅度", "surface_amplitude", "0.03"),
            ("地表起伏频率", "surface_frequency", "1.5"),
            ("干土层含石率", "dry_rock_ratio", "0.05"),
            ("饱水层含石率", "saturated_rock_ratio", "0.15"),
            ("最小石块半径", "rock_min_radius", "0.05"),
            ("最大石块半径", "rock_max_radius", "0.25"),
            ("空洞中心 X", "void_center_x", "5.0"),
            ("空洞中心 Y", "void_center_y", "2.5"),
            ("空洞半径 X", "void_radius_x", "0.3"),
            ("空洞半径 Y", "void_radius_y", "0.2"),
        ]

        for i, (label, key, default) in enumerate(ls_params):
            ttk.Label(landslide_box, text=label, width=14).grid(
                row=i // 2, column=(i % 2) * 2, sticky="w", padx=(0, 4)
            )
            ent = ttk.Entry(
                landslide_box, textvariable=self.landslide_vars[key], width=10
            )
            ent.grid(
                row=i // 2, column=(i % 2) * 2 + 1, sticky="ew", pady=2, padx=(0, 8)
            )

        # 裂缝模型参数
        crack_box = ttk.LabelFrame(
            cont, text="🔪 裂缝参数", style="Section.TLabelframe", padding=8
        )
        crack_box.grid(row=3, column=0, sticky="ew", pady=(0, 6))
        crack_box.columnconfigure(1, weight=1)

        # 裂缝类型选择
        ttk.Label(crack_box, text="裂缝类型:").grid(
            row=0, column=0, sticky="w", padx=(0, 4)
        )
        crack_type_combo = ttk.Combobox(
            crack_box,
            textvariable=self.crack_vars["crack_type"],
            values=["linear", "oblique", "curved", "network", "multiple"],
            state="readonly",
            width=12,
        )
        crack_type_combo.grid(row=0, column=1, sticky="ew", pady=2)

        crack_params = [
            ("起点 X", "crack_start_x", "1.0"),
            ("起点 Y", "crack_start_y", "0.5"),
            ("终点 X", "crack_end_x", "4.0"),
            ("终点 Y", "crack_end_y", "1.5"),
            ("裂缝宽度", "crack_width", "0.03"),
            ("角度(°)", "crack_angle", "45"),  # 斜裂缝角度
            ("数量", "crack_count", "3"),  # 网状/多条数量
            ("曲率", "crack_curvature", "0.5"),  # 弧形曲率
        ]

        for i, (label, key, default) in enumerate(crack_params):
            ttk.Label(crack_box, text=label, width=10).grid(
                row=i + 1, column=0, sticky="w", padx=(0, 4)
            )
            ent = ttk.Entry(crack_box, textvariable=self.crack_vars[key], width=12)
            ent.grid(row=i + 1, column=1, sticky="ew", pady=2)

        # 批量生成设置
        batch_box = ttk.LabelFrame(
            cont, text="📦 批量生成", style="Section.TLabelframe", padding=8
        )
        batch_box.grid(row=4, column=0, sticky="ew", pady=(0, 6))
        batch_box.columnconfigure(1, weight=1)

        ttk.Label(batch_box, text="样本数量", width=12).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Entry(batch_box, textvariable=self.batch_count_var, width=12).grid(
            row=0, column=1, sticky="ew"
        )

        ttk.Label(batch_box, text="种子步长", width=12).grid(
            row=1, column=0, sticky="w"
        )
        ttk.Entry(batch_box, textvariable=self.batch_seed_step_var, width=12).grid(
            row=1, column=1, sticky="ew"
        )

        ttk.Checkbutton(
            batch_box, text="批量时自动运行gprMax", variable=self.batch_run_var
        ).grid(row=2, column=0, columnspan=2, sticky="w")

    def _build_left_panel(self, parent):
        """左侧面板 - 快捷操作和状态"""
        # 标题
        title_label = ttk.Label(
            parent,
            text="GPRMax 数据生成器",
            style="Title.TLabel",
            font=("微软雅黑", 14, "bold"),
        )
        title_label.grid(row=0, column=0, sticky="w", pady=(0, 12))

        # 快速设置
        quick_box = ttk.LabelFrame(
            parent, text="⚡ 快速设置", style="Section.TLabelframe", padding=8
        )
        quick_box.grid(row=1, column=0, sticky="ew", pady=(0, 8))

        # 参数预设
        ttk.Label(quick_box, text="参数预设:").grid(row=0, column=0, sticky="w", pady=2)
        preset_combo = ttk.Combobox(
            quick_box,
            textvariable=self.preset_var,
            state="readonly",
            values=["fast", "balanced", "highres", "示例-金属圆柱体"],
            width=15,
        )
        preset_combo.grid(row=0, column=1, sticky="ew", pady=2)
        ttk.Button(quick_box, text="应用", command=self.on_apply_preset, width=8).grid(
            row=0, column=2, padx=(4, 0)
        )

        # 模型类型
        ttk.Label(quick_box, text="模型类型:").grid(row=1, column=0, sticky="w", pady=2)
        type_box = ttk.Frame(quick_box)
        type_box.grid(row=1, column=1, columnspan=2, sticky="w")
        ttk.Radiobutton(
            type_box, text="滑坡/空洞", variable=self.model_type_var, value="landslide"
        ).pack(side="left")
        ttk.Radiobutton(
            type_box, text="裂缝", variable=self.model_type_var, value="crack"
        ).pack(side="left", padx=(8, 0))

        # GPU 加速选项
        ttk.Checkbutton(
            quick_box, text="启用 GPU 加速", variable=self.use_gpu_var
        ).grid(row=2, column=0, columnspan=3, sticky="w", pady=(4, 0))

        # 输出设置
        output_box = ttk.LabelFrame(
            parent, text="📁 输出设置", style="Section.TLabelframe", padding=8
        )
        output_box.grid(row=2, column=0, sticky="ew", pady=(0, 8))

        ttk.Label(output_box, text="输出目录:").grid(
            row=0, column=0, sticky="w", pady=2
        )
        ttk.Entry(output_box, textvariable=self.output_dir_var, width=20).grid(
            row=0, column=1, columnspan=2, sticky="ew"
        )
        ttk.Button(
            output_box,
            text="浏览",
            width=6,
            command=lambda: self._browse_dir(self.output_dir_var),
        ).grid(row=0, column=3, padx=(4, 0))

        ttk.Label(output_box, text="文件名前缀:").grid(
            row=1, column=0, sticky="w", pady=2
        )
        ttk.Entry(output_box, textvariable=self.output_name_var, width=20).grid(
            row=1, column=1, columnspan=2, sticky="ew"
        )

        ttk.Label(output_box, text="随机种子:").grid(
            row=2, column=0, sticky="w", pady=2
        )
        ttk.Entry(output_box, textvariable=self.seed_var, width=20).grid(
            row=2, column=1, columnspan=2, sticky="ew"
        )

        ttk.Checkbutton(
            output_box, text="目录加时间戳", variable=self.use_timestamp_var
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=4)
        ttk.Checkbutton(
            output_box, text="生成后自动运行gprMax", variable=self.auto_run_var
        ).grid(row=4, column=0, columnspan=2, sticky="w")

        # 操作按钮
        action_box = ttk.LabelFrame(
            parent, text="🎮 操作", style="Section.TLabelframe", padding=8
        )
        action_box.grid(row=3, column=0, sticky="ew", pady=(0, 8))

        ttk.Button(
            action_box,
            text="✨ 生成模型",
            style="Primary.TButton",
            command=self.on_generate,
        ).pack(fill="x", pady=3)
        ttk.Button(action_box, text="🚀 运行完整流程", command=self.on_run_full).pack(
            fill="x", pady=3
        )
        ttk.Button(
            action_box, text="📦 批量生成数据集", command=self.on_batch_generate
        ).pack(fill="x", pady=3)
        ttk.Button(action_box, text="💾 保存参数", command=self.on_save_params).pack(
            fill="x", pady=3
        )
        ttk.Button(action_box, text="📂 加载参数", command=self.on_load_params).pack(
            fill="x", pady=3
        )
        ttk.Button(
            action_box, text="📂 打开输出目录", command=self.on_open_output_dir
        ).pack(fill="x", pady=3)

        # 进度状态
        status_box = ttk.LabelFrame(
            parent, text="📊 状态", style="Section.TLabelframe", padding=8
        )
        status_box.grid(row=4, column=0, sticky="ew", pady=(0, 8))

        self.status_var.set("就绪")
        ttk.Label(status_box, textvariable=self.status_var, wraplength=220).pack(
            anchor="w"
        )
        ttk.Progressbar(status_box, variable=self.progress_var, maximum=100).pack(
            fill="x", pady=(6, 4)
        )

        # 估算信息
        est_box = ttk.LabelFrame(
            parent, text="📐 估算", style="Section.TLabelframe", padding=8
        )
        est_box.grid(row=5, column=0, sticky="ew", pady=(0, 8))
        est_box.columnconfigure(0, weight=1)
        ttk.Label(
            est_box,
            textvariable=self.estimate_var,
            style="Muted.TLabel",
            wraplength=240,
            justify="left",
        ).pack(anchor="w")

    def _build_right_panel(self, parent):
        preview_box = ttk.LabelFrame(
            parent, text="预览与估计", style="Section.TLabelframe", padding=8
        )
        preview_box.grid(row=0, column=0, sticky="nsew", pady=(0, 8))
        preview_box.columnconfigure(0, weight=1)
        preview_box.rowconfigure(0, weight=1)

        self.preview_label = ttk.Label(preview_box, text="暂无预览图", anchor="center")
        self.preview_label.grid(row=0, column=0, sticky="nsew")
        ttk.Label(
            preview_box,
            textvariable=self.estimate_var,
            style="Muted.TLabel",
            wraplength=900,
            justify="left",
        ).grid(row=1, column=0, sticky="ew", pady=(8, 0))

        notebook = ttk.Notebook(parent)
        notebook.grid(row=1, column=0, sticky="nsew")

        info_tab = ttk.Frame(notebook, padding=8)
        log_tab = ttk.Frame(notebook, padding=8)
        notebook.add(info_tab, text="输出信息")
        notebook.add(log_tab, text="运行日志")

        info_tab.rowconfigure(0, weight=1)
        info_tab.columnconfigure(0, weight=1)
        self.info_text = tk.Text(
            info_tab,
            height=10,
            wrap="word",
            bg="#20242b",
            fg="#e7ebf0",
            insertbackground="#ffffff",
            relief="flat",
        )
        self.info_text.grid(row=0, column=0, sticky="nsew")
        info_scroll = ttk.Scrollbar(
            info_tab, orient="vertical", command=self.info_text.yview
        )
        info_scroll.grid(row=0, column=1, sticky="ns")
        self.info_text.configure(yscrollcommand=info_scroll.set)

        log_tab.rowconfigure(0, weight=1)
        log_tab.columnconfigure(0, weight=1)
        self.log_text = tk.Text(
            log_tab,
            wrap="word",
            bg="#20242b",
            fg="#e7ebf0",
            insertbackground="#ffffff",
            relief="flat",
        )
        self.log_text.grid(row=0, column=0, sticky="nsew")
        log_scroll = ttk.Scrollbar(
            log_tab, orient="vertical", command=self.log_text.yview
        )
        log_scroll.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=log_scroll.set)

    def _build_model_type_widget(self, parent):
        box = ttk.Frame(parent)
        ttk.Radiobutton(
            box, text="滑坡/空洞", variable=self.model_type_var, value="landslide"
        ).pack(side="left")
        ttk.Radiobutton(
            box, text="裂缝", variable=self.model_type_var, value="crack"
        ).pack(side="left", padx=(8, 0))
        return box

    def _add_row(
        self,
        parent,
        label: str,
        variable=None,
        widget=None,
        browse_dir=False,
        browse_file=False,
    ):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text=label, width=16).pack(side="left")  # 从20改为16
        if widget is None:
            entry = ttk.Entry(row, textvariable=variable, width=28)
            entry.pack(side="left", fill="x", expand=True)
        else:
            widget.pack(side="left", fill="x", expand=True)
        if browse_dir:
            ttk.Button(
                row,
                text="浏览",
                width=6,
                command=lambda v=variable: self._browse_dir(v),
            ).pack(side="left", padx=(6, 0))
        if browse_file:
            ttk.Button(
                row,
                text="浏览",
                width=6,
                command=lambda v=variable: self._browse_file(v),
            ).pack(side="left", padx=(6, 0))

    def _pack_row(self, parent, label: str, widget):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text=label, width=20).pack(side="left")
        widget.pack(side="left", fill="x", expand=True)

    def _bind_live_updates(self):
        all_vars = (
            [
                self.model_type_var,
                self.output_dir_var,
                self.output_name_var,
                self.seed_var,
                self.n_traces_var,
                self.center_freq_var,
                self.time_window_var,
            ]
            + list(self.common_vars.values())
            + list(self.landslide_vars.values())
            + list(self.crack_vars.values())
        )
        for var in all_vars:
            var.trace_add("write", lambda *_: self._schedule_live_update())

    def _schedule_live_update(self):
        if self._live_update_job is not None:
            self.after_cancel(self._live_update_job)
        self._live_update_job = self.after(180, self._refresh_live_info)

    def _to_float(self, value: str, name: str) -> float:
        try:
            return float(value)
        except Exception:
            raise ValueError(f"参数 {name} 不是有效数字: {value}")

    def _to_int(self, value: str, name: str) -> int:
        try:
            return int(float(value))
        except Exception:
            raise ValueError(f"参数 {name} 不是有效整数: {value}")

    def gather_common(self) -> Dict[str, Any]:
        seed_str = self.seed_var.get().strip()
        seed = None if seed_str == "" else self._to_int(seed_str, "随机种子")
        params = {
            "output_dir": self.output_dir_var.get().strip(),
            "output_name": self.output_name_var.get().strip() or "gpr_model",
            "model_x": self._to_float(self.common_vars["model_x"].get(), "model_x"),
            "model_y": self._to_float(self.common_vars["model_y"].get(), "model_y"),
            "dx": self._to_float(self.common_vars["dx"].get(), "dx"),
            "dy": self._to_float(self.common_vars["dy"].get(), "dy"),
            "water_table_depth": self._to_float(
                self.common_vars["water_table_depth"].get(), "water_table_depth"
            ),
            "src_start_x": self._to_float(
                self.common_vars["src_start_x"].get(), "src_start_x"
            ),
            "n_traces": self._to_int(self.n_traces_var.get(), "n_traces"),
            "center_freq_mhz": self._to_float(self.center_freq_var.get(), "中心频率"),
            "time_window_ns": self._to_float(self.time_window_var.get(), "时间窗"),
            "seed": seed,
            "use_timestamp_dir": bool(self.use_timestamp_var.get()),
        }
        self.validate_common(params)
        return params

    def validate_common(self, params: Dict[str, Any]):
        if params["dx"] <= 0 or params["dy"] <= 0:
            raise ValueError("dx 和 dy 必须大于 0")
        if params["model_x"] <= 0 or params["model_y"] <= 0:
            raise ValueError("模型尺寸必须大于 0")
        if params["n_traces"] <= 0:
            raise ValueError("n_traces 必须大于 0")
        if (
            params["water_table_depth"] < 0
            or params["water_table_depth"] > params["model_y"]
        ):
            raise ValueError("潜水面深度必须在 [0, model_y] 范围内")
        if params["src_start_x"] < 0 or params["src_start_x"] >= params["model_x"]:
            raise ValueError("扫描起始位置必须在 [0, model_x) 范围内")
        if params["center_freq_mhz"] <= 0 or params["time_window_ns"] <= 0:
            raise ValueError("中心频率和时间窗必须大于 0")
        nx = int(round(params["model_x"] / params["dx"]))
        ny = int(round(params["model_y"] / params["dy"]))
        if nx * ny > 8_000_000:
            raise ValueError("网格过大，可能内存不足。请增大 dx/dy 或减小模型尺寸。")

    def gather_model_params(self) -> Dict[str, Any]:
        common = self.gather_common()
        if self.model_type_var.get() == "landslide":
            common.update(
                {
                    "surface_amplitude": self._to_float(
                        self.landslide_vars["surface_amplitude"].get(),
                        "surface_amplitude",
                    ),
                    "surface_frequency": self._to_float(
                        self.landslide_vars["surface_frequency"].get(),
                        "surface_frequency",
                    ),
                    "dry_rock_ratio": self._to_float(
                        self.landslide_vars["dry_rock_ratio"].get(), "dry_rock_ratio"
                    ),
                    "saturated_rock_ratio": self._to_float(
                        self.landslide_vars["saturated_rock_ratio"].get(),
                        "saturated_rock_ratio",
                    ),
                    "rock_min_radius": self._to_float(
                        self.landslide_vars["rock_min_radius"].get(), "rock_min_radius"
                    ),
                    "rock_max_radius": self._to_float(
                        self.landslide_vars["rock_max_radius"].get(), "rock_max_radius"
                    ),
                    "void_center_x": self._to_float(
                        self.landslide_vars["void_center_x"].get(), "void_center_x"
                    ),
                    "void_center_y": self._to_float(
                        self.landslide_vars["void_center_y"].get(), "void_center_y"
                    ),
                    "void_radius_x": self._to_float(
                        self.landslide_vars["void_radius_x"].get(), "void_radius_x"
                    ),
                    "void_radius_y": self._to_float(
                        self.landslide_vars["void_radius_y"].get(), "void_radius_y"
                    ),
                }
            )
            self.validate_landslide(common)
        else:
            common.update(
                {
                    "crack_type": self.crack_vars["crack_type"].get(),
                    "crack_start_x": self._to_float(
                        self.crack_vars["crack_start_x"].get(), "crack_start_x"
                    ),
                    "crack_start_y": self._to_float(
                        self.crack_vars["crack_start_y"].get(), "crack_start_y"
                    ),
                    "crack_end_x": self._to_float(
                        self.crack_vars["crack_end_x"].get(), "crack_end_x"
                    ),
                    "crack_end_y": self._to_float(
                        self.crack_vars["crack_end_y"].get(), "crack_end_y"
                    ),
                    "crack_width": self._to_float(
                        self.crack_vars["crack_width"].get(), "crack_width"
                    ),
                    "crack_fill": self.crack_vars["crack_fill"].get(),
                    "crack_angle": self._to_float(
                        self.crack_vars["crack_angle"].get(), "crack_angle"
                    ),
                    "crack_count": self._to_int(
                        self.crack_vars["crack_count"].get(), "crack_count"
                    ),
                    "crack_curvature": self._to_float(
                        self.crack_vars["crack_curvature"].get(), "crack_curvature"
                    ),
                }
            )
            self.validate_crack(common)
        return common

    def validate_landslide(self, params: Dict[str, Any]):
        for key in ["dry_rock_ratio", "saturated_rock_ratio"]:
            if not 0.0 <= params[key] <= 1.0:
                raise ValueError(f"{key} 必须在 0 到 1 之间")
        if params["surface_amplitude"] < 0:
            raise ValueError("地表起伏幅度不能小于 0")
        if params["rock_min_radius"] <= 0 or params["rock_max_radius"] <= 0:
            raise ValueError("岩块半径必须大于 0")
        if params["rock_min_radius"] > params["rock_max_radius"]:
            raise ValueError("rock_min_radius 不能大于 rock_max_radius")
        if params["void_radius_x"] <= 0 or params["void_radius_y"] <= 0:
            raise ValueError("空洞半轴必须大于 0")
        if not (
            0 <= params["void_center_x"] <= params["model_x"]
            and 0 <= params["void_center_y"] <= params["model_y"]
        ):
            raise ValueError("空洞中心必须位于模型范围内")
        if not (
            0 <= params["void_center_x"] - params["void_radius_x"]
            and params["void_center_x"] + params["void_radius_x"] <= params["model_x"]
        ):
            raise ValueError("空洞 x 方向超出模型范围")
        if not (
            0 <= params["void_center_y"] - params["void_radius_y"]
            and params["void_center_y"] + params["void_radius_y"] <= params["model_y"]
        ):
            raise ValueError("空洞 y 方向超出模型范围")

    def validate_crack(self, params: Dict[str, Any]):
        if params["crack_width"] <= 0:
            raise ValueError("裂缝宽度必须大于 0")
        for key in ["crack_start_x", "crack_end_x"]:
            if not 0 <= params[key] <= params["model_x"]:
                raise ValueError(f"{key} 超出模型范围")
        for key in ["crack_start_y", "crack_end_y"]:
            if not 0 <= params[key] <= params["model_y"]:
                raise ValueError(f"{key} 超出模型范围")
        if params["crack_fill"] not in {"air", "water"}:
            raise ValueError("裂缝填充类型必须为 air 或 water")

    def _refresh_live_info(self):
        try:
            params = self.gather_model_params()
            nx = int(round(params["model_x"] / params["dx"]))
            ny = int(round(params["model_y"] / params["dy"]))
            cells = nx * ny
            src_start_x = params.get("src_start_x", 0.05)
            src_step = StandaloneModelBuilder._calc_src_steps(
                params["model_x"], params["n_traces"], params["dx"], src_start_x
            )
            # 计算实际扫描范围
            actual_end = src_start_x + (params["n_traces"] - 1) * src_step
            scan_ratio = actual_end / params["model_x"]
            h5_mb = cells * 2 / (1024 * 1024)
            desc = [
                f"网格估计: {nx} × {ny} = {cells:,} cells",
                f"预计 HDF5 原始数据约: {h5_mb:.2f} MB",
                f"扫描步长 src_steps ≈ {src_step:.4f} m",
                f"实际扫描终点 ≈ {actual_end:.2f} m ({scan_ratio:.0%})",
                f"输出目录: {params['output_dir']}",
            ]
            warn = []
            if cells > 3_000_000:
                warn.append("当前网格较大，预览和正演可能偏慢。")
            if params["dx"] < 0.003 or params["dy"] < 0.003:
                warn.append("dx/dy 很小，计算量会明显上升。")
            if params["n_traces"] > 120:
                warn.append("道数较多，gprMax 正演时间会增加。")
            if scan_ratio < 0.9:
                warn.append(
                    f"当前道数下实际扫描范围仅 {scan_ratio:.0%}，建议减少道数或增大模型。"
                )
            self.estimate_var.set(
                "\n".join(desc + (["警告: " + "；".join(warn)] if warn else []))
            )
            self.validation_var.set("参数校验通过。")
        except Exception as e:
            self.estimate_var.set("无法计算实时估计。")
            self.validation_var.set(f"参数检查: {e}")

    def _export_csv_files(self, output_dir: str, base_name: str, n_traces: int):
        """导出 CSV 文件，格式为 lineData_0000001.csv"""
        try:
            import h5py

            self.log("开始导出 CSV 文件...")

            # 查找所有 .out 文件
            out_files = sorted(
                [
                    f
                    for f in os.listdir(output_dir)
                    if f.endswith(".out") and "merged" not in f
                ]
            )

            if not out_files:
                self.log("警告: 找不到 .out 文件")
                return

            # 读取数据并导出
            for i, fname in enumerate(out_files, 1):
                fpath = os.path.join(output_dir, fname)
                with h5py.File(fpath, "r") as f:
                    data = f["rxs"]["rx1"]["Ez"][:]

                # 保存为 lineData_XXXXXX.csv 格式
                csv_name = f"lineData_{i:07d}.csv"
                csv_path = os.path.join(output_dir, csv_name)
                np.savetxt(csv_path, data, delimiter=",", fmt="%.8f")

            self.log(f"已导出 {len(out_files)} 个 CSV 文件")

        except Exception as e:
            self.log(f"导出 CSV 失败: {e}")

    def on_apply_preset(self):
        preset = self.preset_var.get()
        if preset == "fast":
            self.common_vars["model_x"].set("5.0")
            self.common_vars["model_y"].set("2.0")
            self.common_vars["dx"].set("0.01")
            self.common_vars["dy"].set("0.01")
            self.n_traces_var.set("30")
            self.center_freq_var.set("150")
            self.time_window_var.set("50")
        elif preset == "balanced":
            self.common_vars["model_x"].set("5.0")
            self.common_vars["model_y"].set("2.0")
            self.common_vars["dx"].set("0.005")
            self.common_vars["dy"].set("0.005")
            self.n_traces_var.set("50")
            self.center_freq_var.set("200")
            self.time_window_var.set("60")
        elif preset == "highres":
            self.common_vars["model_x"].set("5.0")
            self.common_vars["model_y"].set("2.0")
            self.common_vars["dx"].set("0.0025")
            self.common_vars["dy"].set("0.0025")
            self.n_traces_var.set("80")
            self.center_freq_var.set("250")
            self.time_window_var.set("70")
        elif preset == "示例-金属圆柱体":
            # 官方示例参数
            self.common_vars["model_x"].set("0.24")
            self.common_vars["model_y"].set("0.21")
            self.common_vars["dx"].set("0.002")
            self.common_vars["dy"].set("0.002")
            self.common_vars["water_table_depth"].set("0.17")
            self.n_traces_var.set("70")
            self.center_freq_var.set("1500")
            self.time_window_var.set("3")
            # 设置空洞参数
            self.landslide_vars["void_center_x"].set("0.12")
            self.landslide_vars["void_center_y"].set("0.08")
            self.landslide_vars["void_radius_x"].set("0.01")
            self.landslide_vars["void_radius_y"].set("0.01")
        self.log(f"已应用预设: {preset}")
        self._refresh_live_info()

    def on_generate(self):
        self._run_in_thread(run_full=False)

    def on_run_full(self):
        self._run_in_thread(run_full=True)

    def on_batch_generate(self):
        if self.worker is not None and self.worker.is_alive():
            messagebox.showwarning("提示", "已有任务在运行，请稍后。")
            return
        try:
            base_params = self.gather_model_params()
            batch_count = self._to_int(self.batch_count_var.get(), "样本数量")
            seed_step = self._to_int(self.batch_seed_step_var.get(), "种子步长")
            start_idx = self._to_int(self.batch_start_idx_var.get(), "起始编号")
            if batch_count <= 0:
                raise ValueError("样本数量必须大于 0")
        except Exception as e:
            messagebox.showerror("参数错误", str(e))
            return

        def task():
            try:
                self._threadsafe_set_busy(True)
                self._threadsafe_progress(0.0, "开始批量生成")
                summary = []
                base_seed = base_params.get("seed")
                base_name = base_params["output_name"]
                total = batch_count
                for i in range(batch_count):
                    params = dict(base_params)
                    idx = start_idx + i
                    params["output_name"] = f"{base_name}_{idx:04d}"
                    if base_seed is not None:
                        params["seed"] = base_seed + i * seed_step
                    else:
                        params["seed"] = None
                    self._threadsafe_progress(
                        i / max(total, 1), f"批量生成 {i + 1}/{total}"
                    )
                    self.log(f"[{i + 1}/{total}] 生成 {params['output_name']}...")
                    artifacts = self.generate_artifacts(params)
                    if self.batch_run_var.get() or self.auto_run_var.get():
                        self.run_gprmax_pipeline(artifacts, params["n_traces"])
                    summary.append(
                        {
                            "index": idx,
                            "seed": params["seed"],
                            "output_dir": artifacts.output_dir,
                            "hdf5_path": artifacts.hdf5_path,
                            "in_path": artifacts.in_path,
                            "preview_path": artifacts.preview_path,
                            "bscan_path": artifacts.bscan_path,
                            "model_info": artifacts.model_info,
                        }
                    )
                    self.last_artifacts = artifacts
                    self.after(
                        0, lambda p=artifacts.preview_path: self.update_preview(p)
                    )
                    self.after(0, lambda a=artifacts: self.update_info(a))

                summary_dir = (
                    self.last_artifacts.output_dir or base_params["output_dir"]
                )
                base_dir = (
                    os.path.dirname(summary_dir)
                    if os.path.isdir(summary_dir)
                    else base_params["output_dir"]
                )
                summary_path = os.path.join(base_dir, f"{base_name}_batch_summary.json")
                with open(summary_path, "w", encoding="utf-8") as f:
                    json.dump(summary, f, ensure_ascii=False, indent=2)
                self.log(f"批量生成完成，摘要已保存: {summary_path}")
                self._threadsafe_progress(1.0, "批量生成完成")
            except Exception as e:
                self.log(f"批量生成失败: {e}")
                self.log(traceback.format_exc())
                self.after(0, lambda: messagebox.showerror("批量生成失败", str(e)))
            finally:
                self._threadsafe_set_busy(False)

        self.worker = threading.Thread(target=task, daemon=True)
        self.worker.start()

    def _run_in_thread(self, run_full: bool):
        if self.worker is not None and self.worker.is_alive():
            messagebox.showwarning("提示", "已有任务在运行，请稍后。")
            return
        try:
            params = self.gather_model_params()
        except Exception as e:
            messagebox.showerror("参数错误", str(e))
            return

        def task():
            try:
                self._threadsafe_set_busy(True)
                self.log("=" * 64)
                self.log("开始执行任务")
                self.log(f"模型类型: {self.model_type_var.get()}")
                self._threadsafe_progress(0.02, "开始生成")
                artifacts = self.generate_artifacts(params)
                self.last_artifacts = artifacts
                self.after(0, lambda: self.update_preview(artifacts.preview_path))
                self.after(0, lambda: self.update_info(artifacts))
                if run_full or self.auto_run_var.get():
                    self.run_gprmax_pipeline(artifacts, params["n_traces"])
                    self.after(0, lambda: self.update_info(self.last_artifacts))
                self.log("任务完成")
                self._threadsafe_progress(1.0, "任务完成")
            except Exception as e:
                self.log(f"发生错误: {e}")
                self.log(traceback.format_exc())
                self.after(0, lambda: messagebox.showerror("运行失败", str(e)))
            finally:
                self._threadsafe_set_busy(False)

        self.worker = threading.Thread(target=task, daemon=True)
        self.worker.start()

    def _threadsafe_set_busy(self, busy: bool):
        self.after(0, lambda: self._set_busy(busy))

    def _set_busy(self, busy: bool):
        self.status_var.set("运行中..." if busy else "就绪")

    def _threadsafe_progress(self, frac: float, text: str):
        self.after(0, lambda f=frac, t=text: self.set_progress(f, t))

    def set_progress(self, frac: float, text: str):
        self.progress_var.set(max(0.0, min(100.0, frac * 100.0)))
        self.status_var.set(text)

    def generate_artifacts(self, params: Dict[str, Any]) -> GenerationArtifacts:
        model_type = self.model_type_var.get()
        self.log(f"生成模型类型: {model_type}")
        if model_type == "landslide":
            return self.builder.build_landslide_model(**params)

        # 裂缝类型需要特别记录
        crack_type = params.get("crack_type", "linear")
        self.log(f"裂缝类型: {crack_type}")
        return self.builder.build_crack_model(**params)

    def run_gprmax_pipeline(self, artifacts: GenerationArtifacts, n_traces: int):
        import os

        python_exec = self.python_var.get().strip() or sys.executable
        if not artifacts.in_path:
            raise RuntimeError("未找到 .in 文件，无法运行 gprMax")

        self._threadsafe_progress(0.72, "调用 gprMax 正演")
        self.log(f"使用 Python: {python_exec}")
        self.log("开始调用 gprMax 正演...")

        # 确保使用 gprMax 虚拟环境中的 Python
        cmd = [python_exec, "-m", "gprMax", artifacts.in_path, "-n", str(n_traces)]

        # 如果启用 GPU 加速，添加 -gpu 参数
        if self.use_gpu_var.get():
            cmd.append("-gpu")
            self.log("GPU 加速模式已启用 (-gpu)")

            # 动态检测 CUDA 路径
            cuda_base = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
            if os.path.isdir(cuda_base):
                # 查找所有已安装的 CUDA 版本
                cuda_versions = sorted(
                    [d for d in os.listdir(cuda_base) if d.startswith("v")],
                    reverse=True,
                )
                for ver in cuda_versions:
                    cuda_bin = os.path.join(cuda_base, ver, "bin")
                    if os.path.isdir(cuda_bin) and cuda_bin not in os.environ.get(
                        "PATH", ""
                    ):
                        os.environ["PATH"] = cuda_bin + ";" + os.environ.get("PATH", "")
                        self.log(f"已添加 CUDA 路径: {cuda_bin}")
                        break

            # 查找 MSVC 编译器路径（如果存在）
            vs_path = r"E:\Visual Studio 2022"
            vs_path_typo = r"E:\sisual stdio 2022"  # 兼容旧路径拼写
            vs_base = vs_path if os.path.isdir(vs_path) else vs_path_typo
            if os.path.isdir(vs_base):
                vc_tools = os.path.join(vs_base, "VC", "Tools", "MSVC")
                if os.path.isdir(vc_tools):
                    msvc_versions = sorted(os.listdir(vc_tools), reverse=True)
                    for ver in msvc_versions:
                        vc_bin = os.path.join(vc_tools, ver, "bin", "Hostx64", "x64")
                        if os.path.isdir(vc_bin) and vc_bin not in os.environ.get(
                            "PATH", ""
                        ):
                            os.environ["PATH"] = (
                                vc_bin + ";" + os.environ.get("PATH", "")
                            )
                            self.log(f"已添加 MSVC 路径: {vc_bin}")
                            break

        # 设置工作目录为 gprMax 根目录（自动检测）
        # 优先使用 .in 文件所在目录，其次使用 GUI 当前工作目录
        in_file_dir = os.path.dirname(os.path.abspath(artifacts.in_path))
        gprmax_dir = in_file_dir

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=gprmax_dir)
        if result.stdout:
            self.log(result.stdout)
        if result.stderr:
            self.log(result.stderr)
        if result.returncode != 0:
            raise RuntimeError("gprMax 运行失败，请检查 Python 路径和 gprMax 环境。")

        self._threadsafe_progress(0.88, "合并输出并生成 B-scan")
        self.log("gprMax 正演完成，开始合并输出...")
        merge_cmd = [python_exec, "-m", "tools.outputfiles_merge", artifacts.in_path]
        merge_res = subprocess.run(
            merge_cmd, capture_output=True, text=True, cwd=gprmax_dir
        )
        if merge_res.stdout:
            self.log(merge_res.stdout)
        if merge_res.stderr:
            self.log(merge_res.stderr)

        base_name = os.path.splitext(os.path.basename(artifacts.in_path))[0]
        out_dir = os.path.dirname(artifacts.in_path)
        merged = os.path.join(out_dir, f"{base_name}_merged.out")
        if not os.path.exists(merged):
            candidates = sorted(
                [
                    os.path.join(out_dir, f)
                    for f in os.listdir(out_dir)
                    if f.endswith(".out")
                ],
                key=os.path.getmtime,
            )
            if not candidates:
                raise RuntimeError("正演完成后未找到 .out 文件")
            merged = candidates[-1]

        bscan_png = os.path.join(out_dir, f"{base_name}_bscan.png")
        self.builder.plot_gprmax_bscan(merged, bscan_png)
        artifacts.merged_out_path = merged
        artifacts.bscan_path = bscan_png

        # 导出为 lineData CSV 格式
        self._export_csv_files(out_dir, base_name, n_traces)
        self.last_artifacts = artifacts
        self.after(0, lambda: self.update_preview(bscan_png))
        self.log(f"B-scan 已保存: {bscan_png}")

    def on_save_params(self):
        try:
            params = {
                "model_type": self.model_type_var.get(),
                "output_dir": self.output_dir_var.get(),
                "output_name": self.output_name_var.get(),
                "python": self.python_var.get(),
                "seed": self.seed_var.get(),
                "n_traces": self.n_traces_var.get(),
                "center_freq_mhz": self.center_freq_var.get(),
                "time_window_ns": self.time_window_var.get(),
                "use_timestamp_dir": self.use_timestamp_var.get(),
                "auto_run": self.auto_run_var.get(),
                "preset": self.preset_var.get(),
                "batch_count": self.batch_count_var.get(),
                "batch_seed_step": self.batch_seed_step_var.get(),
                "batch_start_index": self.batch_start_idx_var.get(),
                "batch_run": self.batch_run_var.get(),
                "common": {k: v.get() for k, v in self.common_vars.items()},
                "landslide": {k: v.get() for k, v in self.landslide_vars.items()},
                "crack": {k: v.get() for k, v in self.crack_vars.items()},
            }
            path = filedialog.asksaveasfilename(
                title="保存参数",
                defaultextension=".json",
                filetypes=[("JSON files", "*.json")],
            )
            if not path:
                return
            with open(path, "w", encoding="utf-8") as f:
                json.dump(params, f, ensure_ascii=False, indent=2)
            self.log(f"参数已保存: {path}")
        except Exception as e:
            messagebox.showerror("保存失败", str(e))

    def on_load_params(self):
        path = filedialog.askopenfilename(
            title="加载参数", filetypes=[("JSON files", "*.json")]
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                params = json.load(f)
            self.model_type_var.set(params.get("model_type", "landslide"))
            self.output_dir_var.set(params.get("output_dir", self.output_dir_var.get()))
            self.output_name_var.set(
                params.get("output_name", self.output_name_var.get())
            )
            self.python_var.set(params.get("python", self.python_var.get()))
            self.seed_var.set(str(params.get("seed", self.seed_var.get())))
            self.n_traces_var.set(str(params.get("n_traces", self.n_traces_var.get())))
            self.center_freq_var.set(
                str(params.get("center_freq_mhz", self.center_freq_var.get()))
            )
            self.time_window_var.set(
                str(params.get("time_window_ns", self.time_window_var.get()))
            )
            self.use_timestamp_var.set(bool(params.get("use_timestamp_dir", True)))
            self.auto_run_var.set(bool(params.get("auto_run", False)))
            self.preset_var.set(params.get("preset", self.preset_var.get()))
            self.batch_count_var.set(
                str(params.get("batch_count", self.batch_count_var.get()))
            )
            self.batch_seed_step_var.set(
                str(params.get("batch_seed_step", self.batch_seed_step_var.get()))
            )
            self.batch_start_idx_var.set(
                str(params.get("batch_start_index", self.batch_start_idx_var.get()))
            )
            self.batch_run_var.set(bool(params.get("batch_run", False)))
            for k, v in params.get("common", {}).items():
                if k in self.common_vars:
                    self.common_vars[k].set(str(v))
            for k, v in params.get("landslide", {}).items():
                if k in self.landslide_vars:
                    self.landslide_vars[k].set(str(v))
            for k, v in params.get("crack", {}).items():
                if k in self.crack_vars:
                    self.crack_vars[k].set(str(v))
            self.log(f"参数已加载: {path}")
            self._refresh_live_info()
        except Exception as e:
            messagebox.showerror("加载失败", str(e))

    def on_open_output_dir(self):
        path = self.last_artifacts.output_dir or self.output_dir_var.get().strip()
        if not path:
            messagebox.showwarning("提示", "没有可打开的目录")
            return
        path = os.path.abspath(path)
        if not os.path.exists(path):
            messagebox.showwarning("提示", f"目录不存在: {path}")
            return
        try:
            if sys.platform.startswith("win"):
                os.startfile(path)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])
        except Exception as e:
            messagebox.showerror("打开失败", str(e))

    def update_preview(self, image_path: str):
        if not image_path or not os.path.exists(image_path):
            self.preview_label.configure(text="预览图不存在", image="")
            return
        try:
            from PIL import Image, ImageTk  # type: ignore

            image = Image.open(image_path)
            preview_w = max(640, self.preview_label.winfo_width())
            preview_h = max(320, self.preview_label.winfo_height())
            image.thumbnail((preview_w - 20, preview_h - 20))
            img = ImageTk.PhotoImage(image)
            self.image_cache = img
            self.preview_label.configure(image=img, text="")
        except Exception:
            try:
                img = tk.PhotoImage(file=image_path)
                self.image_cache = img
                self.preview_label.configure(image=img, text="")
            except Exception as e:
                self.preview_label.configure(
                    text=f"无法显示预览图\n{image_path}\n{e}", image=""
                )

    def update_info(self, artifacts: GenerationArtifacts):
        info = {
            "output_dir": artifacts.output_dir,
            "hdf5_path": artifacts.hdf5_path,
            "materials_path": artifacts.materials_path,
            "in_path": artifacts.in_path,
            "preview_path": artifacts.preview_path,
            "bscan_path": artifacts.bscan_path,
            "merged_out_path": artifacts.merged_out_path,
            "model_info": artifacts.model_info,
        }
        self.info_text.delete("1.0", tk.END)
        self.info_text.insert(tk.END, json.dumps(info, ensure_ascii=False, indent=2))

    def log(self, message: str):
        if not message.endswith("\n"):
            message += "\n"
        self.log_queue.put(message)

    def _poll_log_queue(self):
        try:
            while True:
                msg = self.log_queue.get_nowait()
                self.log_text.insert(tk.END, msg)
                self.log_text.see(tk.END)
        except queue.Empty:
            pass
        self.after(120, self._poll_log_queue)

    def clear_log(self):
        self.log_text.delete("1.0", tk.END)

    def _browse_dir(self, var: tk.StringVar):
        path = filedialog.askdirectory(title="选择目录")
        if path:
            var.set(path)

    def _browse_file(self, var: tk.StringVar):
        path = filedialog.askopenfilename(title="选择文件")
        if path:
            var.set(path)


def main():
    # 添加调试信息
    print("[DEBUG] 启动 GPRMax GUI...")
    print(f"[DEBUG] Python: {sys.version}")
    print(f"[DEBUG] 工作目录: {os.getcwd()}")

    if _MISSING_DEPS:
        missing = ", ".join(sorted(set(_MISSING_DEPS)))
        msg = (
            "缺少依赖库：" + missing + "\n\n"
            "请先在当前 Python 环境执行：\n"
            "python -m pip install numpy matplotlib pillow h5py\n"
        )
        print(f"[ERROR] {msg}")
        try:
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("依赖缺失", msg)
            root.destroy()
        except Exception as e:
            print(f"[ERROR] 无法显示错误对话框: {e}")
            print(msg)
        return

    print("[DEBUG] 依赖检查通过，启动 GUI...")

    try:
        app = GPRMaxStandaloneGUI()
        print("[DEBUG] GUI 实例创建成功")
        app.mainloop()
    except Exception as e:
        print(f"[ERROR] GUI 启动失败: {e}")
        import traceback

        traceback.print_exc()
        input("\n按回车键退出...")


if __name__ == "__main__":
    main()
