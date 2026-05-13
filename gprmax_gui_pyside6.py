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

"""PySide6 GUI for building physically-correct 2D gprMax models.

This GUI is intentionally docs-first:
- lower-left origin matches gprMax
- standard models are generated with direct object commands
- official cylinder B-scan example can be recreated faithfully
- audit panel checks wavelength sampling, scan coverage, PML/air spacing,
  and time-window adequacy before running
"""

from __future__ import annotations

import argparse
import decimal as d
import html
import json
import math
import os
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import h5py
import matplotlib
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Polygon, Rectangle
from PySide6 import QtCore, QtGui, QtWidgets

from tools.outputfiles_merge import merge_files


matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False


C0 = 299792458.0
DEFAULT_PYTHON = r"E:\gprMax\gprMax-v.3.1.7\.venv\Scripts\python.exe"
APP_TITLE = "gprMax 物理建模工作台"
APP_VERSION = "0.1"
PROJECT_ROOT = r"E:\gprMax\gprMax-v.3.1.7"
CUDA_ROOT = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
VS_ROOT = r"E:\vs2022"
MSVC_BIN = os.path.join(
    VS_ROOT, "VC", "Tools", "MSVC", "14.39.33519", "bin", "Hostx64", "x64"
)
DEFAULT_CURVED_CRACK_PATH_TEXT = (
    "0.120 0.200\n0.260 0.200\n0.340 0.180\n0.400 0.180\n0.450 0.140\n0.430 0.100"
)


HOST_PRESETS = {
    "half_space": {"name": "half_space", "eps_r": 6.0, "sigma": 0.0},
    "dry_soil": {"name": "dry_soil", "eps_r": 6.0, "sigma": 0.001},
    "wet_soil": {"name": "wet_soil", "eps_r": 25.0, "sigma": 0.01},
    "concrete": {"name": "concrete", "eps_r": 7.0, "sigma": 0.001},
    "custom": {"name": "host_material", "eps_r": 6.0, "sigma": 0.0},
}

TARGET_PRESETS = {
    "pec": {"name": "pec", "eps_r": None, "sigma": None, "builtin": True},
    "free_space": {
        "name": "free_space",
        "eps_r": None,
        "sigma": None,
        "builtin": True,
    },
    "water_fill": {
        "name": "water_fill",
        "eps_r": 81.0,
        "sigma": 0.0,
        "builtin": False,
    },
    "concrete": {
        "name": "target_concrete",
        "eps_r": 7.0,
        "sigma": 0.001,
        "builtin": False,
    },
    "custom": {
        "name": "target_material",
        "eps_r": 9.0,
        "sigma": 0.0,
        "builtin": False,
    },
}

PRESETS = {
    "official_cylinder_bscan": {
        "label": "官方圆柱体 B-scan",
        "title": "B-scan from a metal cylinder buried in a dielectric half-space",
        "domain_x": 0.240,
        "domain_y": 0.210,
        "dx": 0.002,
        "dy": 0.002,
        "time_window_ns": 3.0,
        "host_preset": "half_space",
        "host_name": "half_space",
        "host_eps_r": 6.0,
        "host_sigma": 0.0,
        "ground_surface_y": 0.170,
        "lift_off": 0.0,
        "source_start_x": 0.040,
        "receiver_offset": 0.040,
        "scan_step": 0.002,
        "n_traces": 60,
        "center_freq_mhz": 1500.0,
        "target_shape": "cylinder",
        "target_preset": "pec",
        "target_name": "pec",
        "target_eps_r": 0.0,
        "target_sigma": 0.0,
        "target_center_x": 0.120,
        "target_center_y": 0.080,
        "target_radius": 0.010,
        "target_width": 0.020,
        "target_height": 0.020,
        "target_orientation": "horizontal",
        "target_angle_deg": 0.0,
        "write_geometry_view": False,
        "geometry_only": False,
        "source_type": "hertzian_dipole",
        "waveform_type": "ricker",
        "source_resistance": 0.0,
        "source_polarisation": "z",
    },
    "realistic_pipe_bscan": {
        "label": "真实感管线 B-scan",
        "title": "Realistic B-scan with direct wave and buried pipe hyperbola",
        "domain_x": 1.200,
        "domain_y": 0.700,
        "dx": 0.005,
        "dy": 0.005,
        "time_window_ns": 18.0,
        "host_preset": "dry_soil",
        "host_name": "dry_soil",
        "host_eps_r": 9.0,
        "host_sigma": 0.001,
        "ground_surface_y": 0.520,
        "lift_off": 0.020,
        "source_start_x": 0.120,
        "receiver_offset": 0.100,
        "scan_step": 0.010,
        "n_traces": 90,
        "center_freq_mhz": 500.0,
        "target_shape": "cylinder",
        "target_preset": "pec",
        "target_name": "pec",
        "target_eps_r": 0.0,
        "target_sigma": 0.0,
        "target_center_x": 0.620,
        "target_center_y": 0.260,
        "target_radius": 0.030,
        "target_width": 0.060,
        "target_height": 0.060,
        "target_orientation": "horizontal",
        "target_angle_deg": 0.0,
        "write_geometry_view": False,
        "geometry_only": False,
        "source_type": "hertzian_dipole",
        "waveform_type": "ricker",
        "source_resistance": 0.0,
        "source_polarisation": "z",
    },
    "uav_pipe_gain_workflow_bscan": {
        "label": "UAV 管线 raw+增益流程 B-scan",
        "title": "UAV-GPR raw B-scan for background removal and mild gain workflow",
        "domain_x": 1.200,
        "domain_y": 0.850,
        "dx": 0.005,
        "dy": 0.005,
        "time_window_ns": 24.0,
        "host_preset": "dry_soil",
        "host_name": "dry_soil",
        "host_eps_r": 9.0,
        "host_sigma": 0.004,
        "ground_surface_y": 0.550,
        "lift_off": 0.150,
        "source_start_x": 0.100,
        "receiver_offset": 0.120,
        "scan_step": 0.010,
        "n_traces": 90,
        "center_freq_mhz": 500.0,
        "target_shape": "cylinder",
        "target_preset": "pec",
        "target_name": "pec",
        "target_eps_r": 0.0,
        "target_sigma": 0.0,
        "target_center_x": 0.620,
        "target_center_y": 0.220,
        "target_radius": 0.035,
        "target_width": 0.070,
        "target_height": 0.070,
        "target_orientation": "horizontal",
        "target_angle_deg": 0.0,
        "background_layers": [
            {
                "name": "uav_shallow_dry_soil",
                "eps_r": 7.5,
                "sigma": 0.003,
                "y_min": 0.390,
                "y_max": 0.460,
            },
            {
                "name": "uav_deeper_moist_soil",
                "eps_r": 11.0,
                "sigma": 0.006,
                "y_min": 0.120,
                "y_max": 0.180,
            },
        ],
        "write_geometry_view": False,
        "geometry_only": False,
        "source_type": "hertzian_dipole",
        "waveform_type": "ricker",
        "source_resistance": 0.0,
        "source_polarisation": "z",
    },
    "air_void_halfspace": {
        "label": "半空间空气空洞",
        "title": "B-scan from an air void buried in a dielectric half-space",
        "domain_x": 0.600,
        "domain_y": 0.350,
        "dx": 0.002,
        "dy": 0.002,
        "time_window_ns": 8.0,
        "host_preset": "half_space",
        "host_name": "half_space",
        "host_eps_r": 6.0,
        "host_sigma": 0.0,
        "ground_surface_y": 0.260,
        "lift_off": 0.0,
        "source_start_x": 0.080,
        "receiver_offset": 0.040,
        "scan_step": 0.004,
        "n_traces": 90,
        "center_freq_mhz": 900.0,
        "target_shape": "cylinder",
        "target_preset": "free_space",
        "target_name": "free_space",
        "target_eps_r": 0.0,
        "target_sigma": 0.0,
        "target_center_x": 0.300,
        "target_center_y": 0.150,
        "target_radius": 0.030,
        "target_width": 0.060,
        "target_height": 0.060,
        "target_orientation": "horizontal",
        "target_angle_deg": 0.0,
        "write_geometry_view": False,
        "geometry_only": False,
        "source_type": "hertzian_dipole",
        "waveform_type": "ricker",
        "source_resistance": 0.0,
        "source_polarisation": "z",
    },
    "water_void_halfspace": {
        "label": "半空间充水空洞",
        "title": "B-scan from a water-filled void buried in a dielectric half-space",
        "domain_x": 0.600,
        "domain_y": 0.350,
        "dx": 0.002,
        "dy": 0.002,
        "time_window_ns": 10.0,
        "host_preset": "half_space",
        "host_name": "half_space",
        "host_eps_r": 6.0,
        "host_sigma": 0.0,
        "ground_surface_y": 0.260,
        "lift_off": 0.0,
        "source_start_x": 0.080,
        "receiver_offset": 0.040,
        "scan_step": 0.004,
        "n_traces": 90,
        "center_freq_mhz": 600.0,
        "target_shape": "cylinder",
        "target_preset": "water_fill",
        "target_name": "water_fill",
        "target_eps_r": 81.0,
        "target_sigma": 0.0,
        "target_center_x": 0.300,
        "target_center_y": 0.150,
        "target_radius": 0.030,
        "target_width": 0.060,
        "target_height": 0.060,
        "target_orientation": "horizontal",
        "target_angle_deg": 0.0,
        "write_geometry_view": False,
        "geometry_only": False,
        "source_type": "hertzian_dipole",
        "waveform_type": "ricker",
        "source_resistance": 0.0,
        "source_polarisation": "z",
    },
    "air_crack_halfspace": {
        "label": "半空间空气裂缝",
        "title": "B-scan from an air-filled crack buried in a dielectric half-space",
        "domain_x": 0.600,
        "domain_y": 0.350,
        "dx": 0.002,
        "dy": 0.002,
        "time_window_ns": 8.0,
        "host_preset": "half_space",
        "host_name": "half_space",
        "host_eps_r": 6.0,
        "host_sigma": 0.0,
        "ground_surface_y": 0.260,
        "lift_off": 0.0,
        "source_start_x": 0.080,
        "receiver_offset": 0.040,
        "scan_step": 0.004,
        "n_traces": 90,
        "center_freq_mhz": 900.0,
        "target_shape": "crack",
        "target_preset": "free_space",
        "target_name": "free_space",
        "target_eps_r": 0.0,
        "target_sigma": 0.0,
        "target_center_x": 0.300,
        "target_center_y": 0.150,
        "target_radius": 0.010,
        "target_width": 0.120,
        "target_height": 0.010,
        "target_orientation": "horizontal",
        "target_angle_deg": 0.0,
        "write_geometry_view": False,
        "geometry_only": False,
    },
    "water_crack_halfspace": {
        "label": "半空间充水裂缝",
        "title": "B-scan from a water-filled crack buried in a dielectric half-space",
        "domain_x": 0.600,
        "domain_y": 0.350,
        "dx": 0.002,
        "dy": 0.002,
        "time_window_ns": 10.0,
        "host_preset": "half_space",
        "host_name": "half_space",
        "host_eps_r": 6.0,
        "host_sigma": 0.0,
        "ground_surface_y": 0.260,
        "lift_off": 0.0,
        "source_start_x": 0.080,
        "receiver_offset": 0.040,
        "scan_step": 0.004,
        "n_traces": 90,
        "center_freq_mhz": 600.0,
        "target_shape": "crack",
        "target_preset": "water_fill",
        "target_name": "water_fill",
        "target_eps_r": 81.0,
        "target_sigma": 0.0,
        "target_center_x": 0.300,
        "target_center_y": 0.150,
        "target_radius": 0.010,
        "target_width": 0.120,
        "target_height": 0.010,
        "target_orientation": "horizontal",
        "target_angle_deg": 0.0,
        "use_curved_crack": False,
        "crack_path_text": "",
        "cylinder_tilt_deg": 0.0,
        "write_geometry_view": False,
        "geometry_only": False,
        "source_type": "hertzian_dipole",
        "waveform_type": "ricker",
        "source_resistance": 0.0,
        "source_polarisation": "z",
    },
    "air_curved_crack_halfspace": {
        "label": "半空间空气曲折裂缝",
        "title": "B-scan from a curved air-filled crack buried in a dielectric half-space",
        "domain_x": 0.600,
        "domain_y": 0.350,
        "dx": 0.002,
        "dy": 0.002,
        "time_window_ns": 8.0,
        "host_preset": "half_space",
        "host_name": "half_space",
        "host_eps_r": 6.0,
        "host_sigma": 0.0,
        "ground_surface_y": 0.260,
        "lift_off": 0.0,
        "source_start_x": 0.080,
        "receiver_offset": 0.040,
        "scan_step": 0.004,
        "n_traces": 90,
        "center_freq_mhz": 900.0,
        "target_shape": "crack",
        "target_preset": "free_space",
        "target_name": "free_space",
        "target_eps_r": 0.0,
        "target_sigma": 0.0,
        "target_center_x": 0.300,
        "target_center_y": 0.150,
        "target_radius": 0.010,
        "target_width": 0.120,
        "target_height": 0.010,
        "target_orientation": "horizontal",
        "target_angle_deg": 0.0,
        "use_curved_crack": True,
        "crack_path_text": DEFAULT_CURVED_CRACK_PATH_TEXT,
        "cylinder_tilt_deg": 0.0,
        "write_geometry_view": False,
        "geometry_only": False,
        "source_type": "hertzian_dipole",
        "waveform_type": "ricker",
        "source_resistance": 0.0,
        "source_polarisation": "z",
    },
    "water_curved_crack_halfspace": {
        "label": "半空间充水曲折裂缝",
        "title": "B-scan from a curved water-filled crack buried in a dielectric half-space",
        "domain_x": 0.600,
        "domain_y": 0.350,
        "dx": 0.002,
        "dy": 0.002,
        "time_window_ns": 10.0,
        "host_preset": "half_space",
        "host_name": "half_space",
        "host_eps_r": 6.0,
        "host_sigma": 0.0,
        "ground_surface_y": 0.260,
        "lift_off": 0.0,
        "source_start_x": 0.080,
        "receiver_offset": 0.040,
        "scan_step": 0.004,
        "n_traces": 90,
        "center_freq_mhz": 600.0,
        "target_shape": "crack",
        "target_preset": "water_fill",
        "target_name": "water_fill",
        "target_eps_r": 81.0,
        "target_sigma": 0.0,
        "target_center_x": 0.300,
        "target_center_y": 0.150,
        "target_radius": 0.010,
        "target_width": 0.120,
        "target_height": 0.010,
        "target_orientation": "horizontal",
        "target_angle_deg": 0.0,
        "use_curved_crack": True,
        "crack_path_text": DEFAULT_CURVED_CRACK_PATH_TEXT,
        "cylinder_tilt_deg": 0.0,
        "write_geometry_view": False,
        "geometry_only": False,
        "source_type": "hertzian_dipole",
        "waveform_type": "ricker",
        "source_resistance": 0.0,
        "source_polarisation": "z",
    },
}


def human_cells(value: float) -> str:
    return "{:.1f}".format(value)


def material_velocity(eps_r: float) -> float:
    return C0 / math.sqrt(max(eps_r, 1e-9))


def highest_significant_frequency(center_freq_hz: float) -> float:
    return center_freq_hz * 3.0


def gprmax_round_cells(value: float, dl: float) -> int:
    if dl <= 0:
        return 0
    return int(
        d.Decimal(value / dl).quantize(
            d.Decimal("1"),
            rounding=d.ROUND_HALF_DOWN,
        )
    )


def numeric_sort_key(path: str) -> Tuple[str, int]:
    basename = os.path.basename(path)
    digits = "".join(ch for ch in basename if ch.isdigit())
    return basename, int(digits or "0")


def parse_xy_points(text: str) -> List[Tuple[float, float]]:
    chunks = []
    normalized = text.replace("；", ";").replace("，", ",")
    for line in normalized.splitlines():
        chunks.extend(line.split(";"))

    points = []
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [part for part in chunk.replace(",", " ").split() if part]
        if len(parts) != 2:
            raise ValueError("曲折轨迹点格式应为每行一个 `x y` 或 `x, y`。")
        points.append((float(parts[0]), float(parts[1])))

    return points


def build_crack_segments_from_points(
    points: List[Tuple[float, float]],
    opening: float,
    material_name: str,
) -> List[CrackSpec]:
    segments = []
    for start, end in zip(points, points[1:]):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.hypot(dx, dy)
        if length <= 1e-12:
            continue
        segments.append(
            CrackSpec(
                center_x=0.5 * (start[0] + end[0]),
                center_y=0.5 * (start[1] + end[1]),
                width=length,
                height=opening,
                orientation="angled",
                angle_deg=math.degrees(math.atan2(dy, dx)),
                material_name=material_name,
            )
        )
    return segments


@dataclass
class MaterialSpec:
    name: str
    eps_r: float
    sigma: float
    mu_r: float = 1.0
    sigma_star: float = 0.0
    builtin: bool = False

    def input_line(self) -> str:
        return "#material: {0:g} {1:g} {2:g} {3:g} {4}".format(
            self.eps_r,
            self.sigma,
            self.mu_r,
            self.sigma_star,
            self.name,
        )


@dataclass
class CrackSpec:
    """单条裂缝的规格"""

    center_x: float = 0.300
    center_y: float = 0.150
    width: float = 0.120
    height: float = 0.010
    orientation: str = "horizontal"
    angle_deg: float = 30.0
    material_name: str = "free_space"

    def to_dict(self) -> dict:
        return {
            "center_x": self.center_x,
            "center_y": self.center_y,
            "width": self.width,
            "height": self.height,
            "orientation": self.orientation,
            "angle_deg": self.angle_deg,
            "material_name": self.material_name,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CrackSpec":
        return cls(**data)

    @property
    def crack_angle_deg(self) -> float:
        if self.orientation == "vertical":
            return 90.0
        if self.orientation == "angled":
            return self.angle_deg
        return 0.0

    def size_x(self) -> float:
        if self.orientation == "vertical":
            return self.height
        return self.width

    def size_y(self) -> float:
        if self.orientation == "vertical":
            return self.width
        return self.height

    def corners_xy(self) -> List[Tuple[float, float]]:
        half_length = 0.5 * self.width
        half_opening = 0.5 * self.height
        theta = math.radians(self.crack_angle_deg)
        ux = (math.cos(theta), math.sin(theta))
        uy = (-math.sin(theta), math.cos(theta))
        return [
            (
                self.center_x - half_length * ux[0] - half_opening * uy[0],
                self.center_y - half_length * ux[1] - half_opening * uy[1],
            ),
            (
                self.center_x + half_length * ux[0] - half_opening * uy[0],
                self.center_y + half_length * ux[1] - half_opening * uy[1],
            ),
            (
                self.center_x + half_length * ux[0] + half_opening * uy[0],
                self.center_y + half_length * ux[1] + half_opening * uy[1],
            ),
            (
                self.center_x - half_length * ux[0] + half_opening * uy[0],
                self.center_y - half_length * ux[1] + half_opening * uy[1],
            ),
        ]

    def bounds(self) -> Tuple[float, float, float, float]:
        corners = self.corners_xy()
        xs = [p[0] for p in corners]
        ys = [p[1] for p in corners]
        return (min(xs), max(xs), min(ys), max(ys))

    def min_dimension(self) -> float:
        return min(self.width, self.height)

    def input_lines(self, dz: float, material_map: Dict[str, str]) -> List[str]:
        mat_name = material_map.get(self.material_name, self.material_name)
        if self.orientation == "angled":
            p1, p2, p3, p4 = self.corners_xy()
            return [
                "#triangle: {0:.3f} {1:.3f} 0 {2:.3f} {3:.3f} 0 {4:.3f} {5:.3f} 0 {6:.3f} {7}".format(
                    p1[0], p1[1], p2[0], p2[1], p3[0], p3[1], dz, mat_name
                ),
                "#triangle: {0:.3f} {1:.3f} 0 {2:.3f} {3:.3f} 0 {4:.3f} {5:.3f} 0 {6:.3f} {7}".format(
                    p1[0], p1[1], p3[0], p3[1], p4[0], p4[1], dz, mat_name
                ),
            ]
        else:
            x_min, x_max, y_min, y_max = self.bounds()
            return [
                "#box: {0:.3f} {1:.3f} 0 {2:.3f} {3:.3f} {4:.3f} {5}".format(
                    x_min, y_min, x_max, y_max, dz, mat_name
                )
            ]


@dataclass
class SimulationConfig:
    title: str = PRESETS["official_cylinder_bscan"]["title"]
    output_root: str = r"D:\ClawX-Data\sim\gprmax_outcsv"
    output_name: str = "gpr_model"
    python_executable: str = DEFAULT_PYTHON
    use_gpu: bool = False
    geometry_fixed: bool = True
    geometry_only: bool = False
    timestamp_output: bool = True
    write_geometry_view: bool = False

    domain_x: float = 0.240
    domain_y: float = 0.210
    dx: float = 0.002
    dy: float = 0.002
    time_window_ns: float = 3.0

    host_name: str = "half_space"
    host_eps_r: float = 6.0
    host_sigma: float = 0.0
    ground_surface_y: float = 0.170
    lift_off: float = 0.0

    source_start_x: float = 0.040
    receiver_offset: float = 0.040
    scan_step: float = 0.002
    n_traces: int = 60
    center_freq_mhz: float = 1500.0

    target_shape: str = "cylinder"
    target_name: str = "pec"
    target_eps_r: float = 0.0
    target_sigma: float = 0.0
    target_center_x: float = 0.120
    target_center_y: float = 0.080
    target_radius: float = 0.010
    target_width: float = 0.020
    target_height: float = 0.020
    target_orientation: str = "horizontal"
    target_angle_deg: float = 0.0
    use_curved_crack: bool = False
    crack_path_text: str = ""

    # 圆柱体倾斜角度（用于创建倾斜管道效果）
    cylinder_tilt_deg: float = 0.0  # 0 = 垂直于截面, 正值 = 向右倾斜

    # 多条裂缝模式（当 target_shape == "crack" 时使用）
    use_multi_cracks: bool = False
    cracks: List[CrackSpec] = field(default_factory=list)
    background_layers: List[Dict[str, float]] = field(default_factory=list)

    source_type: str = "hertzian_dipole"
    waveform_type: str = "ricker"
    source_resistance: float = 0.0
    source_polarisation: str = "z"

    preset_key: str = "official_cylinder_bscan"

    @property
    def dz(self) -> float:
        return self.dx

    @property
    def center_freq_hz(self) -> float:
        return self.center_freq_mhz * 1e6

    @property
    def time_window_s(self) -> float:
        return self.time_window_ns * 1e-9

    @property
    def source_y(self) -> float:
        return self.ground_surface_y + self.lift_off

    @property
    def receiver_start_x(self) -> float:
        return self.source_start_x + self.receiver_offset

    @property
    def receiver_y(self) -> float:
        return self.source_y

    @property
    def scan_step_cells(self) -> int:
        return gprmax_round_cells(self.scan_step, self.dx)

    @property
    def effective_scan_step(self) -> float:
        return self.scan_step_cells * self.dx

    @property
    def source_end_x(self) -> float:
        return self.source_start_x + (self.n_traces - 1) * self.effective_scan_step

    @property
    def receiver_end_x(self) -> float:
        return self.receiver_start_x + (self.n_traces - 1) * self.effective_scan_step

    @property
    def target_size_x(self) -> float:
        if self.target_shape == "cylinder":
            return 2.0 * self.target_radius
        if self.target_shape == "crack" and self.target_orientation == "vertical":
            return self.target_height
        return self.target_width

    @property
    def target_size_y(self) -> float:
        if self.target_shape == "cylinder":
            return 2.0 * self.target_radius
        if self.target_shape == "crack" and self.target_orientation == "vertical":
            return self.target_width
        return self.target_height

    @property
    def crack_angle_deg(self) -> float:
        if self.target_orientation == "vertical":
            return 90.0
        if self.target_orientation == "angled":
            return self.target_angle_deg
        return 0.0

    def crack_corners_xy(self) -> List[Tuple[float, float]]:
        half_length = 0.5 * self.target_width
        half_opening = 0.5 * self.target_height
        theta = math.radians(self.crack_angle_deg)
        ux = (math.cos(theta), math.sin(theta))
        uy = (-math.sin(theta), math.cos(theta))
        return [
            (
                self.target_center_x - half_length * ux[0] - half_opening * uy[0],
                self.target_center_y - half_length * ux[1] - half_opening * uy[1],
            ),
            (
                self.target_center_x + half_length * ux[0] - half_opening * uy[0],
                self.target_center_y + half_length * ux[1] - half_opening * uy[1],
            ),
            (
                self.target_center_x + half_length * ux[0] + half_opening * uy[0],
                self.target_center_y + half_length * ux[1] + half_opening * uy[1],
            ),
            (
                self.target_center_x - half_length * ux[0] + half_opening * uy[0],
                self.target_center_y - half_length * ux[1] + half_opening * uy[1],
            ),
        ]

    def curved_crack_points(self) -> List[Tuple[float, float]]:
        return parse_xy_points(self.crack_path_text)

    def crack_segments(self) -> List[CrackSpec]:
        if self.target_shape != "crack":
            return []
        if self.use_multi_cracks and self.cracks:
            return list(self.cracks)
        if self.use_curved_crack:
            return build_crack_segments_from_points(
                self.curved_crack_points(),
                self.target_height,
                self.target_name,
            )
        return [
            CrackSpec(
                center_x=self.target_center_x,
                center_y=self.target_center_y,
                width=self.target_width,
                height=self.target_height,
                orientation=self.target_orientation,
                angle_deg=self.target_angle_deg,
                material_name=self.target_name,
            )
        ]


@dataclass
class BuildArtifacts:
    output_dir: str
    input_path: str
    preview_path: str
    metadata_path: str
    manifest_path: str = ""
    primary_out_path: str = ""
    merged_out_path: str = ""
    bscan_png_path: str = ""
    background_removed_png_path: str = ""
    background_removed_gain_png_path: str = ""
    geometry_view_path: str = ""
    output_out_paths: List[str] = field(default_factory=list)


@dataclass
class AuditMessage:
    level: str
    text: str


@dataclass
class AuditReport:
    messages: List[AuditMessage] = field(default_factory=list)
    derived: Dict[str, float] = field(default_factory=dict)

    def add(self, level: str, text: str) -> None:
        self.messages.append(AuditMessage(level=level, text=text))

    def has_errors(self) -> bool:
        return any(msg.level == "error" for msg in self.messages)

    def to_html(self) -> str:
        colors = {
            "error": "#ff6b6b",
            "warning": "#f9c74f",
            "info": "#8ecae6",
            "success": "#90be6d",
        }
        parts = [
            "<div style='font-family:Segoe UI; font-size:12px;'>",
            "<h3 style='margin:0 0 8px 0;'>物理审计</h3>",
        ]
        if self.derived:
            parts.append("<p style='margin:0 0 8px 0; color:#d8dee9;'>")
            parts.append(
                "网格: {0:.0f} x {1:.0f} = {2:.0f} cells<br>"
                "发射源 y: {3:.3f} m<br>"
                "目标中心: ({4:.3f}, {5:.3f}) m<br>"
                "扫描范围: {6:.3f} -> {7:.3f} m<br>"
                "预计最近双程时延 TWT: {8:.3f} ns".format(
                    self.derived.get("nx", 0.0),
                    self.derived.get("ny", 0.0),
                    self.derived.get("cells", 0.0),
                    self.derived.get("source_y", 0.0),
                    self.derived.get("target_x", 0.0),
                    self.derived.get("target_y", 0.0),
                    self.derived.get("source_start_x", 0.0),
                    self.derived.get("source_end_x", 0.0),
                    self.derived.get("closest_twt_ns", 0.0),
                )
            )
            parts.append("</p>")
        if not self.messages:
            parts.append("<p style='color:#90be6d;'>没有审计消息。</p>")
        for msg in self.messages:
            parts.append(
                "<p style='margin:4px 0; color:{0};'><b>[{1}]</b> {2}</p>".format(
                    colors.get(msg.level, "#d8dee9"),
                    msg.level.upper(),
                    html.escape(msg.text),
                )
            )
        parts.append("</div>")
        return "".join(parts)


class PhysicsAuditor(object):
    def build_report(self, config: SimulationConfig) -> AuditReport:
        report = AuditReport()
        nx = int(round(config.domain_x / config.dx))
        ny = int(round(config.domain_y / config.dy))
        cells = nx * ny
        source_y = config.source_y
        min_dl = min(config.dx, config.dy)

        target_x_min, target_x_max, target_y_min, target_y_max = self._target_bounds(
            config
        )
        target_x = 0.5 * (target_x_min + target_x_max)
        target_y = 0.5 * (target_y_min + target_y_max)

        report.derived.update(
            {
                "nx": float(nx),
                "ny": float(ny),
                "cells": float(cells),
                "source_y": source_y,
                "target_x": target_x,
                "target_y": target_y,
                "source_start_x": config.source_start_x,
                "source_end_x": config.source_end_x,
                "requested_scan_step": config.scan_step,
                "effective_scan_step": config.effective_scan_step,
                "scan_step_cells": float(config.scan_step_cells),
            }
        )

        if config.ground_surface_y <= 0 or config.ground_surface_y >= config.domain_y:
            report.add("error", "地表 y 坐标必须严格位于计算域内部。")

        if source_y > config.domain_y:
            report.add("error", "发射源/接收器的 y 坐标超出了计算域。")

        if target_y_min < 0 or target_y_max > config.ground_surface_y:
            report.add(
                "error",
                "目标必须位于地表以下的宿主半空间内部。",
            )

        if target_x_min < 0 or target_x_max > config.domain_x:
            report.add("error", "目标在 x 方向超出了计算域。")

        if config.receiver_end_x > config.domain_x:
            report.add("error", "接收器扫描越出计算域，请减小道数或步长。")

        if config.source_end_x > config.domain_x:
            report.add("error", "发射源扫描越出计算域，请减小道数或步长。")

        if config.receiver_start_x < 0:
            report.add("error", "接收器起始 x 坐标超出计算域。")

        if config.scan_step <= 0:
            report.add("error", "扫描步长必须大于 0。")
        elif config.scan_step_cells < 1:
            report.add(
                "error",
                "扫描步长小于一个网格，gprMax 会量化为 0；请至少设置为 dx。"
            )
        elif abs(config.effective_scan_step - config.scan_step) > 1e-12:
            report.add(
                "warning",
                "扫描步长 {0:.6g} m 不是 dx={1:.6g} m 的整数倍；gprMax 实际会按 {2} 个网格执行，即 {3:.6g} m。预览图已按实际步长绘制。".format(
                    config.scan_step,
                    config.dx,
                    config.scan_step_cells,
                    config.effective_scan_step,
                ),
            )

        if config.n_traces <= 0:
            report.add("error", "扫描道数必须大于 0。")

        if config.n_traces > 1:
            if config.geometry_fixed:
                report.add(
                    "info",
                    "已启用 --geometry-fixed：仅首道重建几何，可提升多道 B-scan 吞吐。",
                )
            else:
                report.add(
                    "warning",
                    "当前未启用 --geometry-fixed，多道扫描会重复几何构建，速度更慢。",
                )

        if cells < 50000:
            report.add(
                "info",
                "当前模型网格较小，GPU 利用率偏低通常是正常现象（更像受启动/调度开销限制）。",
            )

        fmax = highest_significant_frequency(config.center_freq_hz)
        host_cells_per_lambda = material_velocity(config.host_eps_r) / (fmax * min_dl)
        if host_cells_per_lambda < 10:
            report.add(
                "error",
                "宿主材料波长仅被采样为 {0} 个网格，请提高分辨率或降低频率。".format(
                    human_cells(host_cells_per_lambda)
                ),
            )
        else:
            report.add(
                "success",
                "按 3 倍中心频率估算，宿主材料波长采样数为 {0} 个网格。".format(
                    human_cells(host_cells_per_lambda)
                ),
            )

        if config.target_name not in ["pec", "free_space"] and config.target_eps_r > 0:
            target_cells_per_lambda = material_velocity(config.target_eps_r) / (
                fmax * min_dl
            )
            if target_cells_per_lambda < 10:
                report.add(
                    "error",
                    "目标材料波长仅被采样为 {0} 个网格，请降低频率或细化网格。".format(
                        human_cells(target_cells_per_lambda)
                    ),
                )
            else:
                report.add(
                    "info",
                    "按 3 倍中心频率估算，目标材料波长采样数为 {0} 个网格。".format(
                        human_cells(target_cells_per_lambda)
                    ),
                )

        target_min_dimension = self._target_min_dimension(config)
        target_cells = target_min_dimension / min_dl
        if target_cells < 5:
            report.add(
                "error",
                "目标最小尺寸只有 {0} 个网格，分辨率明显不足。".format(
                    human_cells(target_cells)
                ),
            )
        elif target_cells < 10:
            report.add(
                "warning",
                "目标最小尺寸为 {0} 个网格，可以使用但不理想。".format(
                    human_cells(target_cells)
                ),
            )
        else:
            report.add(
                "success",
                "目标最小尺寸为 {0} 个网格。".format(human_cells(target_cells)),
            )

        if config.target_shape == "crack":
            crack_opening_cells = target_min_dimension / min_dl
            if config.use_curved_crack:
                points = config.curved_crack_points()
                if len(points) < 2:
                    report.add("error", "曲折裂缝轨迹至少需要 2 个点。")
                else:
                    report.add(
                        "info",
                        "曲折裂缝轨迹包含 {0} 个控制点、{1} 个分段。".format(
                            len(points), max(len(points) - 1, 0)
                        ),
                    )
            report.add(
                "info",
                "裂缝方向: {0}；当前裂缝开度约为 {1} 个网格。".format(
                    self._crack_orientation_label(config),
                    human_cells(crack_opening_cells),
                ),
            )
            if config.target_orientation == "angled" and not config.use_curved_crack:
                report.add(
                    "info",
                    "裂缝倾角为 {0:.1f} deg。".format(config.crack_angle_deg),
                )

        top_air_cells = (config.domain_y - source_y) / config.dy
        if top_air_cells < 15:
            report.add(
                "warning",
                "发射源上方空气层只有 {0} 个网格，手册建议至少 15-20 个网格。".format(
                    human_cells(top_air_cells)
                ),
            )
        else:
            report.add(
                "success",
                "发射源上方空气层为 {0} 个网格。".format(human_cells(top_air_cells)),
            )

        inner_pml_buffer_cells = self._inner_pml_buffer_cells(config)
        if inner_pml_buffer_cells < 0:
            report.add("error", "发射源、接收器或目标与默认 10-cell PML 区域重叠。")
        elif inner_pml_buffer_cells < 10:
            report.add(
                "warning",
                "超出默认 10-cell PML 之后的最小净距仅为 {0} 个网格；官方示例约保留 10 个网格。".format(
                    human_cells(inner_pml_buffer_cells)
                ),
            )
        else:
            report.add(
                "success",
                "超出默认 10-cell PML 之后的最小净距为 {0} 个网格。".format(
                    human_cells(inner_pml_buffer_cells)
                ),
            )

        if config.source_type == "voltage_source":
            if config.source_resistance == 0.0:
                report.add(
                    "warning",
                    "Voltage source 内阻为 0 Ω（硬源），波形结束后会完全反射；"
                    "建议设置非零内阻以模拟真实天线。",
                )
            else:
                report.add(
                    "info",
                    "Voltage source 内阻为 {0} Ω。".format(config.source_resistance),
                )
        elif config.source_type == "transmission_line":
            if config.source_resistance <= 0.0 or config.source_resistance >= 376.73:
                report.add(
                    "error",
                    "Transmission line 特征阻抗必须在 0–376.73 Ω 之间（当前 {0} Ω）。".format(
                        config.source_resistance
                    ),
                )
            else:
                report.add(
                    "info",
                    "Transmission line 特征阻抗为 {0} Ω。".format(
                        config.source_resistance
                    ),
                )
        else:
            report.add("info", "源类型为 Hertzian dipole（理想软源）。")

        for layer in config.background_layers:
            y_min = float(layer["y_min"])
            y_max = float(layer["y_max"])
            if y_min < 0 or y_max > config.ground_surface_y or y_max <= y_min:
                report.add(
                    "error",
                    "背景层 {0} 必须位于地表以下且 y_min < y_max。".format(
                        layer["name"]
                    ),
                )
        if config.background_layers:
            report.add(
                "info",
                "背景层数量: {0}；用于提供可背景抑制的弱水平反射。".format(
                    len(config.background_layers)
                ),
            )

        scan_mid_start = config.source_start_x + 0.5 * config.receiver_offset
        scan_mid_end = config.source_end_x + 0.5 * config.receiver_offset
        if target_x_max < scan_mid_start or target_x_min > scan_mid_end:
            report.add(
                "warning",
                "目标 x 不在测线中点覆盖范围内，双曲线可能被截断或缺失。",
            )
        else:
            report.add("success", "目标 x 位于测线中点覆盖范围内。")

        closest_twt_ns = self._closest_two_way_time_ns(config)
        report.derived["closest_twt_ns"] = closest_twt_ns
        if config.time_window_ns < closest_twt_ns * 1.35:
            report.add(
                "warning",
                "时间窗可能偏短；预计最近双程时延为 {0:.3f} ns。".format(
                    closest_twt_ns
                ),
            )
        else:
            report.add(
                "success",
                "时间窗足以覆盖预计最近双程时延 {0:.3f} ns。".format(closest_twt_ns),
            )

        if config.preset_key == "official_cylinder_bscan":
            report.add(
                "info",
                "该预设用于复现 gprMax 官方圆柱体 B-scan 示例。",
            )

        return report

    def _target_bounds(
        self, config: SimulationConfig
    ) -> Tuple[float, float, float, float]:
        if config.target_shape == "cylinder":
            return (
                config.target_center_x - config.target_radius,
                config.target_center_x + config.target_radius,
                config.target_center_y - config.target_radius,
                config.target_center_y + config.target_radius,
            )
        if config.target_shape == "crack":
            segments = config.crack_segments()
            if not segments:
                return (0.0, 0.0, 0.0, 0.0)
            xs = []
            ys = []
            for crack in segments:
                x_min, x_max, y_min, y_max = crack.bounds()
                xs.extend([x_min, x_max])
                ys.extend([y_min, y_max])
            return (min(xs), max(xs), min(ys), max(ys))
        return (
            config.target_center_x - 0.5 * config.target_size_x,
            config.target_center_x + 0.5 * config.target_size_x,
            config.target_center_y - 0.5 * config.target_size_y,
            config.target_center_y + 0.5 * config.target_size_y,
        )

    def _target_min_dimension(self, config: SimulationConfig) -> float:
        if config.target_shape == "cylinder":
            return 2.0 * config.target_radius
        if config.target_shape == "crack":
            segments = config.crack_segments()
            if not segments:
                return 0.0
            return min(crack.min_dimension() for crack in segments)
        return min(config.target_size_x, config.target_size_y)

    def _inner_pml_buffer_cells(self, config: SimulationConfig) -> float:
        pml = 10.0
        target_x_min, target_x_max, target_y_min, target_y_max = self._target_bounds(
            config
        )
        clearances = [
            config.source_start_x / config.dx - pml,
            config.receiver_start_x / config.dx - pml,
            (config.domain_x - config.source_end_x) / config.dx - pml,
            (config.domain_x - config.receiver_end_x) / config.dx - pml,
            config.source_y / config.dy - pml,
            (config.domain_y - config.source_y) / config.dy - pml,
            target_x_min / config.dx - pml,
            (config.domain_x - target_x_max) / config.dx - pml,
            target_y_min / config.dy - pml,
            (config.domain_y - target_y_max) / config.dy - pml,
        ]
        return min(clearances)

    def _closest_two_way_time_ns(self, config: SimulationConfig) -> float:
        velocity = material_velocity(config.host_eps_r)
        min_time = None
        if config.target_shape == "crack":
            target_points = [
                (crack.center_x, crack.center_y) for crack in config.crack_segments()
            ]
        else:
            target_points = [(config.target_center_x, config.target_center_y)]
        for index in range(config.n_traces):
            tx_x = config.source_start_x + index * config.scan_step
            rx_x = tx_x + config.receiver_offset
            for target_x, target_y in target_points:
                tx_path = math.hypot(tx_x - target_x, config.source_y - target_y)
                rx_path = math.hypot(rx_x - target_x, config.receiver_y - target_y)
                twt = (tx_path + rx_path) / velocity * 1e9
                if min_time is None or twt < min_time:
                    min_time = twt
        return float(min_time or 0.0)

    def _crack_orientation_label(self, config: SimulationConfig) -> str:
        if config.use_curved_crack:
            return "曲折轨迹"
        if config.target_orientation == "angled":
            return "斜裂缝"
        if config.target_orientation == "vertical":
            return "竖直"
        return "水平"

    def _audit_multi_cracks(
        self, config: SimulationConfig, report: AuditReport, min_dl: float
    ) -> None:
        """审计多条裂缝模式"""
        cracks = config.cracks
        report.add("info", "多条裂缝模式：共 {0} 条裂缝。".format(len(cracks)))

        # 检查每条裂缝
        for i, crack in enumerate(cracks):
            prefix = "裂缝 #{0}: ".format(i + 1)
            x_min, x_max, y_min, y_max = crack.bounds()

            # 检查是否在域内
            if x_min < 0 or x_max > config.domain_x:
                report.add("error", prefix + "x 方向超出计算域。")
            if y_min < 0 or y_max > config.ground_surface_y:
                report.add("error", prefix + "y 方向超出宿主半空间。")

            # 检查 PML 距离
            pml = 10.0
            clearances = [
                x_min / config.dx - pml,
                (config.domain_x - x_max) / config.dx - pml,
                y_min / config.dy - pml,
                (config.domain_y - y_max) / config.dy - pml,
            ]
            min_clearance = min(clearances)
            if min_clearance < 0:
                report.add("warning", prefix + "与 PML 边界区域重叠。")
            elif min_clearance < 5:
                report.add("warning", prefix + "距 PML 边界过近（< 5 cells）。")

            # 检查分辨率
            min_cells = crack.min_dimension() / min_dl
            if min_cells < 5:
                report.add(
                    "error",
                    prefix
                    + "最小尺寸仅 {0:.1f} 个网格，分辨率不足。".format(min_cells),
                )
            elif min_cells < 10:
                report.add(
                    "warning",
                    prefix + "最小尺寸 {0:.1f} 个网格，分辨率偏低。".format(min_cells),
                )

            # 裂缝信息
            orientation_label = (
                "斜裂缝"
                if crack.orientation == "angled"
                else ("竖直" if crack.orientation == "vertical" else "水平")
            )
            opening_cells = min(crack.width, crack.height) / min_dl
            report.add(
                "info",
                prefix
                + "{0}，中心({1:.3f}, {2:.3f})，开度约 {3:.1f} 个网格。".format(
                    orientation_label, crack.center_x, crack.center_y, opening_cells
                ),
            )


class ScenarioBuilder(object):
    def build_input_text(self, config: SimulationConfig) -> str:
        if config.preset_key == "official_cylinder_bscan" and self._is_exact_official(
            config
        ):
            return self._build_exact_official_text()

        lines = [
            "#title: {0}".format(config.title),
            "#domain: {0:.3f} {1:.3f} {2:.3f}".format(
                config.domain_x,
                config.domain_y,
                config.dz,
            ),
            "#dx_dy_dz: {0:.3f} {1:.3f} {2:.3f}".format(
                config.dx,
                config.dy,
                config.dz,
            ),
            "#time_window: {0:.9g}".format(config.time_window_s),
            "",
        ]

        for material in self._custom_materials(config):
            lines.append(material.input_line())

        lines.extend(
            [
                "",
                "#waveform: {0} 1 {1:g} my_wave".format(
                    config.waveform_type,
                    config.center_freq_hz,
                ),
            ]
        )

        if config.source_type == "hertzian_dipole":
            lines.append(
                "#hertzian_dipole: {0} {1:.3f} {2:.3f} 0 my_wave".format(
                    config.source_polarisation,
                    config.source_start_x,
                    config.source_y,
                )
            )
        elif config.source_type == "voltage_source":
            lines.append(
                "#voltage_source: {0} {1:.3f} {2:.3f} 0 {3} my_wave".format(
                    config.source_polarisation,
                    config.source_start_x,
                    config.source_y,
                    config.source_resistance,
                )
            )
        elif config.source_type == "transmission_line":
            lines.append(
                "#transmission_line: {0} {1:.3f} {2:.3f} 0 {3} my_wave".format(
                    config.source_polarisation,
                    config.source_start_x,
                    config.source_y,
                    config.source_resistance,
                )
            )

        lines.append(
            "#rx: {0:.3f} {1:.3f} 0".format(
                config.receiver_start_x,
                config.receiver_y,
            )
        )

        if config.n_traces > 1:
            lines.append("#src_steps: {0:.3f} 0 0".format(config.effective_scan_step))
            lines.append("#rx_steps: {0:.3f} 0 0".format(config.effective_scan_step))

        lines.extend(["", self._host_box_line(config)])
        lines.extend(self._background_layer_lines(config))
        lines.extend(self._target_lines(config))

        if config.write_geometry_view:
            lines.extend(
                [
                    "",
                    "#geometry_view: 0 0 0 {0:.3f} {1:.3f} {2:.3f} {3:.3f} {4:.3f} {5:.3f} {6}_geometry n".format(
                        config.domain_x,
                        config.domain_y,
                        config.dz,
                        config.dx,
                        config.dy,
                        config.dz,
                        config.output_name,
                    ),
                ]
            )

        return "\n".join(lines) + "\n"

    def build_files(
        self,
        config: SimulationConfig,
        report: AuditReport,
        output_dir: Optional[str] = None,
    ) -> BuildArtifacts:
        if output_dir is None:
            output_dir = self._make_output_dir(config)

        os.makedirs(output_dir, exist_ok=True)
        input_path = os.path.join(output_dir, "{0}.in".format(config.output_name))
        preview_path = os.path.join(
            output_dir, "{0}_preview.png".format(config.output_name)
        )
        metadata_path = os.path.join(
            output_dir, "{0}_metadata.json".format(config.output_name)
        )
        manifest_path = os.path.join(
            output_dir, "{0}_manifest.json".format(config.output_name)
        )

        input_text = self.build_input_text(config)
        with open(input_path, "w", encoding="utf-8") as fobj:
            fobj.write(input_text)

        preview_figure = self.create_geometry_figure(config)
        preview_figure.savefig(preview_path, dpi=160, bbox_inches="tight")

        metadata = {
            "created_at": datetime.now().isoformat(),
            "app": "{0} {1}".format(APP_TITLE, APP_VERSION),
            "config": asdict(config),
            "audit": {
                "messages": [asdict(msg) for msg in report.messages],
                "derived": report.derived,
            },
            "gprmax_processing_notes": self._gprmax_processing_notes(config),
        }
        with open(metadata_path, "w", encoding="utf-8") as fobj:
            json.dump(metadata, fobj, ensure_ascii=False, indent=2)

        return BuildArtifacts(
            output_dir=output_dir,
            input_path=input_path,
            preview_path=preview_path,
            metadata_path=metadata_path,
            manifest_path=manifest_path,
            geometry_view_path=os.path.join(
                output_dir, "{0}_geometry.vti".format(config.output_name)
            ),
        )

    def _gprmax_processing_notes(self, config: SimulationConfig) -> Dict[str, object]:
        return {
            "source": "gprMax FDTD simulation",
            "primary_format": "HDF5 .out",
            "preferred_bscan_dataset": "/rxs/rx1/Ez",
            "data_layout": "samples x traces; single A-scan is stored as samples x 1 by GUI readers",
            "dt_units": "seconds",
            "position_units": "meters",
            "requested_scan_step_m": config.scan_step,
            "effective_scan_step_m": config.effective_scan_step,
            "scan_step_cells": config.scan_step_cells,
            "recommended_velocity_m_per_ns": material_velocity(config.host_eps_r) / 1e9,
            "uav_lift_off_m": config.lift_off,
            "background_layers": config.background_layers,
            "simulation_processing_guidance": [
                "Keep the primary .out/_merged.out as raw FDTD field data; do not overwrite it with filtered or gained data.",
                "A realistic processing preview is background removal followed by mild time-varying gain; AGC is optional and can distort relative amplitudes.",
                "Use f-k migration only when evaluating migration behavior, with velocity from the known host permittivity.",
            ],
        }

    def create_geometry_figure(self, config: SimulationConfig) -> Figure:
        figure = Figure(figsize=(9, 5.4), dpi=120)
        ax = figure.add_subplot(111)
        ax.set_facecolor("#0f172a")
        figure.patch.set_facecolor("#111827")

        air_rect = Rectangle(
            (0.0, config.ground_surface_y),
            config.domain_x,
            config.domain_y - config.ground_surface_y,
            facecolor="#e2e8f0",
            edgecolor="#94a3b8",
            linewidth=1.0,
        )
        host_rect = Rectangle(
            (0.0, 0.0),
            config.domain_x,
            config.ground_surface_y,
            facecolor="#d4a373",
            edgecolor="#8d5a2b",
            linewidth=1.0,
        )
        ax.add_patch(host_rect)
        ax.add_patch(air_rect)

        for index, layer in enumerate(self._background_layers(config)):
            layer_rect = Rectangle(
                (0.0, float(layer["y_min"])),
                config.domain_x,
                float(layer["y_max"]) - float(layer["y_min"]),
                facecolor="#b08968" if index % 2 == 0 else "#a3a380",
                edgecolor="#6b4f2a",
                linewidth=0.8,
                alpha=0.65,
            )
            ax.add_patch(layer_rect)

        if config.target_shape == "crack":
            crack_segments = config.crack_segments()
            for crack in crack_segments:
                if crack.orientation == "angled":
                    crack_patch = Polygon(
                        crack.corners_xy(),
                        closed=True,
                        facecolor=self._crack_color(crack.material_name),
                        edgecolor="#0b1220",
                        linewidth=1.2,
                    )
                else:
                    x_min, x_max, y_min, y_max = crack.bounds()
                    crack_patch = Rectangle(
                        (x_min, y_min),
                        x_max - x_min,
                        y_max - y_min,
                        facecolor=self._crack_color(crack.material_name),
                        edgecolor="#0b1220",
                        linewidth=1.2,
                    )
                ax.add_patch(crack_patch)

            if config.use_curved_crack:
                points = config.curved_crack_points()
                if len(points) >= 2:
                    ax.plot(
                        [point[0] for point in points],
                        [point[1] for point in points],
                        color="#0f172a",
                        linewidth=1.0,
                        linestyle=":",
                    )
        elif config.target_shape == "cylinder":
            target_patch = Circle(
                (config.target_center_x, config.target_center_y),
                radius=config.target_radius,
                facecolor=self._target_color(config),
                edgecolor="#0b1220",
                linewidth=1.2,
            )
            ax.add_patch(target_patch)
        else:
            target_patch = Rectangle(
                (
                    config.target_center_x - 0.5 * config.target_size_x,
                    config.target_center_y - 0.5 * config.target_size_y,
                ),
                config.target_size_x,
                config.target_size_y,
                facecolor=self._target_color(config),
                edgecolor="#0b1220",
                linewidth=1.2,
            )
            ax.add_patch(target_patch)

        pml_x = 10 * config.dx
        pml_y = 10 * config.dy
        safe_rect = Rectangle(
            (pml_x, pml_y),
            config.domain_x - 2 * pml_x,
            config.domain_y - 2 * pml_y,
            facecolor="none",
            edgecolor="#38bdf8",
            linewidth=1.0,
            linestyle="--",
        )
        ax.add_patch(safe_rect)

        ax.plot(
            [config.source_start_x, config.source_end_x],
            [config.source_y, config.source_y],
            color="#22c55e",
            linewidth=2.0,
            label="Tx path",
        )
        ax.plot(
            [config.receiver_start_x, config.receiver_end_x],
            [config.receiver_y, config.receiver_y],
            color="#60a5fa",
            linewidth=2.0,
            label="Rx path",
        )

        ax.scatter(
            [config.source_start_x, config.receiver_start_x],
            [config.source_y, config.receiver_y],
            c=["#22c55e", "#60a5fa"],
            s=52,
            zorder=5,
        )

        ax.set_xlim(0.0, config.domain_x)
        ax.set_ylim(0.0, config.domain_y)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(config.title)
        ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.35)
        ax.legend(loc="upper right")
        figure.tight_layout()
        return figure

    def _custom_materials(self, config: SimulationConfig) -> List[MaterialSpec]:
        materials: Dict[str, MaterialSpec] = {}

        materials[config.host_name] = MaterialSpec(
            name=config.host_name,
            eps_r=config.host_eps_r,
            sigma=config.host_sigma,
            builtin=False,
        )

        for layer in self._background_layers(config):
            name = str(layer["name"])
            if name not in materials:
                materials[name] = MaterialSpec(
                    name=name,
                    eps_r=float(layer["eps_r"]),
                    sigma=float(layer["sigma"]),
                    builtin=False,
                )

        if config.target_name not in ["pec", "free_space"]:
            materials[config.target_name] = MaterialSpec(
                name=config.target_name,
                eps_r=config.target_eps_r,
                sigma=config.target_sigma,
                builtin=False,
            )

        if config.target_shape == "crack":
            for crack in config.crack_segments():
                if crack.material_name in ["pec", "free_space"]:
                    continue
                if crack.material_name not in materials:
                    materials[crack.material_name] = MaterialSpec(
                        name=crack.material_name,
                        eps_r=config.target_eps_r,
                        sigma=config.target_sigma,
                        builtin=False,
                    )

        return list(materials.values())

    def _host_box_line(self, config: SimulationConfig) -> str:
        return "#box: 0 0 0 {0:.3f} {1:.3f} {2:.3f} {3}".format(
            config.domain_x,
            config.ground_surface_y,
            config.dz,
            config.host_name,
        )

    def _background_layers(self, config: SimulationConfig) -> List[Dict[str, float]]:
        layers = []
        for layer in config.background_layers:
            y_min = float(layer["y_min"])
            y_max = float(layer["y_max"])
            if y_max <= y_min:
                continue
            layers.append(
                {
                    "name": str(layer["name"]),
                    "eps_r": float(layer["eps_r"]),
                    "sigma": float(layer["sigma"]),
                    "y_min": y_min,
                    "y_max": y_max,
                }
            )
        return layers

    def _background_layer_lines(self, config: SimulationConfig) -> List[str]:
        lines = []
        for layer in self._background_layers(config):
            lines.append(
                "#box: 0 {0:.3f} 0 {1:.3f} {2:.3f} {3:.3f} {4}".format(
                    layer["y_min"],
                    config.domain_x,
                    layer["y_max"],
                    config.dz,
                    layer["name"],
                )
            )
        return lines

    def _target_lines(self, config: SimulationConfig) -> List[str]:
        if config.target_shape == "crack":
            lines = []
            for crack in config.crack_segments():
                lines.extend(crack.input_lines(config.dz, {}))
            return lines

        if config.target_shape == "cylinder":
            return [
                "#cylinder: {0:.3f} {1:.3f} 0 {0:.3f} {1:.3f} {2:.3f} {3:.3f} {4}".format(
                    config.target_center_x,
                    config.target_center_y,
                    config.dz,
                    config.target_radius,
                    config.target_name,
                )
            ]
        x1 = config.target_center_x - 0.5 * config.target_size_x
        x2 = config.target_center_x + 0.5 * config.target_size_x
        y1 = config.target_center_y - 0.5 * config.target_size_y
        y2 = config.target_center_y + 0.5 * config.target_size_y
        return [
            "#box: {0:.3f} {1:.3f} 0 {2:.3f} {3:.3f} {4:.3f} {5}".format(
                x1,
                y1,
                x2,
                y2,
                config.dz,
                config.target_name,
            )
        ]

    def _make_output_dir(self, config: SimulationConfig) -> str:
        if config.timestamp_output:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dirname = "{0}_{1}".format(config.output_name, stamp)
        else:
            dirname = config.output_name
        return os.path.join(config.output_root, dirname)

    def _is_exact_official(self, config: SimulationConfig) -> bool:
        return (
            abs(config.domain_x - 0.240) < 1e-9
            and abs(config.domain_y - 0.210) < 1e-9
            and abs(config.dx - 0.002) < 1e-9
            and abs(config.dy - 0.002) < 1e-9
            and abs(config.time_window_ns - 3.0) < 1e-9
            and config.host_name == "half_space"
            and abs(config.host_eps_r - 6.0) < 1e-9
            and abs(config.host_sigma - 0.0) < 1e-9
            and abs(config.ground_surface_y - 0.170) < 1e-9
            and abs(config.source_start_x - 0.040) < 1e-9
            and abs(config.receiver_offset - 0.040) < 1e-9
            and abs(config.scan_step - 0.002) < 1e-9
            and config.n_traces == 60
            and abs(config.center_freq_mhz - 1500.0) < 1e-9
            and config.target_shape == "cylinder"
            and config.target_name == "pec"
            and abs(config.target_center_x - 0.120) < 1e-9
            and abs(config.target_center_y - 0.080) < 1e-9
            and abs(config.target_radius - 0.010) < 1e-9
            and not config.write_geometry_view
            and config.source_type == "hertzian_dipole"
            and config.waveform_type == "ricker"
            and config.source_polarisation == "z"
            and abs(config.source_resistance - 0.0) < 1e-9
        )

    def _build_exact_official_text(self) -> str:
        return (
            "#title: B-scan from a metal cylinder buried in a dielectric half-space\n"
            "#domain: 0.240 0.210 0.002\n"
            "#dx_dy_dz: 0.002 0.002 0.002\n"
            "#time_window: 3e-9\n"
            "\n"
            "#material: 6 0 1 0 half_space\n"
            "\n"
            "#waveform: ricker 1 1.5e9 my_ricker\n"
            "#hertzian_dipole: z 0.040 0.170 0 my_ricker\n"
            "#rx: 0.080 0.170 0\n"
            "#src_steps: 0.002 0 0\n"
            "#rx_steps: 0.002 0 0\n"
            "\n"
            "#box: 0 0 0 0.240 0.170 0.002 half_space\n"
            "#cylinder: 0.120 0.080 0 0.120 0.080 0.002 0.010 pec\n"
        )

    def _target_color(self, config: SimulationConfig) -> str:
        if config.target_name == "pec":
            return "#111827"
        if config.target_name in ["free_space", "air_void"]:
            return "#f8fafc"
        if "water" in config.target_name:
            return "#2563eb"
        return "#ef4444"

    def _crack_color(self, material_name: str) -> str:
        if material_name in ["pec", "metal"]:
            return "#111827"
        if material_name in ["free_space", "air"]:
            return "#f8fafc"
        if "water" in material_name:
            return "#2563eb"
        return "#ef4444"


class GprMaxRunner(object):
    def __init__(self, log_callback: Optional[Callable[[str], None]] = None):
        self.log_callback = log_callback or (lambda message: None)

    def log(self, message: str) -> None:
        try:
            self.log_callback(message)
        except UnicodeEncodeError:
            safe_message = message.encode("gbk", errors="replace").decode("gbk")
            self.log_callback(safe_message)

    def run(
        self, config: SimulationConfig, artifacts: BuildArtifacts
    ) -> BuildArtifacts:
        use_gpu = config.use_gpu
        if use_gpu and not self._nvcc_available():
            self.log("已勾选 GPU，但当前环境未找到 nvcc；自动回退到 CPU。")
            use_gpu = False

        return_code, output = self._run_process(config, artifacts, use_gpu)
        if return_code != 0 and use_gpu:
            gpu_failure_markers = [
                "pycuda.driver.CompileError",
                "nvcc preprocessing",
                "PyCUDA ERROR",
            ]
            if any(marker in output for marker in gpu_failure_markers):
                self.log("GPU 编译/运行失败，自动回退到 CPU 重试。")
                return_code, output = self._run_process(config, artifacts, False)

        if return_code != 0:
            raise RuntimeError("gprMax 运行失败，退出码为 {0}".format(return_code))

        if config.geometry_only:
            return artifacts

        basefilename = os.path.splitext(artifacts.input_path)[0]
        artifacts.output_out_paths = self._collect_output_out_paths(
            basefilename, config.n_traces
        )

        if config.n_traces > 1:
            self.log("使用 tools.outputfiles_merge 合并 A-scan")
            merge_files(basefilename)
            merged_path = basefilename + "_merged.out"
            if not os.path.exists(merged_path):
                raise RuntimeError("未生成合并输出文件: {0}".format(merged_path))
            artifacts.merged_out_path = merged_path
            artifacts.primary_out_path = merged_path
        else:
            single_path = basefilename + ".out"
            if not os.path.exists(single_path):
                raise RuntimeError("未生成输出文件: {0}".format(single_path))
            artifacts.primary_out_path = single_path
            self.log("单道 A-scan 输出文件: {0}".format(single_path))

        data, dt = self.load_merged_bscan(artifacts.primary_out_path)
        bscan_png = os.path.join(
            artifacts.output_dir,
            "{0}_bscan.png".format(
                os.path.splitext(os.path.basename(artifacts.input_path))[0]
            ),
        )
        figure = create_bscan_figure(data, dt)
        figure.savefig(bscan_png, dpi=160, bbox_inches="tight")
        artifacts.bscan_png_path = bscan_png

        background_removed = remove_horizontal_background(data)
        background_removed_png = os.path.join(
            artifacts.output_dir,
            "{0}_background_removed.png".format(
                os.path.splitext(os.path.basename(artifacts.input_path))[0]
            ),
        )
        figure = create_bscan_figure(
            background_removed,
            dt,
            title="背景抑制 Ez B-scan",
        )
        figure.savefig(background_removed_png, dpi=160, bbox_inches="tight")
        artifacts.background_removed_png_path = background_removed_png

        gained = apply_mild_time_gain(background_removed, dt)
        background_removed_gain_png = os.path.join(
            artifacts.output_dir,
            "{0}_background_removed_mild_gain.png".format(
                os.path.splitext(os.path.basename(artifacts.input_path))[0]
            ),
        )
        figure = create_bscan_figure(
            gained,
            dt,
            title="背景抑制 + 温和时间增益 Ez B-scan",
        )
        figure.savefig(background_removed_gain_png, dpi=160, bbox_inches="tight")
        artifacts.background_removed_gain_png_path = background_removed_gain_png

        self._write_manifest(config, artifacts)
        self.log("B-scan 预览已保存到 {0}".format(bscan_png))
        self.log("背景抑制预览已保存到 {0}".format(background_removed_png))
        self.log(
            "背景抑制 + 温和时间增益预览已保存到 {0}".format(
                background_removed_gain_png
            )
        )
        self.log("数据清单已保存到 {0}".format(artifacts.manifest_path))
        return artifacts

    def _run_process(
        self,
        config: SimulationConfig,
        artifacts: BuildArtifacts,
        use_gpu: bool,
    ) -> Tuple[int, str]:
        command = [
            config.python_executable or sys.executable,
            "-m",
            "gprMax",
            artifacts.input_path,
        ]

        if config.n_traces > 1:
            command.extend(["-n", str(config.n_traces)])
            if config.geometry_fixed:
                command.append("--geometry-fixed")
        if use_gpu:
            command.append("-gpu")
        if config.geometry_only:
            command.append("--geometry-only")

        self.log("运行模式: {0}".format("GPU" if use_gpu else "CPU"))
        self.log("Running: {0}".format(" ".join(command)))

        process = self._spawn_process(
            command=command,
            cwd=artifacts.output_dir,
            use_gpu=use_gpu,
        )

        output_lines = []
        t0 = time.perf_counter()
        if process.stdout is not None:
            for line in process.stdout:
                text = line.rstrip()
                output_lines.append(text)
                if text:
                    self.log(text)

        return_code = process.wait()
        elapsed = time.perf_counter() - t0
        self.log("仿真耗时: {0:.2f} s".format(elapsed))
        return return_code, "\n".join(output_lines)

    def _nvcc_available(self) -> bool:
        try:
            result = subprocess.run(
                "where nvcc",
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=15,
                shell=True,
            )
            return result.returncode == 0 and bool(result.stdout.strip())
        except Exception:
            return False

    def _spawn_process(
        self,
        command: List[str],
        cwd: str,
        use_gpu: bool,
    ) -> subprocess.Popen:
        if use_gpu:
            env = self._build_gpu_env()
            if env is not None:
                return subprocess.Popen(
                    command,
                    cwd=cwd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    universal_newlines=True,
                    bufsize=1,
                    encoding="utf-8",
                    errors="replace",
                    env=env,
                )

        env = os.environ.copy()
        return subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            universal_newlines=True,
            bufsize=1,
            encoding="utf-8",
            errors="replace",
            env=env,
        )

    def _build_gpu_env(self) -> Optional[Dict[str, str]]:
        vcvars = os.path.join(VS_ROOT, "VC", "Auxiliary", "Build", "vcvars64.bat")
        nvcc = os.path.join(CUDA_ROOT, "bin", "nvcc.exe")
        if not os.path.exists(vcvars) or not os.path.exists(nvcc):
            return None

        try:
            dump_env_cmd = (
                "call {0} >nul && "
                'set "CUDA_PATH={1}" && '
                'set "CUDA_HOME={1}" && '
                'set "GPRMAX_PYCUDA_CCBIN={2}" && '
                'set "PATH={4};{2};{3};%PATH%" && set'
            ).format(
                vcvars,
                CUDA_ROOT,
                MSVC_BIN,
                os.path.join(CUDA_ROOT, "bin")
                + ";"
                + os.path.join(CUDA_ROOT, "libnvvp"),
                PROJECT_ROOT,
            )
            result = subprocess.run(
                ["cmd.exe", "/d", "/s", "/c", dump_env_cmd],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=60,
            )
            if result.returncode != 0:
                return None

            env = os.environ.copy()
            for line in result.stdout.splitlines():
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                clean_key = key.strip().replace("\\", "").replace('"', "")
                clean_value = value.strip().strip('"')
                env[clean_key] = clean_value
            return env
        except Exception:
            return None

    def _collect_output_out_paths(self, basefilename: str, n_traces: int) -> List[str]:
        if n_traces <= 1:
            return [basefilename + ".out"]
        paths = []
        for trace in range(1, n_traces + 1):
            path = basefilename + str(trace) + ".out"
            if os.path.exists(path):
                paths.append(path)
        return paths

    def _json_safe(self, value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        if isinstance(value, (list, tuple)):
            return [self._json_safe(item) for item in value]
        return value

    def _summarise_out_file(self, path: str) -> Dict[str, object]:
        summary = {
            "path": path,
            "exists": os.path.exists(path),
        }
        if not os.path.exists(path):
            return summary

        with h5py.File(path, "r") as fobj:
            summary["attrs"] = {
                key: self._json_safe(fobj.attrs[key]) for key in fobj.attrs
            }
            receivers = {}
            if "rxs" in fobj:
                for rxname in fobj["rxs"]:
                    rxgroup = fobj["rxs"][rxname]
                    receivers[rxname] = {
                        "attrs": {
                            key: self._json_safe(rxgroup.attrs[key])
                            for key in rxgroup.attrs
                        },
                        "datasets": {
                            key: list(rxgroup[key].shape)
                            for key in rxgroup.keys()
                            if isinstance(rxgroup[key], h5py.Dataset)
                        },
                    }
            summary["receivers"] = receivers
        return summary

    def _write_manifest(
        self, config: SimulationConfig, artifacts: BuildArtifacts
    ) -> None:
        if not artifacts.manifest_path:
            return

        manifest = {
            "created_at": datetime.now().isoformat(),
            "app": "{0} {1}".format(APP_TITLE, APP_VERSION),
            "input_file": artifacts.input_path,
            "metadata_file": artifacts.metadata_path,
            "preview_file": artifacts.preview_path,
            "primary_out_file": artifacts.primary_out_path,
            "merged_out_file": artifacts.merged_out_path,
            "raw_out_files": artifacts.output_out_paths,
            "bscan_preview_file": artifacts.bscan_png_path,
            "background_removed_preview_file": artifacts.background_removed_png_path,
            "background_removed_mild_gain_preview_file": artifacts.background_removed_gain_png_path,
            "component": "Ez",
            "requested_scan_step_m": config.scan_step,
            "effective_scan_step_m": config.effective_scan_step,
            "scan_step_cells": config.scan_step_cells,
            "recommended_velocity_m_per_ns": material_velocity(config.host_eps_r) / 1e9,
            "uav_lift_off_m": config.lift_off,
            "background_layers": config.background_layers,
            "gprmax_notes": ScenarioBuilder()._gprmax_processing_notes(config),
            "primary_out_summary": self._summarise_out_file(artifacts.primary_out_path),
        }
        with open(artifacts.manifest_path, "w", encoding="utf-8") as fobj:
            json.dump(manifest, fobj, ensure_ascii=False, indent=2)

    def load_merged_bscan(self, merged_path: str) -> Tuple[np.ndarray, float]:
        with h5py.File(merged_path, "r") as fobj:
            dt = float(fobj.attrs["dt"])
            data = np.array(fobj["/rxs/rx1/Ez"][:], dtype=np.float32)
        if data.ndim == 1:
            data = data[:, np.newaxis]
        return data, dt


def remove_horizontal_background(data: np.ndarray) -> np.ndarray:
    if data.ndim == 1:
        return data - np.mean(data)
    return data - np.mean(data, axis=1, keepdims=True)


def apply_mild_time_gain(data: np.ndarray, dt: float) -> np.ndarray:
    if data.shape[0] == 0:
        return data
    time_ns = np.arange(data.shape[0], dtype=np.float32) * float(dt) * 1e9
    gain = 1.0 + 0.18 * time_ns
    gain = np.minimum(gain, 4.0)
    if data.ndim == 1:
        return data * gain
    return data * gain[:, np.newaxis]


def create_bscan_figure(
    data: np.ndarray, dt: float, title: str = "原始 Ez B-scan"
) -> Figure:
    figure = Figure(figsize=(9, 5.4), dpi=120)
    ax = figure.add_subplot(111)
    vmax = np.percentile(np.abs(data), 99.5)
    if vmax <= 0:
        vmax = 1.0
    image = ax.imshow(
        data,
        extent=[0, data.shape[1], data.shape[0] * dt * 1e9, 0],
        interpolation="nearest",
        aspect="auto",
        cmap="seismic",
        vmin=-vmax,
        vmax=vmax,
    )
    ax.set_xlabel("道号")
    ax.set_ylabel("时间 [ns]")
    ax.set_title(title)
    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.4)
    figure.colorbar(image, ax=ax, shrink=0.85, label="场强 [V/m]")
    figure.tight_layout()
    return figure


class PlotCanvas(QtWidgets.QWidget):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super(PlotCanvas, self).__init__(parent)
        self._layout = QtWidgets.QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self.canvas = FigureCanvas(Figure(figsize=(6, 4), dpi=110))
        self.canvas.figure.patch.set_facecolor("#111827")
        self._layout.addWidget(self.canvas)

    def replace_figure(self, figure: Figure) -> None:
        old_canvas = self.canvas
        self._layout.removeWidget(old_canvas)
        old_canvas.setParent(None)
        old_canvas.deleteLater()
        self.canvas = FigureCanvas(figure)
        self._layout.addWidget(self.canvas)


class RunnerThread(QtCore.QThread):
    progress = QtCore.Signal(int, str)
    log_message = QtCore.Signal(str)
    preview_ready = QtCore.Signal(str)
    input_ready = QtCore.Signal(str)
    audit_ready = QtCore.Signal(str)
    bscan_ready = QtCore.Signal(object, float)
    success = QtCore.Signal(str, object)
    failure = QtCore.Signal(str)

    def __init__(
        self,
        config: SimulationConfig,
        run_simulation: bool,
        parent: Optional[QtCore.QObject] = None,
    ):
        super(RunnerThread, self).__init__(parent)
        self.config = config
        self.run_simulation = run_simulation

    def run(self) -> None:
        try:
            auditor = PhysicsAuditor()
            builder = ScenarioBuilder()
            runner = GprMaxRunner(log_callback=self.log_message.emit)

            self.progress.emit(5, "执行物理审计")
            report = auditor.build_report(self.config)
            self.audit_ready.emit(report.to_html())
            if report.has_errors():
                raise RuntimeError("物理审计失败，请先修复红色错误项。")

            self.progress.emit(15, "生成输入文件")
            artifacts = builder.build_files(self.config, report)
            self.preview_ready.emit(artifacts.preview_path)
            with open(artifacts.input_path, "r", encoding="utf-8") as fobj:
                self.input_ready.emit(fobj.read())

            if not self.run_simulation:
                self.progress.emit(100, "构建完成")
                self.success.emit("模型文件已生成。", artifacts)
                return

            self.progress.emit(35, "运行 gprMax")
            artifacts = runner.run(self.config, artifacts)

            if artifacts.primary_out_path:
                data, dt = runner.load_merged_bscan(artifacts.primary_out_path)
                self.bscan_ready.emit(data, dt)

            self.progress.emit(100, "仿真完成")
            self.success.emit("仿真与合并已完成。", artifacts)
        except Exception:
            self.failure.emit(traceback.format_exc())


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("{0} {1}".format(APP_TITLE, APP_VERSION))
        self.resize(1600, 980)
        self.worker = None
        self.current_artifacts = None
        self.current_bscan_data = None
        self.current_bscan_dt = None
        self.builder = ScenarioBuilder()
        self.auditor = PhysicsAuditor()

        self._apply_theme()
        self._build_ui()
        self.apply_preset("official_cylinder_bscan")

    def _apply_theme(self) -> None:
        palette = QtGui.QPalette()
        palette.setColor(QtGui.QPalette.Window, QtGui.QColor("#111827"))
        palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor("#e5e7eb"))
        palette.setColor(QtGui.QPalette.Base, QtGui.QColor("#0f172a"))
        palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor("#1f2937"))
        palette.setColor(QtGui.QPalette.Text, QtGui.QColor("#e5e7eb"))
        palette.setColor(QtGui.QPalette.Button, QtGui.QColor("#1f2937"))
        palette.setColor(QtGui.QPalette.ButtonText, QtGui.QColor("#e5e7eb"))
        palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor("#2563eb"))
        palette.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor("#ffffff"))
        self.setPalette(palette)
        self.setStyleSheet(
            """
            QWidget { font-family: Segoe UI; font-size: 11px; }
            QMainWindow, QFrame, QSplitter, QScrollArea, QGroupBox { background: #111827; color: #e5e7eb; }
            QGroupBox { border: 1px solid #334155; border-radius: 8px; margin-top: 12px; padding-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; color: #93c5fd; }
            QLineEdit, QPlainTextEdit, QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox { background: #0f172a; border: 1px solid #334155; border-radius: 6px; padding: 4px 6px; color: #e5e7eb; }
            QPushButton { background: #1d4ed8; border: none; border-radius: 8px; padding: 8px 12px; color: white; font-weight: 600; }
            QPushButton:hover { background: #2563eb; }
            QPushButton:disabled { background: #374151; color: #9ca3af; }
            QLabel#titleLabel { font-size: 20px; font-weight: 700; color: #dbeafe; }
            QLabel#subTitleLabel { color: #94a3b8; }
            QProgressBar { background: #0f172a; border: 1px solid #334155; border-radius: 6px; text-align: center; }
            QProgressBar::chunk { background: #22c55e; border-radius: 5px; }
            QTabWidget::pane { border: 1px solid #334155; }
            QTabBar::tab { background: #1f2937; color: #e5e7eb; padding: 8px 12px; }
            QTabBar::tab:selected { background: #2563eb; }
            """
        )

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel(APP_TITLE)
        title.setObjectName("titleLabel")
        subtitle = QtWidgets.QLabel("基于手册的 2D 建模工作台")
        subtitle.setObjectName("subTitleLabel")
        title_box = QtWidgets.QVBoxLayout()
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header.addLayout(title_box)
        header.addStretch(1)

        self.preset_combo = QtWidgets.QComboBox()
        for preset_key, preset in PRESETS.items():
            self.preset_combo.addItem(preset["label"], preset_key)
        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        header.addWidget(QtWidgets.QLabel("预设"))
        header.addWidget(self.preset_combo)

        self.build_button = QtWidgets.QPushButton("生成输入文件")
        self.build_button.clicked.connect(self.on_build_only)
        header.addWidget(self.build_button)

        self.run_button = QtWidgets.QPushButton("生成并运行")
        self.run_button.clicked.connect(self.on_build_and_run)
        header.addWidget(self.run_button)
        layout.addLayout(header)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        layout.addWidget(splitter, 1)

        left_scroll = QtWidgets.QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setMinimumWidth(430)
        splitter.addWidget(left_scroll)

        left_container = QtWidgets.QWidget()
        left_scroll.setWidget(left_container)
        left_layout = QtWidgets.QVBoxLayout(left_container)
        left_layout.setSpacing(10)

        self._build_output_group(left_layout)
        self._build_domain_group(left_layout)
        self._build_material_group(left_layout)
        self._build_survey_group(left_layout)
        self._build_source_group(left_layout)
        self._build_target_group(left_layout)
        self._build_runtime_group(left_layout)
        left_layout.addStretch(1)

        right_widget = QtWidgets.QWidget()
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(1, 1)
        right_layout = QtWidgets.QVBoxLayout(right_widget)
        right_layout.setSpacing(10)

        self.audit_browser = QtWidgets.QTextBrowser()
        self.audit_browser.setMinimumHeight(220)
        right_layout.addWidget(self.audit_browser)

        self.tabs = QtWidgets.QTabWidget()
        right_layout.addWidget(self.tabs, 1)

        self.geometry_canvas = PlotCanvas()
        self.bscan_canvas = PlotCanvas()
        self.tabs.addTab(self.geometry_canvas, "几何预览")
        self.tabs.addTab(self.bscan_canvas, "B-scan")

        self.input_view = QtWidgets.QPlainTextEdit()
        self.input_view.setReadOnly(True)
        self.tabs.addTab(self.input_view, "生成的 .in")

        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.tabs.addTab(self.log_view, "运行日志")

        footer = QtWidgets.QHBoxLayout()
        self.status_label = QtWidgets.QLabel("就绪")
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        footer.addWidget(self.status_label)
        footer.addWidget(self.progress_bar, 1)

        save_button = QtWidgets.QPushButton("保存预设")
        save_button.clicked.connect(self.on_save_preset)
        footer.addWidget(save_button)

        load_button = QtWidgets.QPushButton("加载预设")
        load_button.clicked.connect(self.on_load_preset)
        footer.addWidget(load_button)

        self.open_result_button = QtWidgets.QPushButton("打开本次结果目录")
        self.open_result_button.setEnabled(False)
        self.open_result_button.clicked.connect(self.on_open_output)
        footer.addWidget(self.open_result_button)

        open_root_button = QtWidgets.QPushButton("打开输出根目录")
        open_root_button.clicked.connect(self.on_open_output_root)
        footer.addWidget(open_root_button)

        layout.addLayout(footer)

        self.refresh_timer = QtCore.QTimer(self)
        self.refresh_timer.setSingleShot(True)
        self.refresh_timer.setInterval(200)
        self.refresh_timer.timeout.connect(self.refresh_preview_and_audit)

    def _build_output_group(self, layout: QtWidgets.QVBoxLayout) -> None:
        group = QtWidgets.QGroupBox("输出")
        form = QtWidgets.QFormLayout(group)
        self.output_root_edit = QtWidgets.QLineEdit(r"D:\ClawX-Data\sim\gprmax_outcsv")
        browse = QtWidgets.QPushButton("浏览")
        browse.clicked.connect(self.on_browse_output_root)
        browse_row = QtWidgets.QHBoxLayout()
        browse_row.addWidget(self.output_root_edit)
        browse_row.addWidget(browse)
        form.addRow("输出根目录", self._wrap_layout(browse_row))
        self.output_name_edit = QtWidgets.QLineEdit("gpr_model")
        form.addRow("输出名前缀", self.output_name_edit)
        self.timestamp_check = QtWidgets.QCheckBox("输出目录加时间戳")
        self.timestamp_check.setChecked(True)
        form.addRow("", self.timestamp_check)
        layout.addWidget(group)

    def _build_domain_group(self, layout: QtWidgets.QVBoxLayout) -> None:
        group = QtWidgets.QGroupBox("计算域 / 宿主介质")
        form = QtWidgets.QFormLayout(group)
        self.domain_x_spin = self._double_spin(0.02, 20.0, 0.240, 3, 0.001)
        self.domain_y_spin = self._double_spin(0.02, 20.0, 0.210, 3, 0.001)
        self.dx_spin = self._double_spin(0.0005, 0.05, 0.002, 4, 0.0005)
        self.dy_spin = self._double_spin(0.0005, 0.05, 0.002, 4, 0.0005)
        self.ground_surface_y_spin = self._double_spin(0.001, 20.0, 0.170, 3, 0.001)
        self.lift_off_spin = self._double_spin(0.0, 5.0, 0.0, 3, 0.001)
        self.time_window_spin = self._double_spin(0.1, 200.0, 3.0, 3, 0.1)
        form.addRow("计算域 x (m)", self.domain_x_spin)
        form.addRow("计算域 y (m)", self.domain_y_spin)
        form.addRow("dx (m)", self.dx_spin)
        form.addRow("dy (m)", self.dy_spin)
        form.addRow("地表 y (m)", self.ground_surface_y_spin)
        form.addRow("离地高度 Lift-off (m)", self.lift_off_spin)
        form.addRow("时间窗 (ns)", self.time_window_spin)
        layout.addWidget(group)

    def _build_material_group(self, layout: QtWidgets.QVBoxLayout) -> None:
        group = QtWidgets.QGroupBox("材料")
        form = QtWidgets.QFormLayout(group)

        self.host_preset_combo = QtWidgets.QComboBox()
        for key in HOST_PRESETS:
            self.host_preset_combo.addItem(key, key)
        self.host_preset_combo.currentIndexChanged.connect(self._on_host_preset_changed)
        self.host_name_edit = QtWidgets.QLineEdit("half_space")
        self.host_eps_spin = self._double_spin(1.0, 100.0, 6.0, 3, 0.1)
        self.host_sigma_spin = self._double_spin(0.0, 10.0, 0.0, 6, 0.0001)

        self.target_preset_combo = QtWidgets.QComboBox()
        for key in TARGET_PRESETS:
            self.target_preset_combo.addItem(key, key)
        self.target_preset_combo.currentIndexChanged.connect(
            self._on_target_preset_changed
        )
        self.target_name_edit = QtWidgets.QLineEdit("pec")
        self.target_eps_spin = self._double_spin(1.0, 100.0, 9.0, 3, 0.1)
        self.target_sigma_spin = self._double_spin(0.0, 10.0, 0.0, 6, 0.0001)

        form.addRow("宿主预设", self.host_preset_combo)
        form.addRow("宿主名称", self.host_name_edit)
        form.addRow("宿主 eps_r", self.host_eps_spin)
        form.addRow("宿主 sigma (S/m)", self.host_sigma_spin)
        form.addRow(self._hline())
        form.addRow("目标预设", self.target_preset_combo)
        form.addRow("目标名称", self.target_name_edit)
        form.addRow("目标 eps_r", self.target_eps_spin)
        form.addRow("目标 sigma (S/m)", self.target_sigma_spin)
        layout.addWidget(group)

    def _build_survey_group(self, layout: QtWidgets.QVBoxLayout) -> None:
        group = QtWidgets.QGroupBox("测线参数")
        form = QtWidgets.QFormLayout(group)
        self.source_start_x_spin = self._double_spin(0.0, 20.0, 0.040, 3, 0.001)
        self.receiver_offset_spin = self._double_spin(0.0, 5.0, 0.040, 3, 0.001)
        self.scan_step_spin = self._double_spin(0.0005, 1.0, 0.002, 4, 0.0005)
        self.n_traces_spin = QtWidgets.QSpinBox()
        self.n_traces_spin.setRange(1, 5000)
        self.n_traces_spin.setValue(60)
        self.center_freq_spin = self._double_spin(10.0, 5000.0, 1500.0, 1, 10.0)
        form.addRow("发射源起始 x (m)", self.source_start_x_spin)
        form.addRow("Tx-Rx 间距 (m)", self.receiver_offset_spin)
        form.addRow("扫描步长 (m)", self.scan_step_spin)
        form.addRow("扫描道数", self.n_traces_spin)
        form.addRow("中心频率 (MHz)", self.center_freq_spin)
        layout.addWidget(group)

    def _build_source_group(self, layout: QtWidgets.QVBoxLayout) -> None:
        group = QtWidgets.QGroupBox("源与波形")
        form = QtWidgets.QFormLayout(group)

        self.source_type_combo = QtWidgets.QComboBox()
        self.source_type_combo.addItems(
            ["hertzian_dipole", "voltage_source", "transmission_line"]
        )
        self.source_type_combo.currentIndexChanged.connect(
            self._update_source_controls
        )

        self.waveform_type_combo = QtWidgets.QComboBox()
        self.waveform_type_combo.addItems(
            [
                "ricker",
                "gaussian",
                "gaussiandot",
                "gaussiandotnorm",
                "gaussiandotdot",
                "gaussiandotdotnorm",
                "sine",
                "contsine",
            ]
        )

        self.source_polarisation_combo = QtWidgets.QComboBox()
        self.source_polarisation_combo.addItems(["x", "y", "z"])
        self.source_polarisation_combo.setCurrentText("z")

        self.source_resistance_spin = self._double_spin(0.0, 1000.0, 0.0, 1, 1.0)

        form.addRow("源类型", self.source_type_combo)
        form.addRow("波形类型", self.waveform_type_combo)
        form.addRow("极化方向", self.source_polarisation_combo)
        form.addRow("源内阻 (Ω)", self.source_resistance_spin)

        self.source_hint_label = QtWidgets.QLabel(
            "Hertzian dipole：理想电流源（软源），无内阻概念。\n"
            "Voltage source：内阻为 0 时是硬源（完全反射）；非零时为有阻源。\n"
            "Transmission line：内阻需在 0–376.73 Ω 之间。"
        )
        self.source_hint_label.setWordWrap(True)
        self.source_hint_label.setStyleSheet("color:#94a3b8; font-size:10px;")
        form.addRow("", self.source_hint_label)

        layout.addWidget(group)
        self._update_source_controls()

    def _update_source_controls(self) -> None:
        is_hertzian = self.source_type_combo.currentText() == "hertzian_dipole"
        self.source_resistance_spin.setEnabled(not is_hertzian)
        if is_hertzian:
            self.source_hint_label.setText(
                "Hertzian dipole：理想电流源（软源），内阻参数无效。"
            )
        elif self.source_type_combo.currentText() == "voltage_source":
            self.source_hint_label.setText(
                "Voltage source：内阻为 0 时是硬源（波形结束后完全反射）；"
                "建议非零以模拟真实天线馈电。折叠偶极子典型值约 300 Ω。"
            )
        else:
            self.source_hint_label.setText(
                "Transmission line：特征阻抗需在 0–376.73 Ω 之间。"
                "折叠偶极子典型值约 300 Ω。"
            )

    def _build_target_group(self, layout: QtWidgets.QVBoxLayout) -> None:
        group = QtWidgets.QGroupBox("目标体")
        form = QtWidgets.QFormLayout(group)
        self.target_form = form
        self.target_shape_combo = QtWidgets.QComboBox()
        self.target_shape_combo.addItems(["cylinder", "box", "crack"])
        self.target_shape_combo.currentIndexChanged.connect(
            self._update_target_controls
        )
        self.target_orientation_combo = QtWidgets.QComboBox()
        self.target_orientation_combo.addItems(["horizontal", "vertical", "angled"])
        self.target_orientation_combo.currentIndexChanged.connect(
            self._update_target_controls
        )
        self.target_center_x_spin = self._double_spin(0.0, 20.0, 0.120, 3, 0.001)
        self.target_center_y_spin = self._double_spin(0.0, 20.0, 0.080, 3, 0.001)
        self.target_radius_spin = self._double_spin(0.001, 5.0, 0.010, 3, 0.001)
        self.target_width_spin = self._double_spin(0.001, 10.0, 0.020, 3, 0.001)
        self.target_height_spin = self._double_spin(0.001, 10.0, 0.020, 3, 0.001)
        self.target_angle_spin = self._double_spin(-89.0, 89.0, 30.0, 1, 1.0)
        self.use_curved_crack_check = QtWidgets.QCheckBox("使用曲折裂缝轨迹")
        self.use_curved_crack_check.toggled.connect(self._update_target_controls)
        self.crack_path_edit = QtWidgets.QPlainTextEdit()
        self.crack_path_edit.setPlaceholderText(
            "每行一个点，例如:\n0.120 0.200\n0.260 0.200"
        )
        self.crack_path_edit.setFixedHeight(110)
        self.curved_crack_example_button = QtWidgets.QPushButton("载入示例轨迹")
        self.curved_crack_example_button.clicked.connect(
            self.on_load_curved_crack_example
        )
        self.target_hint_label = QtWidgets.QLabel(
            "裂缝模式下：宽度=长度，高度=开度；方向支持 horizontal / vertical / angled。"
        )
        self.target_hint_label.setWordWrap(True)
        self.target_hint_label.setStyleSheet("color:#94a3b8;")
        form.addRow("形状", self.target_shape_combo)
        form.addRow("", self.use_curved_crack_check)
        form.addRow("裂缝方向", self.target_orientation_combo)
        form.addRow("裂缝倾角 (deg)", self.target_angle_spin)
        form.addRow("轨迹点 (x y)", self.crack_path_edit)
        form.addRow("", self.curved_crack_example_button)
        form.addRow("中心 x (m)", self.target_center_x_spin)
        form.addRow("中心 y (m)", self.target_center_y_spin)
        form.addRow("半径 (m)", self.target_radius_spin)
        form.addRow("宽度 (m)", self.target_width_spin)
        form.addRow("高度 (m)", self.target_height_spin)
        form.addRow("", self.target_hint_label)
        layout.addWidget(group)
        self._update_target_controls()

    def _build_runtime_group(self, layout: QtWidgets.QVBoxLayout) -> None:
        group = QtWidgets.QGroupBox("运行设置")
        form = QtWidgets.QFormLayout(group)
        self.python_edit = QtWidgets.QLineEdit(DEFAULT_PYTHON)
        self.use_gpu_check = QtWidgets.QCheckBox("使用 GPU")
        self.use_gpu_check.setChecked(True)
        self.geometry_fixed_check = QtWidgets.QCheckBox(
            "固定几何加速 (--geometry-fixed)"
        )
        self.geometry_fixed_check.setChecked(True)
        self.geometry_only_check = QtWidgets.QCheckBox("仅构建几何 Geometry Only")
        self.write_geometry_check = QtWidgets.QCheckBox("写出 #geometry_view")
        form.addRow("Python", self.python_edit)
        form.addRow("", self.use_gpu_check)
        form.addRow("", self.geometry_fixed_check)
        form.addRow("", self.geometry_only_check)
        form.addRow("", self.write_geometry_check)
        layout.addWidget(group)

        widgets = [
            self.output_root_edit,
            self.output_name_edit,
            self.domain_x_spin,
            self.domain_y_spin,
            self.dx_spin,
            self.dy_spin,
            self.ground_surface_y_spin,
            self.lift_off_spin,
            self.time_window_spin,
            self.host_preset_combo,
            self.host_name_edit,
            self.host_eps_spin,
            self.host_sigma_spin,
            self.target_preset_combo,
            self.target_name_edit,
            self.target_eps_spin,
            self.target_sigma_spin,
            self.source_start_x_spin,
            self.receiver_offset_spin,
            self.scan_step_spin,
            self.n_traces_spin,
            self.center_freq_spin,
            self.target_shape_combo,
            self.use_curved_crack_check,
            self.target_orientation_combo,
            self.target_center_x_spin,
            self.target_center_y_spin,
            self.target_radius_spin,
            self.target_width_spin,
            self.target_height_spin,
            self.target_angle_spin,
            self.crack_path_edit,
            self.python_edit,
            self.use_gpu_check,
            self.geometry_fixed_check,
            self.geometry_only_check,
            self.write_geometry_check,
            self.timestamp_check,
            self.source_type_combo,
            self.waveform_type_combo,
            self.source_polarisation_combo,
            self.source_resistance_spin,
        ]
        for widget in widgets:
            self._bind_refresh(widget)

    def _bind_refresh(self, widget: QtWidgets.QWidget) -> None:
        if isinstance(widget, QtWidgets.QLineEdit):
            widget.textChanged.connect(self.schedule_refresh)
        elif isinstance(widget, QtWidgets.QPlainTextEdit):
            widget.textChanged.connect(self.schedule_refresh)
        elif isinstance(widget, QtWidgets.QComboBox):
            widget.currentIndexChanged.connect(self.schedule_refresh)
        elif isinstance(widget, QtWidgets.QAbstractSpinBox):
            widget.valueChanged.connect(self.schedule_refresh)
        elif isinstance(widget, QtWidgets.QCheckBox):
            widget.toggled.connect(self.schedule_refresh)

    def _double_spin(
        self,
        minimum: float,
        maximum: float,
        value: float,
        decimals: int,
        step: float,
    ) -> QtWidgets.QDoubleSpinBox:
        widget = QtWidgets.QDoubleSpinBox()
        widget.setRange(minimum, maximum)
        widget.setDecimals(decimals)
        widget.setSingleStep(step)
        widget.setValue(value)
        return widget

    def _wrap_layout(self, layout: QtWidgets.QLayout) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        return widget

    def _hline(self) -> QtWidgets.QWidget:
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setStyleSheet("color:#334155;")
        return line

    def _on_preset_changed(self) -> None:
        self.apply_preset(self.preset_combo.currentData())

    def _on_host_preset_changed(self) -> None:
        preset_key = self.host_preset_combo.currentData()
        preset = HOST_PRESETS[preset_key]
        self.host_name_edit.setText(preset["name"])
        self.host_eps_spin.setValue(preset["eps_r"])
        self.host_sigma_spin.setValue(preset["sigma"])

    def _on_target_preset_changed(self) -> None:
        preset_key = self.target_preset_combo.currentData()
        preset = TARGET_PRESETS[preset_key]
        self.target_name_edit.setText(preset["name"])
        builtin = preset["builtin"]
        self.target_eps_spin.setEnabled(not builtin)
        self.target_sigma_spin.setEnabled(not builtin)
        if not builtin:
            self.target_eps_spin.setValue(float(preset["eps_r"]))
            self.target_sigma_spin.setValue(float(preset["sigma"]))

    def _update_target_controls(self) -> None:
        is_cylinder = self.target_shape_combo.currentText() == "cylinder"
        is_crack = self.target_shape_combo.currentText() == "crack"
        is_curved_crack = is_crack and self.use_curved_crack_check.isChecked()
        is_angled_crack = (
            is_crack
            and not is_curved_crack
            and self.target_orientation_combo.currentText() == "angled"
        )

        self.target_radius_spin.setEnabled(is_cylinder)
        self.target_center_x_spin.setEnabled(not is_curved_crack)
        self.target_center_y_spin.setEnabled(not is_curved_crack)
        self.target_width_spin.setEnabled(not is_cylinder and not is_curved_crack)
        self.target_height_spin.setEnabled(not is_cylinder)
        self.target_orientation_combo.setEnabled(is_crack and not is_curved_crack)

        self.target_angle_spin.setEnabled(is_angled_crack)
        self.target_angle_spin.setVisible(is_angled_crack)
        angle_label = self.target_form.labelForField(self.target_angle_spin)
        if angle_label is not None:
            angle_label.setVisible(is_angled_crack)

        self.use_curved_crack_check.setVisible(is_crack)
        path_label = self.target_form.labelForField(self.crack_path_edit)
        if path_label is not None:
            path_label.setVisible(is_curved_crack)
        self.crack_path_edit.setVisible(is_curved_crack)
        self.curved_crack_example_button.setVisible(is_curved_crack)

        if is_cylinder:
            self.target_hint_label.setText("圆柱体模式：设置半径和中心坐标。")
            self.target_hint_label.setVisible(True)
        elif is_curved_crack:
            self.target_hint_label.setText(
                "曲折裂缝模式：每行一个轨迹点 `x y`；高度表示裂缝开度。"
            )
            self.target_hint_label.setVisible(True)
        elif is_crack:
            self.target_hint_label.setText(
                "裂缝模式：宽度=长度，高度=开度；方向支持 horizontal / vertical / angled。"
            )
            self.target_hint_label.setVisible(True)
        else:
            self.target_hint_label.setText("盒子模式：设置宽度、高度和中心坐标。")
            self.target_hint_label.setVisible(True)

    def apply_preset(self, preset_key: str) -> None:
        if not preset_key or preset_key not in PRESETS:
            return
        preset = PRESETS[preset_key]
        self.preset_combo.blockSignals(True)
        index = self.preset_combo.findData(preset_key)
        if index >= 0:
            self.preset_combo.setCurrentIndex(index)
        self.preset_combo.blockSignals(False)

        self.output_name_edit.setText("gpr_model")
        self.domain_x_spin.setValue(preset["domain_x"])
        self.domain_y_spin.setValue(preset["domain_y"])
        self.dx_spin.setValue(preset["dx"])
        self.dy_spin.setValue(preset["dy"])
        self.time_window_spin.setValue(preset["time_window_ns"])
        self.ground_surface_y_spin.setValue(preset["ground_surface_y"])
        self.lift_off_spin.setValue(preset["lift_off"])
        self.source_start_x_spin.setValue(preset["source_start_x"])
        self.receiver_offset_spin.setValue(preset["receiver_offset"])
        self.scan_step_spin.setValue(preset["scan_step"])
        self.n_traces_spin.setValue(preset["n_traces"])
        self.center_freq_spin.setValue(preset["center_freq_mhz"])
        self.target_shape_combo.setCurrentText(preset["target_shape"])
        self.target_center_x_spin.setValue(preset["target_center_x"])
        self.target_center_y_spin.setValue(preset["target_center_y"])
        self.target_radius_spin.setValue(preset["target_radius"])
        self.target_width_spin.setValue(preset["target_width"])
        self.target_height_spin.setValue(preset["target_height"])
        self.target_orientation_combo.setCurrentText(preset["target_orientation"])
        self.target_angle_spin.setValue(preset["target_angle_deg"])
        self.use_curved_crack_check.setChecked(
            bool(preset.get("use_curved_crack", False))
        )
        self.crack_path_edit.setPlainText(str(preset.get("crack_path_text", "")))
        self.write_geometry_check.setChecked(preset["write_geometry_view"])
        self.geometry_only_check.setChecked(preset["geometry_only"])
        self.host_preset_combo.setCurrentIndex(
            self.host_preset_combo.findData(preset["host_preset"])
        )
        self.host_name_edit.setText(preset["host_name"])
        self.host_eps_spin.setValue(preset["host_eps_r"])
        self.host_sigma_spin.setValue(preset["host_sigma"])
        self.target_preset_combo.setCurrentIndex(
            self.target_preset_combo.findData(preset["target_preset"])
        )
        self.target_name_edit.setText(preset["target_name"])
        self.target_eps_spin.setValue(preset["target_eps_r"])
        self.target_sigma_spin.setValue(preset["target_sigma"])
        self.source_type_combo.setCurrentText(preset.get("source_type", "hertzian_dipole"))
        self.waveform_type_combo.setCurrentText(preset.get("waveform_type", "ricker"))
        self.source_polarisation_combo.setCurrentText(
            preset.get("source_polarisation", "z")
        )
        self.source_resistance_spin.setValue(preset.get("source_resistance", 0.0))
        self._update_target_controls()
        self._update_source_controls()
        self.schedule_refresh()

    def on_load_curved_crack_example(self) -> None:
        self.target_shape_combo.setCurrentText("crack")
        self.use_curved_crack_check.setChecked(True)
        self.crack_path_edit.setPlainText(DEFAULT_CURVED_CRACK_PATH_TEXT)
        self.schedule_refresh()

    def schedule_refresh(self) -> None:
        self.refresh_timer.start()

    def refresh_preview_and_audit(self) -> None:
        try:
            config = self.read_config()
            report = self.auditor.build_report(config)
            self.audit_browser.setHtml(report.to_html())
            figure = self.builder.create_geometry_figure(config)
            self.geometry_canvas.replace_figure(figure)
            self.input_view.setPlainText(self.builder.build_input_text(config))
            self.build_button.setEnabled(not report.has_errors())
            self.run_button.setEnabled(not report.has_errors())
            if report.has_errors():
                self.status_label.setText("物理审计存在错误")
            else:
                self.status_label.setText("就绪")
        except Exception as exc:
            self.audit_browser.setHtml(
                "<p style='color:#ff6b6b; font-family:Segoe UI;'><b>界面参数解析错误</b><br>{0}</p>".format(
                    html.escape(str(exc))
                )
            )
            self.build_button.setEnabled(False)
            self.run_button.setEnabled(False)

    def read_config(self) -> SimulationConfig:
        target_preset = self.target_preset_combo.currentData()
        target_builtin = TARGET_PRESETS[target_preset]["builtin"]
        target_eps = 0.0 if target_builtin else self.target_eps_spin.value()
        target_sigma = 0.0 if target_builtin else self.target_sigma_spin.value()
        return SimulationConfig(
            title=PRESETS[self.preset_combo.currentData()]["title"],
            output_root=self.output_root_edit.text().strip(),
            output_name=self.output_name_edit.text().strip() or "gpr_model",
            python_executable=self.python_edit.text().strip() or sys.executable,
            use_gpu=self.use_gpu_check.isChecked(),
            geometry_fixed=self.geometry_fixed_check.isChecked(),
            geometry_only=self.geometry_only_check.isChecked(),
            timestamp_output=self.timestamp_check.isChecked(),
            write_geometry_view=self.write_geometry_check.isChecked(),
            domain_x=self.domain_x_spin.value(),
            domain_y=self.domain_y_spin.value(),
            dx=self.dx_spin.value(),
            dy=self.dy_spin.value(),
            time_window_ns=self.time_window_spin.value(),
            host_name=self.host_name_edit.text().strip() or "host_material",
            host_eps_r=self.host_eps_spin.value(),
            host_sigma=self.host_sigma_spin.value(),
            ground_surface_y=self.ground_surface_y_spin.value(),
            lift_off=self.lift_off_spin.value(),
            source_start_x=self.source_start_x_spin.value(),
            receiver_offset=self.receiver_offset_spin.value(),
            scan_step=self.scan_step_spin.value(),
            n_traces=self.n_traces_spin.value(),
            center_freq_mhz=self.center_freq_spin.value(),
            target_shape=self.target_shape_combo.currentText(),
            target_name=self.target_name_edit.text().strip() or "pec",
            target_eps_r=target_eps,
            target_sigma=target_sigma,
            target_center_x=self.target_center_x_spin.value(),
            target_center_y=self.target_center_y_spin.value(),
            target_radius=self.target_radius_spin.value(),
            target_width=self.target_width_spin.value(),
            target_height=self.target_height_spin.value(),
            target_orientation=self.target_orientation_combo.currentText(),
            target_angle_deg=self.target_angle_spin.value(),
            use_curved_crack=self.use_curved_crack_check.isChecked(),
            crack_path_text=self.crack_path_edit.toPlainText().strip(),
            background_layers=list(
                PRESETS[self.preset_combo.currentData()].get("background_layers", [])
            ),
            source_type=self.source_type_combo.currentText(),
            waveform_type=self.waveform_type_combo.currentText(),
            source_resistance=self.source_resistance_spin.value(),
            source_polarisation=self.source_polarisation_combo.currentText(),
            preset_key=self.preset_combo.currentData(),
        )

    def on_build_only(self) -> None:
        self.start_worker(run_simulation=False)

    def on_build_and_run(self) -> None:
        self.start_worker(run_simulation=True)

    def start_worker(self, run_simulation: bool) -> None:
        if self.worker is not None and self.worker.isRunning():
            QtWidgets.QMessageBox.warning(self, "忙碌", "已有任务正在运行。")
            return
        config = self.read_config()
        report = self.auditor.build_report(config)
        self.audit_browser.setHtml(report.to_html())
        if report.has_errors():
            QtWidgets.QMessageBox.warning(
                self, "物理审计失败", "请先修复红色错误项，再继续。"
            )
            return
        self.log_view.clear()
        self.progress_bar.setValue(0)
        self.status_label.setText("执行中...")
        self.worker = RunnerThread(
            config=config, run_simulation=run_simulation, parent=self
        )
        self.worker.progress.connect(self.on_progress)
        self.worker.log_message.connect(self.log)
        self.worker.preview_ready.connect(self.on_preview_ready)
        self.worker.input_ready.connect(self.input_view.setPlainText)
        self.worker.audit_ready.connect(self.audit_browser.setHtml)
        self.worker.bscan_ready.connect(self.on_bscan_ready)
        self.worker.success.connect(self.on_success)
        self.worker.failure.connect(self.on_failure)
        self.worker.start()

    def on_progress(self, value: int, text: str) -> None:
        self.progress_bar.setValue(value)
        self.status_label.setText(text)

    def on_preview_ready(self, preview_path: str) -> None:
        if not preview_path or not os.path.exists(preview_path):
            return
        try:
            figure = self.builder.create_geometry_figure(self.read_config())
            self.geometry_canvas.replace_figure(figure)
        except Exception:
            self.log(traceback.format_exc())

    def on_bscan_ready(self, data: np.ndarray, dt: float) -> None:
        self.current_bscan_data = data
        self.current_bscan_dt = dt
        figure = create_bscan_figure(data, dt)
        self.bscan_canvas.replace_figure(figure)
        self.tabs.setCurrentWidget(self.bscan_canvas)

    def on_success(self, message: str, artifacts: BuildArtifacts) -> None:
        self.current_artifacts = artifacts
        self.progress_bar.setValue(100)
        self.status_label.setText("完成")
        self.log(message)
        self.log("结果目录: {0}".format(artifacts.output_dir))
        if artifacts.primary_out_path:
            self.log("主输出 .out: {0}".format(artifacts.primary_out_path))
        if artifacts.manifest_path and os.path.exists(artifacts.manifest_path):
            self.log("数据清单: {0}".format(artifacts.manifest_path))
        self.open_result_button.setEnabled(True)

    def on_failure(self, trace: str) -> None:
        self.progress_bar.setValue(0)
        self.status_label.setText("失败")
        self.log(trace)
        QtWidgets.QMessageBox.critical(self, "任务失败", trace[:4000])

    def on_browse_output_root(self) -> None:
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, "选择输出目录", self.output_root_edit.text()
        )
        if directory:
            self.output_root_edit.setText(directory)

    def on_save_preset(self) -> None:
        config = self.read_config()
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "保存预设", self.output_root_edit.text(), "JSON 文件 (*.json)"
        )
        if not path:
            return
        with open(path, "w", encoding="utf-8") as fobj:
            json.dump(asdict(config), fobj, ensure_ascii=False, indent=2)
        self.log("预设已保存到 {0}".format(path))

    def on_load_preset(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "加载预设", self.output_root_edit.text(), "JSON 文件 (*.json)"
        )
        if not path:
            return
        with open(path, "r", encoding="utf-8") as fobj:
            data = json.load(fobj)
        self._load_config_dict(data)
        self.log("已从 {0} 加载预设".format(path))

    def on_open_output(self) -> None:
        if self.current_artifacts is None or not self.current_artifacts.output_dir:
            QtWidgets.QMessageBox.information(
                self,
                "暂无结果",
                "当前还没有生成结果目录，请先执行“生成输入文件”或“生成并运行”。",
            )
            return

        target = self.current_artifacts.output_dir
        if os.path.exists(target):
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(target))
        else:
            QtWidgets.QMessageBox.warning(
                self,
                "目录不存在",
                "结果目录不存在：{0}".format(target),
            )

    def on_open_output_root(self) -> None:
        target = self.output_root_edit.text().strip()
        if target and os.path.exists(target):
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(target))
        else:
            QtWidgets.QMessageBox.warning(
                self,
                "目录不存在",
                "输出根目录不存在：{0}".format(target),
            )

    def _load_config_dict(self, data: Dict[str, object]) -> None:
        if "preset_key" in data and data["preset_key"] in PRESETS:
            self.apply_preset(str(data["preset_key"]))

        mappings = [
            (self.output_root_edit, "output_root"),
            (self.output_name_edit, "output_name"),
            (self.python_edit, "python_executable"),
        ]
        for widget, key in mappings:
            if key in data:
                widget.setText(str(data[key]))

        spin_mappings = [
            (self.domain_x_spin, "domain_x"),
            (self.domain_y_spin, "domain_y"),
            (self.dx_spin, "dx"),
            (self.dy_spin, "dy"),
            (self.time_window_spin, "time_window_ns"),
            (self.ground_surface_y_spin, "ground_surface_y"),
            (self.lift_off_spin, "lift_off"),
            (self.source_start_x_spin, "source_start_x"),
            (self.receiver_offset_spin, "receiver_offset"),
            (self.scan_step_spin, "scan_step"),
            (self.center_freq_spin, "center_freq_mhz"),
            (self.target_center_x_spin, "target_center_x"),
            (self.target_center_y_spin, "target_center_y"),
            (self.target_radius_spin, "target_radius"),
            (self.target_width_spin, "target_width"),
            (self.target_height_spin, "target_height"),
            (self.target_angle_spin, "target_angle_deg"),
            (self.host_eps_spin, "host_eps_r"),
            (self.host_sigma_spin, "host_sigma"),
            (self.target_eps_spin, "target_eps_r"),
            (self.target_sigma_spin, "target_sigma"),
        ]
        for widget, key in spin_mappings:
            if key in data:
                widget.setValue(float(data[key]))

        if "n_traces" in data:
            self.n_traces_spin.setValue(int(data["n_traces"]))
        if "host_name" in data:
            self.host_name_edit.setText(str(data["host_name"]))
        if "target_name" in data:
            self.target_name_edit.setText(str(data["target_name"]))
        if "target_shape" in data:
            self.target_shape_combo.setCurrentText(str(data["target_shape"]))
        if "target_orientation" in data:
            self.target_orientation_combo.setCurrentText(
                str(data["target_orientation"])
            )
        if "use_curved_crack" in data:
            self.use_curved_crack_check.setChecked(bool(data["use_curved_crack"]))
        if "crack_path_text" in data:
            self.crack_path_edit.setPlainText(str(data["crack_path_text"]))
        if "use_gpu" in data:
            self.use_gpu_check.setChecked(bool(data["use_gpu"]))
        if "geometry_fixed" in data:
            self.geometry_fixed_check.setChecked(bool(data["geometry_fixed"]))
        if "geometry_only" in data:
            self.geometry_only_check.setChecked(bool(data["geometry_only"]))
        if "timestamp_output" in data:
            self.timestamp_check.setChecked(bool(data["timestamp_output"]))
        if "write_geometry_view" in data:
            self.write_geometry_check.setChecked(bool(data["write_geometry_view"]))
        if "source_type" in data:
            self.source_type_combo.setCurrentText(str(data["source_type"]))
        if "waveform_type" in data:
            self.waveform_type_combo.setCurrentText(str(data["waveform_type"]))
        if "source_polarisation" in data:
            self.source_polarisation_combo.setCurrentText(
                str(data["source_polarisation"])
            )
        if "source_resistance" in data:
            self.source_resistance_spin.setValue(float(data["source_resistance"]))
        self._update_target_controls()
        self._update_source_controls()
        self.schedule_refresh()

    def log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_view.appendPlainText("[{0}] {1}".format(timestamp, message))


def build_smoke_config(args: argparse.Namespace) -> SimulationConfig:
    preset_key = getattr(args, "smoke_preset", "official_cylinder_bscan")
    preset = PRESETS[preset_key]
    return SimulationConfig(
        title=preset["title"],
        output_root=args.output_root,
        output_name="{0}_smoke".format(preset_key),
        python_executable=args.python or DEFAULT_PYTHON,
        timestamp_output=True,
        use_gpu=args.gpu,
        geometry_fixed=True,
        geometry_only=False,
        write_geometry_view=False,
        domain_x=preset["domain_x"],
        domain_y=preset["domain_y"],
        dx=preset["dx"],
        dy=preset["dy"],
        time_window_ns=preset["time_window_ns"],
        host_name=preset["host_name"],
        host_eps_r=preset["host_eps_r"],
        host_sigma=preset["host_sigma"],
        ground_surface_y=preset["ground_surface_y"],
        lift_off=preset["lift_off"],
        source_start_x=preset["source_start_x"],
        receiver_offset=preset["receiver_offset"],
        scan_step=preset["scan_step"],
        n_traces=args.traces,
        center_freq_mhz=preset["center_freq_mhz"],
        target_shape=preset["target_shape"],
        target_name=preset["target_name"],
        target_eps_r=preset["target_eps_r"],
        target_sigma=preset["target_sigma"],
        target_center_x=preset["target_center_x"],
        target_center_y=preset["target_center_y"],
        target_radius=preset["target_radius"],
        target_width=preset["target_width"],
        target_height=preset["target_height"],
        target_orientation=preset["target_orientation"],
        target_angle_deg=preset["target_angle_deg"],
        use_curved_crack=bool(preset.get("use_curved_crack", False)),
        crack_path_text=str(preset.get("crack_path_text", "")),
        background_layers=list(preset.get("background_layers", [])),
        source_type=preset.get("source_type", "hertzian_dipole"),
        waveform_type=preset.get("waveform_type", "ricker"),
        source_resistance=preset.get("source_resistance", 0.0),
        source_polarisation=preset.get("source_polarisation", "z"),
        preset_key=preset_key,
    )


def run_smoke_test(args: argparse.Namespace) -> int:
    config = build_smoke_config(args)

    auditor = PhysicsAuditor()
    report = auditor.build_report(config)
    if report.has_errors():
        print("Smoke test audit failed")
        for message in report.messages:
            print("[{0}] {1}".format(message.level.upper(), message.text))
        return 1

    builder = ScenarioBuilder()
    artifacts = builder.build_files(config, report)
    expected_input = builder._build_exact_official_text()
    with open(artifacts.input_path, "r", encoding="utf-8") as fobj:
        built_input = fobj.read()
    if (
        config.preset_key == "official_cylinder_bscan"
        and args.traces == 60
        and built_input.rstrip() != expected_input.rstrip()
    ):
        print("Official preset input mismatch")
        return 1

    print("Built input at {0}".format(artifacts.input_path))
    runner = GprMaxRunner(log_callback=print)
    artifacts = runner.run(config, artifacts)
    if not artifacts.primary_out_path or not os.path.exists(artifacts.primary_out_path):
        print("Primary output missing")
        return 1
    if args.traces > 1 and (
        not artifacts.merged_out_path or not os.path.exists(artifacts.merged_out_path)
    ):
        print("Merged output missing")
        return 1
    if not artifacts.manifest_path or not os.path.exists(artifacts.manifest_path):
        print("Manifest missing")
        return 1

    data, dt = runner.load_merged_bscan(artifacts.primary_out_path)
    print("Smoke test primary output: shape={0}, dt={1}".format(data.shape, dt))
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PySide6 GUI for physically-correct gprMax model building"
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="build and run a preset without opening the GUI",
    )
    parser.add_argument(
        "--smoke-preset",
        default="official_cylinder_bscan",
        choices=sorted(PRESETS.keys()),
        help="preset key to use with --smoke-test",
    )
    parser.add_argument(
        "--output-root",
        default=r"D:\ClawX-Data\sim\gprmax_outcsv",
        help="output root for smoke test",
    )
    parser.add_argument(
        "--python", default=DEFAULT_PYTHON, help="python executable for gprMax"
    )
    parser.add_argument(
        "--traces", type=int, default=20, help="trace count for smoke test"
    )
    parser.add_argument("--gpu", action="store_true", help="use GPU for smoke test")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.smoke_test:
        return run_smoke_test(args)

    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName(APP_TITLE)
    app.setOrganizationName("gprMax")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
