#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPR 裂缝模型生成器 - 生成线性裂缝的GPR模拟数据
"""

import os
import numpy as np
from scipy.spatial import ConvexHull
from skimage.draw import polygon, line
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, Any, Optional


def generate_crack_model(
    output_dir: str = r"D:\ClawX-Data\sim\gprmax_outcsv",
    output_filename: str = "crack_model",
    model_x: float = 5.0,
    model_y: float = 2.0,
    dx: float = 0.005,
    dy: float = 0.005,
    # 裂缝参数
    crack_start_x: float = 1.0,  # 裂缝起点 X
    crack_start_y: float = 0.3,  # 裂缝起点 Y (深度)
    crack_end_x: float = 4.0,    # 裂缝终点 X
    crack_end_y: float = 1.5,    # 裂缝终点 Y (深度)
    crack_width: float = 0.02,   # 裂缝宽度 (m)
    crack_type: str = "air",     # "air"=空气填充, "water"=水填充
    # 土壤参数
    water_table_depth: float = 1.0,
    # 扫描参数
    n_traces: int = 50,
    random_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """生成含裂缝的GPR模型"""
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_dir, f"crack_model_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    nx = int(model_x / dx)
    ny = int(model_y / dy)
    
    # 材料 ID
    MAT_IDS = {
        "free_space": 0,
        "dry_soil": 1,
        "saturated_soil": 2,
        "rock": 3,
        "crack": 4,  # 裂缝 (空气或水填充)
    }
    
    # 初始化模型
    model = np.zeros((ny, nx), dtype=np.int16)
    
    # 1. 生成地表
    x_indices = np.arange(nx)
    surface_y_grid = int(ny * 0.85)
    surface_line = np.full(nx, surface_y_grid)
    
    # 2. 填充土壤
    water_table_y_grid = int(ny - (water_table_depth / dy))
    
    for ix in range(nx):
        surface_idx = int(surface_line[ix])
        dry_soil_end = min(surface_idx, water_table_y_grid)
        
        if surface_idx > dry_soil_end:
            model[dry_soil_end:surface_idx, ix] = MAT_IDS["dry_soil"]
        if dry_soil_end > 0:
            model[0:dry_soil_end, ix] = MAT_IDS["saturated_soil"]
    
    # 3. 生成裂缝 (关键部分)
    # 将物理坐标转换为网格坐标
    x1, y1 = int(crack_start_x / dx), int(crack_start_y / dy)
    x2, y2 = int(crack_end_x / dx), int(crack_end_y / dy)
    width = int(crack_width / dx)
    
    # 使用 Bresenham 算法画线
    rr, cc = line(y1, x1, y2, x2)
    
    # 扩展线宽
    for offset in range(-width//2, width//2 + 1):
        rr_off = np.clip(rr + offset, 0, ny - 1)
        cc_off = np.clip(cc + offset, 0, nx - 1)
        model[rr_off, cc_off] = MAT_IDS["crack"]
    
    # 4. 导出 HDF5
    hdf5_path = os.path.join(output_dir, f"{output_filename}.h5")
    model_3d = model[np.newaxis, :, :].astype(np.int16)
    with h5py.File(hdf5_path, "w") as f:
        f.create_dataset("data", data=model_3d, dtype=np.int16)
        f.attrs["dx_dy_dz"] = (dx, dy, dx)
    
    # 5. 生成材料文件
    materials_path = os.path.join(output_dir, f"{output_filename}_materials.txt")
    
    # 裂缝材料参数 (空气或水)
    if crack_type == "air":
        crack_eps, crack_sigma = 1, 0
    else:  # water
        crack_eps, crack_sigma = 81, 0.00001
    
    materials_params = [
        ("free_space", 1, 0, 1, 0),
        ("dry_soil", 6, 0.01, 1, 0),
        ("saturated_soil", 25, 0.1, 1, 0),
        ("rock", 5, 0.001, 1, 0),
        ("crack", crack_eps, crack_sigma, 1, 0),
    ]
    
    with open(materials_path, "w") as f:
        for name, eps_r, sigma, mu_r, sigma_star in materials_params:
            f.write(f"#material: {eps_r} {sigma} {mu_r} {sigma_star} {name}\n")
    
    # 6. 生成预览图
    png_path = os.path.join(output_dir, f"{output_filename}.png")
    colors = {
        0: [255, 255, 255],  # 空气
        1: [255, 215, 0],    # 干土
        2: [255, 215, 0],    # 饱水土
        3: [139, 0, 0],      # 岩石
        4: [0, 0, 0],        # 裂缝 (黑色)
    }
    rgb = np.zeros((ny, nx, 3), dtype=np.uint8)
    for id_val, color in colors.items():
        rgb[model == id_val] = color
    
    plt.figure(figsize=(10, 4))
    plt.imshow(rgb, origin="lower", aspect="auto", extent=[0, model_x, 0, model_y])
    plt.axhline(y=model_y - water_table_depth, color="cyan", linestyle="--", label="Water Table")
    plt.plot([crack_start_x, crack_end_x], [crack_start_y, crack_end_y], 'r-', linewidth=3, label='Crack')
    plt.title("GPR Crack Model (Cross-section)")
    plt.xlabel("X (m)")
    plt.ylabel("Depth Y (m)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()
    
    # 7. 生成 gprMax .in 文件
    hdf5_basename = os.path.basename(hdf5_path)
    materials_basename = os.path.basename(materials_path)
    
    center_freq = 200e6
    time_window = 60e-9
    
    # 计算扫描步长
    available_width = model_x - 0.35
    src_steps = available_width / (n_traces - 1) if n_traces > 1 else 0.1
    src_steps = max(0.02, min(0.1, src_steps))
    
    in_template = f"""#title: Crack model for GPR simulation
#domain: {model_x} {model_y} {dx}
#dx_dy_dz: {dx} {dy} {dx}
#time_window: {time_window}

#material: 6 0.01 1 0 dry_soil
#material: 25 0.1 1 0 saturated_soil
#material: 5 0.001 1 0 rock
#material: {crack_eps} {crack_sigma} 1 0 crack

#geometry_objects_read: 0 0 0 {hdf5_basename} {materials_basename}

#waveform: ricker 1 {center_freq} my_ricker
#hertzian_dipole: z 0.05 {model_y - 0.05} 0 my_ricker
#rx: 0.05 {model_y - 0.05} 0
#src_steps: {src_steps} 0 0
#rx_steps: {src_steps} 0 0
"""
    
    in_path = os.path.join(output_dir, f"{output_filename}.in")
    with open(in_path, "w", encoding='utf-8') as f:
        f.write(in_template)
    
    print(f"Crack model generated:")
    print(f"  Crack: ({crack_start_x}, {crack_start_y}) -> ({crack_end_x}, {crack_end_y})")
    print(f"  Width: {crack_width}m")
    print(f"  Type: {crack_type}")
    print(f"  Output: {output_dir}")
    
    return {
        "hdf5_path": hdf5_path,
        "png_path": png_path,
        "in_template_path": in_path,
        "output_dir": output_dir,
    }


if __name__ == "__main__":
    # 生成倾斜裂缝模型
    res = generate_crack_model(
        output_dir=r"D:\ClawX-Data\sim\gprmax_outcsv",
        output_filename="crack_model",
        model_x=5.0,
        model_y=2.0,
        dx=0.005,
        crack_start_x=1.0,   # 左上
        crack_start_y=0.5,
        crack_end_x=4.0,     # 右下
        crack_end_y=1.5,
        crack_width=0.02,    # 2cm宽裂缝
        crack_type="air",    # 空气填充
        n_traces=50,
        random_seed=42,
    )
    print(f"\nFiles saved to: {res['output_dir']}")
