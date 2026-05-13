#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复版滑坡/空洞模型生成器
- 模型高度改为 2.5m（适合 GPR 探测）
- 空洞位置调整到 0.8m 深度
- 增大空洞尺寸以增强反射
"""

import os
import sys
import numpy as np
from datetime import datetime
from pathlib import Path

# 添加 gprMax 路径
sys.path.insert(0, r'E:\gprMax\gprMax-v.3.1.7')

def generate_fixed_landslide_model():
    """生成修复版的滑坡/空洞模型"""
    
    # 修复后的参数
    model_x = 5.0          # 模型宽度 5m
    model_y = 2.5          # 模型高度 2.5m（修复：从15m减小）
    dx = dy = 0.005        # 网格 0.5cm
    
    # 计算网格数
    nx = int(model_x / dx)
    ny = int(model_y / dy)
    
    print(f"模型尺寸: {model_x}m x {model_y}m")
    print(f"网格: {nx} x {ny}")
    print(f"总网格数: {nx * ny:,}")
    
    # 材料定义
    # 0: free_space
    # 1: dry_soil (εr=6)
    # 2: saturated_soil (εr=25)
    # 3: rock (εr=5)
    # 4: water_void (εr=81)
    
    # 创建模型数组
    model = np.ones((ny, nx), dtype=np.uint8)  # 默认干土
    
    # 潜水面深度 0.8m
    water_table_y = 0.8
    water_table_ny = int(water_table_y / dy)
    
    # 设置潜水面以下区域为水饱和土
    model[water_table_ny:, :] = 2  # saturated_soil
    
    # 添加地表起伏（正弦波）
    surface_amplitude = 0.03  # 3cm 起伏
    surface_freq = 1.5
    
    x_coords = np.linspace(0, model_x, nx)
    surface_y = surface_amplitude * np.sin(2 * np.pi * surface_freq * x_coords / model_x)
    
    for i, x in enumerate(x_coords):
        surface_ny = int((surface_y[i]) / dy)
        if surface_ny > 0:
            model[:surface_ny, i] = 0  # free_space (空气)
    
    # 添加空洞（修复：位置调整到 0.8m 深度，增大尺寸）
    void_center_x = 2.5    # 中心 X
    void_center_y = 0.8    # 中心 Y（修复：从1.2m调整到0.8m，在潜水面附近）
    void_radius_x = 0.3    # X方向半径 30cm（修复：从0.15m增大）
    void_radius_y = 0.25   # Y方向半径 25cm（修复：从0.15m增大）
    
    void_center_nx = int(void_center_x / dx)
    void_center_ny = int(void_center_y / dy)
    void_radius_nx = int(void_radius_x / dx)
    void_radius_ny = int(void_radius_y / dy)
    
    # 绘制椭圆空洞
    for i in range(ny):
        for j in range(nx):
            if ((j - void_center_nx) / void_radius_nx) ** 2 + \
               ((i - void_center_ny) / void_radius_ny) ** 2 <= 1:
                model[i, j] = 4  # water_void
    
    # 添加一些随机岩石
    np.random.seed(42)
    n_rocks = 50
    for _ in range(n_rocks):
        rx = np.random.randint(0, nx)
        ry = np.random.randint(water_table_ny, ny)  # 只在饱和层以下
        rr = np.random.randint(3, 8)  # 岩石半径 1.5-4cm
        
        for i in range(max(0, ry-rr), min(ny, ry+rr)):
            for j in range(max(0, rx-rr), min(nx, rx+rr)):
                if (i-ry)**2 + (j-rx)**2 <= rr**2:
                    model[i, j] = 3  # rock
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(r'D:\ClawX-Data\sim\gprmax_outcsv') / f'landslide_void_fixed_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存 HDF5 文件
    import h5py
    h5_path = output_dir / 'gpr_model.h5'
    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('data', data=model)
    
    # 保存材料文件
    materials_path = output_dir / 'gpr_model_materials.txt'
    with open(materials_path, 'w') as f:
        f.write('#materials: dry_soil saturated_soil rock water_void air_crack water_crack\n')
        for i in range(ny):
            for j in range(nx):
                f.write(f'{model[i, j]}\n')
    
    # 生成 gprMax 输入文件
    in_path = output_dir / 'gpr_model.in'
    
    # 修复后的参数
    center_freq_mhz = 400
    center_freq_hz = center_freq_mhz * 1e6
    time_window_ns = 60  # 修复：增加到 60ns，确保能探测到 2.5m 深度
    time_window_s = time_window_ns * 1e-9
    
    # 计算源步长（500道覆盖5m）
    n_traces = 500
    src_steps = model_x / n_traces
    
    input_lines = [
        f'#title: GPR Landslide/Void Model (Fixed)',
        f'#domain: {model_x} {model_y} {dx}',
        f'#dx_dy_dz: {dx} {dy} {dx}',
        f'#time_window: {time_window_s}',
        '',
        '#material: 6.0 0.01 1.0 0.0 dry_soil',
        '#material: 25.0 0.1 1.0 0.0 saturated_soil',
        '#material: 5.0 0.001 1.0 0.0 rock',
        '#material: 81.0 1e-05 1.0 0.0 water_void',
        '#material: 1.0 0.0 1.0 0.0 air_crack',
        '#material: 81.0 1e-05 1.0 0.0 water_crack',
        '',
        f'#geometry_objects_read: 0 0 0 gpr_model.h5 gpr_model_materials.txt',
        '',
        f'#waveform: ricker 1 {center_freq_hz} wave1',
        f'#hertzian_dipole: z 0.05 {model_y - 0.05} 0 wave1',
        f'#rx: 0.05 {model_y - 0.05} 0',
        f'#src_steps: {src_steps} 0 0',
        f'#rx_steps: {src_steps} 0 0',
    ]
    
    with open(in_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(input_lines) + '\n')
    
    # 保存预览图
    preview_path = output_dir / 'gpr_model.png'
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    colors = {
        0: (248, 248, 248),  # free_space - white
        1: (194, 178, 128),  # dry_soil - tan
        2: (139, 90, 43),    # saturated_soil - brown
        3: (128, 128, 128),  # rock - gray
        4: (65, 105, 225),   # water_void - blue
    }
    
    rgb = np.zeros((ny, nx, 3), dtype=np.uint8)
    for mid, color in colors.items():
        rgb[model == mid] = color
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(rgb, extent=[0, model_x, model_y, 0])
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'Landslide/Void Model (Fixed)\nModel: {model_x}m x {model_y}m, Void at Y={void_center_y}m')
    
    # 添加潜水面线
    ax.axhline(y=water_table_y, color='cyan', linestyle='--', linewidth=2, label='Water Table')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(preview_path, dpi=150)
    plt.close()
    
    print(f"\n✅ 模型生成完成！")
    print(f"输出目录: {output_dir}")
    print(f"模型文件: {h5_path.name}")
    print(f"输入文件: {in_path.name}")
    print(f"预览图: {preview_path.name}")
    print(f"\n修复内容:")
    print(f"  - 模型高度: 15m → {model_y}m")
    print(f"  - 空洞深度: 1.2m → {void_center_y}m (潜水面附近)")
    print(f"  - 空洞尺寸: 0.15m → {void_radius_x}m x {void_radius_y}m")
    print(f"  - 时间窗: 40ns → {time_window_ns}ns")
    print(f"\n运行 gprMax:")
    print(f"  cd {output_dir}")
    print(f"  python -m gprMax gpr_model.in -n {n_traces}")
    
    return output_dir

if __name__ == '__main__':
    output_dir = generate_fixed_landslide_model()
    print(f"\n{'='*50}")
    print("现在可以运行 gprMax 了！")
