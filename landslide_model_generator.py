#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
复杂滑坡体结构 gprMax 模型生成器 (SFCW 20-170MHz 优化版)
"""

import os
import numpy as np
from scipy.spatial import ConvexHull
from skimage.draw import polygon
import h5py
import matplotlib
from datetime import datetime

matplotlib.use("Agg")  # 适合后端运行的非交互模式
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional


def generate_landslide_model(
    output_dir: str = r"D:\ClawX-Data\sim\gprmax_outcsv",
    output_filename: str = "landslide_model",
    model_x: float = 10.0,
    model_y: float = 4.0,
    # 针对低频雷达大幅增加步长，计算速度提升百倍以上
    dx: float = 0.02,
    dy: float = 0.02,
    water_table_depth: float = 1.5,
    surface_amplitude: float = 0.03,
    surface_frequency: float = 1.5,
    dry_rock_ratio: float = 0.05,
    saturated_rock_ratio: float = 0.15,
    # 适配低频波长，适当增大岩石半径下限，避免细碎石头被直接穿透且生成报错
    rock_min_radius: float = 0.1,
    rock_max_radius: float = 0.4,
    void_center_x: float = 5.0,
    void_center_y: float = 2.5,
    void_radius_x: float = 0.3,
    void_radius_y: float = 0.2,
    random_seed: Optional[int] = None,
    use_timestamp_folder: bool = True,
    n_traces: int = 10,  # Number of traces for GPR scan
) -> Dict[str, Any]:

    # 创建带时间戳的输出文件夹
    if use_timestamp_folder:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"model_{timestamp}"
        output_dir = os.path.join(output_dir, folder_name)

    os.makedirs(output_dir, exist_ok=True)

    if random_seed is not None:
        np.random.seed(random_seed)

    nx = int(model_x / dx)
    ny = int(model_y / dy)

    # 材料 ID 严格从 0 开始连续映射
    MAT_IDS = {
        "free_space": 0,
        "dry_soil": 1,
        "saturated_soil": 2,
        "rock": 3,
        "water_void": 4,
    }

    # 初始化模型矩阵 (全0，即 free_space)
    model = np.zeros((ny, nx), dtype=np.int16)

    # 1. 生成地表起伏 (Y轴0在最底部，ny在顶部)
    x_indices = np.arange(nx)
    surface_y_grid = int(ny * 0.85)
    surface_line = surface_y_grid + (surface_amplitude / dy) * np.sin(
        2 * np.pi * surface_frequency * x_indices / nx
    )
    surface_line += np.random.normal(0, 0.01 / dy, nx)
    surface_line = np.clip(surface_line, int(ny * 0.7), ny - 1).astype(int)

    # 2. 划定水分层级并填充土壤基质
    water_table_y_grid = int(ny - (water_table_depth / dy))
    water_table_y_grid = max(0, min(ny - 1, water_table_y_grid))

    for ix in range(nx):
        surface_idx = surface_line[ix]
        dry_soil_end = min(surface_idx, water_table_y_grid)
        # 填充干土 (潜水面到地表)
        if surface_idx > dry_soil_end:
            model[dry_soil_end:surface_idx, ix] = MAT_IDS["dry_soil"]
        # 填充饱水土 (底部到潜水面)
        if dry_soil_end > 0:
            model[0:dry_soil_end, ix] = MAT_IDS["saturated_soil"]

    # 3. 核心工具：带防卡死机制的 ConvexHull 岩石生成器
    def populate_rocks(model_matrix, y_min, y_max, target_ratio, rock_id, layer_name):
        layer_area = nx * (y_max - y_min)
        target_area = int(layer_area * target_ratio)
        current_area = 0

        # GUI 防卡死：硬性重试上限
        max_attempts = int((target_area / (np.pi * (rock_min_radius / dx) ** 2)) * 10)
        attempts = 0

        rock_min_r_grid = rock_min_radius / dx
        rock_max_r_grid = rock_max_radius / dx

        print(f"[{layer_name}] 开始生成岩石. 目标面积占比: {target_ratio * 100}%")

        while current_area < target_area and attempts < max_attempts:
            attempts += 1

            # 留出边界余量避免越界
            margin = int(rock_max_r_grid) + 1
            if y_max - margin <= y_min + margin:
                break  # 层太薄，无法生成

            cx = np.random.randint(margin, nx - margin)
            cy = np.random.randint(y_min + margin, y_max - margin)
            radius = np.random.uniform(rock_min_r_grid, rock_max_r_grid)

            n_points = np.random.randint(6, 12)
            angles = np.sort(np.random.uniform(0, 2 * np.pi, n_points))
            radii = radius * np.random.uniform(0.6, 1.0, n_points)
            points = np.column_stack(
                [cx + radii * np.cos(angles), cy + radii * np.sin(angles)]
            )

            try:
                hull = ConvexHull(points)
                hull_points = points[hull.vertices]
                rr = np.round(hull_points[:, 1]).astype(int)
                cc = np.round(hull_points[:, 0]).astype(int)

                polygon_y, polygon_x = polygon(rr, cc, shape=(ny, nx))

                # 碰撞检测：只在纯净土层生成石头，不覆盖现有石头或越界到空气中
                existing_materials = model_matrix[polygon_y, polygon_x]
                if np.any(existing_materials == MAT_IDS["rock"]) or np.any(
                    existing_materials == MAT_IDS["free_space"]
                ):
                    continue

                model_matrix[polygon_y, polygon_x] = rock_id
                current_area += len(polygon_x)
            except Exception:
                continue

        if attempts >= max_attempts:
            print(
                f"[{layer_name}] 警告: 空间不足，已达到最大尝试次数。实际生成占比可能略低于设定值。"
            )

    # 填充饱和层岩石
    populate_rocks(
        model, 0, water_table_y_grid, saturated_rock_ratio, MAT_IDS["rock"], "饱水土层"
    )
    # 填充干土层岩石
    populate_rocks(
        model,
        water_table_y_grid,
        np.min(surface_line),
        dry_rock_ratio,
        MAT_IDS["rock"],
        "干土层",
    )

    # 4. 生成充水空洞
    cx, cy = int(void_center_x / dx), int(void_center_y / dy)
    rx, ry = int(void_radius_x / dx), int(void_radius_y / dy)
    y, x = np.ogrid[:ny, :nx]
    mask = ((x - cx) ** 2 / rx**2 + (y - cy) ** 2 / ry**2) <= 1
    model[mask] = MAT_IDS["water_void"]

    # 5. 导出 HDF5
    hdf5_path = os.path.join(output_dir, f"{output_filename}.h5")
    model_3d = model[np.newaxis, :, :].astype(np.int16)
    with h5py.File(hdf5_path, "w") as f:
        f.create_dataset("data", data=model_3d, dtype=np.int16)
        f.attrs["dx_dy_dz"] = (dx, dy, dx)

    # 6. 生成正确的 materials.txt
    # gprMax HDF5读取要求：必须包含完整的 #material 命令定义，按 ID 顺序排列
    materials_path = os.path.join(output_dir, f"{output_filename}_materials.txt")

    # 定义材料参数 (按 ID 顺序: 0, 1, 2, 3, 4)
    materials_params = [
        ("free_space", 1, 0, 1, 0),  # ID 0: 空气
        ("dry_soil", 6, 0.01, 1, 0),  # ID 1: 干土
        ("saturated_soil", 25, 0.1, 1, 0),  # ID 2: 饱水土
        ("rock", 5, 0.001, 1, 0),  # ID 3: 岩石
        ("water_void", 81, 0.00001, 1, 0),  # ID 4: 水空洞
    ]

    with open(materials_path, "w") as f:
        for name, eps_r, sigma, mu_r, sigma_star in materials_params:
            f.write(f"#material: {eps_r} {sigma} {mu_r} {sigma_star} {name}\n")

    # 7. 生成并保存预览图 (修正坐标系)
    png_path = os.path.join(output_dir, f"{output_filename}.png")
    colors = {
        0: [255, 255, 255],  # 空气 (白)
        1: [255, 215, 0],  # 干土 (亮黄色)
        2: [255, 215, 0],  # 饱水土 (亮黄色，与干土统一背景色)
        3: [139, 0, 0],  # 岩石 (深红色)
        4: [0, 0, 139],  # 充水空洞 (深蓝色)
    }
    rgb = np.zeros((ny, nx, 3), dtype=np.uint8)
    for id_val, color in colors.items():
        rgb[model == id_val] = color

    plt.figure(figsize=(10, 4))
    # 注意：origin='lower' 确保物理世界的 Y=0 (底部) 显示在图像下方
    plt.imshow(rgb, origin="lower", aspect="auto", extent=[0, model_x, 0, model_y])
    plt.axhline(
        y=model_y - water_table_depth,
        color="cyan",
        linestyle="--",
        linewidth=1.5,
        label="Water Table",
    )
    plt.title("GPR Landslide Subsurface Model (Cross-section)")
    plt.xlabel("X (m)")
    plt.ylabel("Depth/Elevation Y (m)")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()

    # 8. Generate gprMax config file (optimized parameters)
    hdf5_basename = os.path.basename(hdf5_path)
    materials_basename = os.path.basename(materials_path)

    # Calculate appropriate parameters based on model size
    center_freq = 200e6  # 200MHz better for small model depth detection
    time_window = 60e-9  # Increase to 60ns to ensure deep reflections are visible
    
    # Calculate appropriate src_steps to fit all traces within model
    # Model width = model_x, start at 0.25m, need n_traces within model
    # Available space = model_x - 0.25 - 0.1 (margin) = model_x - 0.35
    available_width = model_x - 0.35
    src_steps = available_width / (n_traces - 1) if n_traces > 1 else 0.1
    src_steps = max(0.02, min(0.1, src_steps))  # Clamp between 0.02 and 0.1
    
    print(f"  Calculated src_steps: {src_steps:.3f}m for {n_traces} traces")

    in_template = f"""#title: Landslide model for GPR simulation
#domain: {model_x} {model_y} {dx}
#dx_dy_dz: {dx} {dy} {dx}
#time_window: {time_window}

#material: 6 0.01 1 0 dry_soil
#material: 25 0.1 1 0 saturated_soil
#material: 5 0.001 1 0 rock
#material: 81 0.00001 1 0 water_void

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

    print(f"  gprMax config parameters:")
    print(f"    Center frequency: {center_freq/1e6:.0f} MHz")
    print(f"    Time window: {time_window*1e9:.0f} ns")
    print(f"    Spatial resolution: {dx*1000:.1f} mm")
    print(f"    Source steps: {src_steps:.3f} m")
    print(f"    Number of traces: {n_traces}")

    return {
        "hdf5_path": hdf5_path,
        "png_path": png_path,
        "in_template_path": in_path,
        "model_x": model_x,
        "model_y": model_y,
    }


def plot_gprmax_bscan(merged_out_path: str, output_png_path: str) -> str:
    """绘制 gprMax B-scan 灰度图（读取所有 .out 文件）

    Parameters
    ----------
    merged_out_path : str
        .out HDF5 文件路径
    output_png_path : str
        要保存的 .png 图片路径

    Returns
    -------
    str
        保存的图片路径
    """
    import h5py
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    matplotlib.use("Agg")

    # 获取输出文件夹
    folder = os.path.dirname(merged_out_path)
    filename = os.path.basename(merged_out_path)

    # 查找所有 .out 文件（排除 merged 文件）
    out_files = sorted(
        [f for f in os.listdir(folder) if f.endswith(".out") and "merged" not in f]
    )

    if not out_files:
        raise FileNotFoundError(f"找不到 .out 文件: {folder}")

    print(f"  找到 {len(out_files)} 个 .out 文件")

    # 读取第一个文件获取采样点数
    first_path = os.path.join(folder, out_files[0])
    with h5py.File(first_path, "r") as f:
        samples = f.attrs.get("Iterations", 3393)
        dt = f.attrs.get("dt", 1.0)  # 时间步长

    # 读取所有道
    traces = len(out_files)
    data = np.zeros((samples, traces), dtype=np.float32)

    for i, fname in enumerate(out_files):
        fpath = os.path.join(folder, fname)
        with h5py.File(fpath, "r") as f:
            data[:, i] = f["rxs"]["rx1"]["Ez"][:]

    print(f"  合并数据形状: {data.shape} ({traces} 道 x {samples} 采样点)")
    print(f"  数据范围: [{np.min(data):.6f}, {np.max(data):.6f}]")

    # 关键修复: 数据已经是 (samples, traces)，不需要转置
    # 显示时: x轴=道数, y轴=时间/深度
    
    # 应用增益补偿（深度越大信号越弱）
    time_axis = np.arange(samples) * dt
    gain = np.exp(0.5 * time_axis / np.max(time_axis))  # 指数增益
    gain = gain[:, np.newaxis]  # 转为列向量
    data_gain = data * gain

    # 计算显示范围（使用增益后的数据）
    vmax = np.max(np.abs(data_gain)) * 0.3  # 降低阈值以显示更多细节
    
    print(f"  显示范围: +/-{vmax:.6f}")

    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 显示 B-scan (注意: origin='upper' 让时间0在顶部，符合GPR惯例)
    im = ax.imshow(
        data_gain, 
        cmap="seismic",  # 使用 seismic 配色更好地显示正负值
        aspect="auto", 
        vmin=-vmax, 
        vmax=vmax,
        origin="upper",  # 时间0在上方（地表）
        extent=[0, traces, samples * dt * 1e9, 0]  # x: 道数, y: 时间(ns)
    )
    
    ax.set_xlabel("Trace Number", fontsize=12)
    ax.set_ylabel("Time (ns)", fontsize=12)
    ax.set_title("GPR B-scan (with exponential gain)", fontsize=14)
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, label="Amplitude")
    
    plt.tight_layout()
    plt.savefig(output_png_path, dpi=150, bbox_inches="tight")
    plt.close()

    return output_png_path


def run_full_pipeline(
    output_dir: str = r"D:\ClawX-Data\sim\gprmax_outcsv",
    output_filename: str = "landslide_model",
    model_x: float = 10.0,
    model_y: float = 4.0,
    dx: float = 0.005,
    dy: float = 0.005,
    water_table_depth: float = 1.5,
    surface_amplitude: float = 0.03,
    surface_frequency: float = 1.5,
    dry_rock_ratio: float = 0.05,
    saturated_rock_ratio: float = 0.15,
    rock_min_radius: float = 0.05,
    rock_max_radius: float = 0.25,
    void_center_x: float = 5.0,
    void_center_y: float = 2.5,
    void_radius_x: float = 0.3,
    void_radius_y: float = 0.2,
    random_seed: Optional[int] = None,
    gprmax_python: Optional[str] = None,
    n_traces: int = 1,
) -> Dict[str, Any]:
    """全自动 GPR 仿真流水线

    执行顺序：生成模型 -> 跑正演 -> 合并数据 -> 画出对比图

    Parameters
    ----------
    output_dir : str
        输出目录
    output_filename : str
        文件名
    model_x, model_y, dx, dy : float
        模型参数
    water_table_depth : float
        潜水面深度
    surface_amplitude, surface_frequency : float
        地表参数
    dry_rock_ratio, saturated_rock_ratio : float
        含石率
    rock_min_radius, rock_max_radius : float
        岩石半径范围
    void_center_x, void_center_y : float
        空洞中心
    void_radius_x, void_radius_y : float
        空洞半径
    random_seed : int, optional
        随机种子
    gprmax_python : str, optional
        gprMax Python 解释器路径
    n_traces : int
        B-scan 扫描线数量

    Returns
    -------
    dict
        包含所有输出文件路径的字典
    """
    import sys
    import subprocess

    # 确定 Python 解释器路径
    if gprmax_python is None:
        gprmax_python = sys.executable

    print("=" * 60)
    print("GPR 全自动仿真流水线")
    print("=" * 60)

    # =========================================================================
    # Step A: 生成模型
    # =========================================================================
    print("\n[Step A] 正在生成滑坡体模型...")
    res = generate_landslide_model(
        output_dir=output_dir,
        output_filename=output_filename,
        model_x=model_x,
        model_y=model_y,
        dx=dx,
        dy=dy,
        water_table_depth=water_table_depth,
        surface_amplitude=surface_amplitude,
        surface_frequency=surface_frequency,
        dry_rock_ratio=dry_rock_ratio,
        saturated_rock_ratio=saturated_rock_ratio,
        rock_min_radius=rock_min_radius,
        rock_max_radius=rock_max_radius,
        void_center_x=void_center_x,
        void_center_y=void_center_y,
        void_radius_x=void_radius_x,
        void_radius_y=void_radius_y,
        random_seed=random_seed,
        use_timestamp_folder=True,
        n_traces=n_traces,  # Pass n_traces to model generator
    )

    in_path = res["in_template_path"]
    output_folder = os.path.dirname(in_path)
    model_png_path = res["png_path"]

    print(f"  模型生成完成: {model_png_path}")

    # =========================================================================
    # Step B: 运行 gprMax 正演
    # =========================================================================
    print("\n[Step B] 正在运行 gprMax 正演...")
    print("  警告: 这可能需要几分钟时间，请耐心等待...")

    try:
        cmd_gprmax = [
            gprmax_python,
            "-m",
            "gprMax",
            in_path,
            "-n",
            str(n_traces),
        ]
        result_gprmax = subprocess.run(
            cmd_gprmax,
            capture_output=True,
            text=True,
            timeout=1800,  # 30分钟超时（50道需要更长时间）
        )

        if result_gprmax.returncode != 0:
            print(f"  错误: gprMax 运行失败!")
            print(f"  STDERR: {result_gprmax.stderr}")
            raise RuntimeError(f"gprMax 运行失败: {result_gprmax.stderr}")

        print("  gprMax 正演完成!")

    except subprocess.TimeoutExpired:
        print("  错误: gprMax 运行超时 (超过10分钟)")
        raise RuntimeError("gprMax 运行超时")
    except Exception as e:
        print(f"  错误: {e}")
        raise

    # =========================================================================
    # Step C: 合并输出数据
    # =========================================================================
    print("\n[Step C] 正在合并输出数据...")

    try:
        # 使用 outputfiles_merge 模块
        cmd_merge = [
            gprmax_python,
            "-m",
            "tools.outputfiles_merge",
            in_path,
        ]
        result_merge = subprocess.run(
            cmd_merge,
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result_merge.returncode != 0:
            print(f"  警告: 数据合并返回非零状态码: {result_merge.returncode}")
            print(f"  STDERR: {result_merge.stderr}")
        else:
            print("  数据合并完成!")

    except subprocess.TimeoutExpired:
        print("  警告: 数据合并超时")
    except Exception as e:
        print(f"  警告: 数据合并失败: {e}")

    # =========================================================================
    # Step D: 绘制 B-scan 灰度图
    # =========================================================================
    print("\n[Step D] 正在绘制 B-scan 灰度图...")

    # 构建 merged 文件路径
    base_name = output_filename.replace(".in", "")
    merged_out_path = os.path.join(output_folder, f"{base_name}_merged.out")
    bscan_png_path = os.path.join(output_folder, f"{base_name}_bscan.png")

    if not os.path.exists(merged_out_path):
        # 优先使用单个 .out 文件（合并文件可能为空）
        out_files = [
            f
            for f in os.listdir(output_folder)
            if f.endswith(".out") and "merged" not in f
        ]
        if out_files:
            merged_out_path = os.path.join(output_folder, out_files[0])
            print(f"  使用单道文件: {out_files[0]}")
        else:
            # 尝试查找合并文件
            possible_files = [
                f
                for f in os.listdir(output_folder)
                if "merged" in f and f.endswith(".out")
            ]
            if possible_files:
                merged_out_path = os.path.join(output_folder, possible_files[0])
                print(f"  找到合并文件: {possible_files[0]}")
            else:
                raise FileNotFoundError("找不到任何 .out 文件")

    try:
        plot_gprmax_bscan(merged_out_path, bscan_png_path)
        print(f"  B-scan 灰度图已保存: {bscan_png_path}")
    except Exception as e:
        print(f"  错误: B-scan 绘图失败: {e}")
        raise

    # =========================================================================
    # Step E: Save CSV format (GUI compatible format)
    # =========================================================================
    print("\n[Step E] Saving CSV format (GUI compatible)...")

    try:
        import h5py
        import numpy as np

        # Read all .out files
        out_files = sorted(
            [
                f
                for f in os.listdir(output_folder)
                if f.endswith(".out") and "merged" not in f
            ]
        )

        if out_files:
            # Read first file to get parameters
            first_path = os.path.join(output_folder, out_files[0])
            with h5py.File(first_path, "r") as f:
                samples = f.attrs.get("Iterations", 3393)
                dt = f.attrs.get("dt", 1.926e-12)  # Time step
                time_window_ns = samples * dt * 1e9  # Convert to ns

            # Read all traces
            traces = len(out_files)
            matrix = np.zeros((samples, traces), dtype=np.float32)

            for i, fname in enumerate(out_files):
                fpath = os.path.join(output_folder, fname)
                with h5py.File(fpath, "r") as f:
                    matrix[:, i] = f["rxs"]["rx1"]["Ez"][:]

            # Calculate trace interval (from .in file src_steps)
            trace_interval = 0.1  # Default 0.1m

            # Save as GUI compatible CSV format
            csv_path = os.path.join(output_folder, f"{output_filename}.csv")
            
            # GUI expects 5 columns: col1,col2,col3,<GPR_data>,col5
            # We write one trace per row group
            # For multi-trace data, we concatenate each as 5 columns
            with open(csv_path, "w") as f:
                # Write metadata header (4 lines)
                f.write(f"Number of Samples = {samples},,,,\n")
                f.write(f"Time windows (ns) = {time_window_ns:.1f},,,,\n")
                f.write(f"Number of Traces = {traces},,,,\n")
                f.write(f"Trace interval (m) = {trace_interval},,,,\n")
                
                # Write data
                for i in range(samples):
                    values = []
                    for j in range(traces):
                        val = matrix[i, j]
                        # Write 5 columns per trace: 0,0,0,GPR_data,0
                        values.extend(["0.0", "0.0", "0.0", f"{val:.7f}", "0.0"])
                    f.write(",".join(values) + "\n")
            
            print(f"  CSV saved: {csv_path}")
            print(f"    Columns: {traces * 5}")
            print(f"    Samples: {samples}")
            print(f"    Traces: {traces}")
            print(f"    Time window: {time_window_ns:.1f} ns")
            
            # Also save pure matrix version (for other uses)
            csv_pure_path = os.path.join(output_folder, f"{output_filename}_matrix.csv")
            np.savetxt(csv_pure_path, matrix, delimiter=",", fmt="%.6e")
            print(f"  Matrix CSV: {csv_pure_path}")
        else:
            print("  Warning: No .out files found")

    except Exception as e:
        print(f"  Error: CSV save failed: {e}")
        import traceback
        traceback.print_exc()

    # =========================================================================
    # 完成
    # =========================================================================
    print("\n" + "=" * 60)
    print("流水线执行完成!")
    print("=" * 60)
    print(f"模型预览图: {model_png_path}")
    print(f"B-scan 灰度图: {bscan_png_path}")
    print("=" * 60)

    return {
        "output_folder": output_folder,
        "model_png_path": model_png_path,
        "bscan_png_path": bscan_png_path,
        "hdf5_path": res["hdf5_path"],
        "in_path": in_path,
        "merged_out_path": merged_out_path,
    }


if __name__ == "__main__":
    res = run_full_pipeline(
        output_dir=r"D:\ClawX-Data\sim\gprmax_outcsv",
        output_filename="landslide_model_v2",
        model_x=5.0,  # 5m x 2m model
        model_y=2.0,
        dx=0.005,  # 5mm resolution for better detail
        dy=0.005,
        water_table_depth=0.8,
        dry_rock_ratio=0.03,
        saturated_rock_ratio=0.10,
        void_center_x=2.5,  # Void at center X=2.5m
        void_center_y=1.2,  # Void at depth Y=1.2m
        void_radius_x=0.20,  # Larger void (40cm wide)
        void_radius_y=0.15,
        n_traces=50,  # 50 traces covering 0.5-4.5m (covers void at 2.5m)
        random_seed=42,
    )
