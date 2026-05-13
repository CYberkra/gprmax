#!/usr/bin/env python3
"""重新合并 gprMax 输出文件为正确的 B-scan 矩阵"""

import os
import sys
import numpy as np

try:
    import h5py
except ImportError:
    print("需要安装 h5py: pip install h5py")
    sys.exit(1)


def merge_gprmax_out(folder_path):
    """合并目录下所有单独的 .out 文件"""

    # 获取所有单独的 .out 文件（排除 merged）
    out_files = [
        f for f in os.listdir(folder_path) if f.endswith(".out") and "merged" not in f
    ]

    # 按文件名中的数字排序
    out_files = sorted(
        out_files, key=lambda x: int("".join(filter(str.isdigit, x)) or 0)
    )

    print(f"找到 {len(out_files)} 个单独的 .out 文件")

    if not out_files:
        print("没有找到单独的 .out 文件")
        return

    # 读取第一个文件获取参数
    first_path = os.path.join(folder_path, out_files[0])
    with h5py.File(first_path, "r") as f:
        attrs = dict(f.attrs)
        iterations = attrs.get("Iterations", 0)
        dt = attrs.get("dt", 0)
        data0 = f["rxs"]["rx1"]["Ez"][:]

    samples = len(data0)
    n_traces = len(out_files)

    print(
        f"参数: iterations={iterations}, dt={dt}, samples={samples}, traces={n_traces}"
    )

    # 创建合并矩阵
    matrix = np.zeros((samples, n_traces), dtype=np.float32)
    matrix[:, 0] = data0

    # 读取其他文件
    for i, fname in enumerate(out_files[1:], 1):
        fpath = os.path.join(folder_path, fname)
        with h5py.File(fpath, "r") as f:
            matrix[:, i] = f["rxs"]["rx1"]["Ez"][:]

    # 保存合并后的文件
    merged_path = os.path.join(folder_path, "gpr_model_merged_fixed.out")
    with h5py.File(merged_path, "w") as f:
        f.create_dataset("rxs/rx1/Ez", data=matrix)
        f.attrs["Iterations"] = iterations
        f.attrs["dt"] = dt
        f.attrs["nx_ny_nz"] = attrs.get("nx_ny_nz", [1, 1, 1])
        f.attrs["Title"] = "Merged B-scan"

    print(f"已保存合并后的文件: {merged_path}")
    print(f"数据形状: {matrix.shape} (samples x traces)")

    # 生成 B-scan 图像
    try:
        import matplotlib.pyplot as plt

        # 不做增益处理，保持原始数据
        vmax = max(1e-12, np.max(np.abs(matrix)) * 0.30)

        fig, ax = plt.subplots(figsize=(12, 7))
        im = ax.imshow(
            matrix,
            cmap="seismic",
            aspect="auto",
            vmin=-vmax,
            vmax=vmax,
            origin="upper",
            extent=[0, n_traces, samples * dt * 1e9, 0],
        )
        ax.set_title("GPR B-scan (Merged)")
        ax.set_xlabel("Trace Number")
        ax.set_ylabel("Time (ns)")
        plt.colorbar(im, ax=ax, label="Amplitude")
        plt.tight_layout()

        bscan_path = os.path.join(folder_path, "gpr_model_merged_fixed_bscan.png")
        plt.savefig(bscan_path, dpi=150)
        plt.close(fig)
        print(f"B-scan 图像已保存: {bscan_path}")

    except ImportError:
        print("matplotlib 未安装，跳过图像生成")

    return merged_path


if __name__ == "__main__":
    folder = r"D:\ClawX-Data\sim\gprmax_outcsv\gpr_model_20260403_020147"
    merge_gprmax_out(folder)
