#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将所有 CSV 合并成一个 B-scan 矩阵 CSV
"""

import os
import numpy as np
from pathlib import Path

def merge_to_bscan(csv_dir, output_file=None):
    """将所有 CSV 合并成 B-scan 矩阵"""
    csv_files = sorted(Path(csv_dir).glob("*.csv"))
    
    if not csv_files:
        print(f"未找到 CSV 文件: {csv_dir}")
        return
    
    print(f"找到 {len(csv_files)} 个 CSV 文件")
    print("合并为 B-scan 矩阵...")
    
    # 读取所有道数据
    traces = []
    for csv_file in csv_files:
        data = np.loadtxt(csv_file, delimiter=',')
        traces.append(data)
    
    # 合并成矩阵 (samples × traces)
    bscan = np.column_stack(traces)
    
    if output_file is None:
        output_file = Path(csv_dir) / "bent_crack_bscan_matrix.csv"
    
    # 保存为 CSV
    np.savetxt(output_file, bscan, delimiter=',', fmt='%.6e')
    
    print(f"B-scan 矩阵: {bscan.shape}")
    print(f"保存到: {output_file}")
    print(f"  行数 (时间采样): {bscan.shape[0]}")
    print(f"  列数 (扫描道数): {bscan.shape[1]}")
    
    return bscan

if __name__ == "__main__":
    csv_dir = r"D:\ClawX-Data\sim\gprmax_outcsv\bent_crack_20260401_135308"
    merge_to_bscan(csv_dir)
