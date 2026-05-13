#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 gprMax .out 文件转换为 CSV 格式
"""

import os
import sys
import h5py
import numpy as np
from pathlib import Path

def out_to_csv(out_file, csv_file=None):
    """将单个 .out 文件转换为 CSV"""
    if csv_file is None:
        csv_file = out_file.replace('.out', '.csv')
    
    # 读取 .out 文件 (HDF5 格式)
    with h5py.File(out_file, 'r') as f:
        # gprMax 数据通常在 /rxs/rx1/ 路径下
        if 'rxs' in f and 'rx1' in f['rxs']:
            data = f['rxs/rx1/Ez'][()]  # Ez 分量
        elif 'Ez' in f:
            data = f['Ez'][()]
        else:
            # 尝试找到数据
            keys = list(f.keys())
            print(f"  可用数据集: {keys}")
            for key in keys:
                if isinstance(f[key], h5py.Dataset):
                    data = f[key][()]
                    break
    
    # 保存为 CSV
    if len(data.shape) == 1:
        # 单道数据
        np.savetxt(csv_file, data, delimiter=',', fmt='%.6e')
    else:
        # 多道数据
        np.savetxt(csv_file, data, delimiter=',', fmt='%.6e')
    
    print(f"  转换: {os.path.basename(out_file)} -> {os.path.basename(csv_file)}")
    return csv_file

def batch_convert(out_dir):
    """批量转换目录中的所有 .out 文件"""
    out_files = sorted(Path(out_dir).glob("*.out"))
    
    if not out_files:
        print(f"未找到 .out 文件: {out_dir}")
        return
    
    print(f"找到 {len(out_files)} 个 .out 文件")
    print("开始转换...")
    
    for i, out_file in enumerate(out_files, 1):
        csv_file = out_file.with_suffix('.csv')
        try:
            out_to_csv(str(out_file), str(csv_file))
            if i % 10 == 0:
                print(f"  已完成 {i}/{len(out_files)}")
        except Exception as e:
            print(f"  错误 {out_file.name}: {e}")
    
    print(f"\n转换完成！CSV 文件保存在: {out_dir}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        out_dir = sys.argv[1]
    else:
        # 默认目录
        out_dir = r"D:\ClawX-Data\sim\gprmax_outcsv\bent_crack_20260401_135308"
    
    batch_convert(out_dir)
