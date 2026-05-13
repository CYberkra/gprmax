#!/usr/bin/env python3
"""Generate images for MATLAB-gprMax PPT."""

import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches

# Set Chinese font support
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ppt_images')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def fig_workflow():
    """Generate workflow diagram."""
    fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')

    boxes = [
        (0.5, 3.5, 2.2, 1.2, '#4F81BD', 'MATLAB\n参数化建模'),
        (3.5, 3.5, 2.2, 1.2, '#9BBB59', 'gprMax\nFDTD正演'),
        (6.5, 3.5, 2.2, 1.2, '#F79646', 'HDF5\n输出文件'),
        (9.5, 3.5, 2.2, 1.2, '#C5504B', 'MATLAB\n后处理分析'),
    ]

    for x, y, w, h, color, text in boxes:
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                             facecolor=color, edgecolor='black', linewidth=2, alpha=0.85)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=14, fontweight='bold', color='white')

    # Arrows
    arrows = [(2.7, 4.1, 3.5, 4.1), (5.7, 4.1, 6.5, 4.1), (8.7, 4.1, 9.5, 4.1)]
    for x1, y1, x2, y2 in arrows:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='black', lw=2.5))

    # Bottom description
    desc_y = 1.8
    ax.text(6, desc_y, '联合仿真闭环：MATLAB 控制参数 → gprMax 求解 Maxwell 方程 → MATLAB 可视化分析',
            ha='center', va='center', fontsize=12, style='italic', color='#333333')

    # Add input/output labels
    ax.text(2.1, 2.8, '.in 输入文件', ha='center', fontsize=10, color='#4F81BD')
    ax.text(5.1, 2.8, '--geometry-fixed', ha='center', fontsize=10, color='#9BBB59')
    ax.text(8.1, 2.8, '.out (HDF5)', ha='center', fontsize=10, color='#F79646')
    ax.text(11.1, 2.8, 'B-scan / A-scan', ha='center', fontsize=10, color='#C5504B')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_workflow.png'), dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print('Generated: fig_workflow.png')


def fig_input_file():
    """Generate input file screenshot-like image."""
    fig, ax = plt.subplots(figsize=(10, 7), facecolor='#1e1e1e')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis('off')

    code_lines = [
        ("#title: B-scan from a metal cylinder buried in a dielectric half-space", '#DCDCAA'),
        ("#domain: 0.240 0.210 0.002", '#CE9178'),
        ("#dx_dy_dz: 0.002 0.002 0.002", '#CE9178'),
        ("#time_window: 3e-9", '#CE9178'),
        ("", '#CCCCCC'),
        ("#material: 6 0 1 0 half_space", '#4EC9B0'),
        ("", '#CCCCCC'),
        ("#waveform: ricker 1 1.5e9 my_ricker", '#DCDCAA'),
        ("#hertzian_dipole: z 0.040 0.170 0 my_ricker", '#DCDCAA'),
        ("#rx: 0.080 0.170 0", '#DCDCAA'),
        ("#src_steps: 0.002 0 0", '#DCDCAA'),
        ("#rx_steps: 0.002 0 0", '#DCDCAA'),
        ("", '#CCCCCC'),
        ("#box: 0 0 0 0.240 0.170 0.002 half_space", '#4EC9B0'),
        ("#cylinder: 0.120 0.080 0 0.120 0.080 0.002 0.010 pec", '#4EC9B0'),
    ]

    y_start = 6.5
    line_height = 0.38
    for i, (line, color) in enumerate(code_lines):
        y = y_start - i * line_height
        # Line number
        ax.text(0.3, y, str(i+1), fontsize=10, color='#858585', ha='right', va='center',
                family='monospace')
        # Code
        ax.text(0.5, y, line, fontsize=11, color=color, ha='left', va='center',
                family='monospace')

    ax.text(5, 0.3, 'gprMax 输入文件 (.in) —— ASCII 文本，# 开头为命令',
            ha='center', fontsize=11, color='#CCCCCC', style='italic')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_input_file.png'), dpi=150, bbox_inches='tight',
                facecolor='#1e1e1e', edgecolor='none')
    plt.close()
    print('Generated: fig_input_file.png')


def fig_Ascan():
    """Generate A-scan plot similar to MATLAB plot_Ascan.m."""
    filename = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'user_models', 'cylinder_Ascan_2D.out')
    if not os.path.exists(filename):
        print(f'Skip fig_Ascan.png: {filename} not found')
        return

    with h5py.File(filename, 'r') as f:
        dt = f.attrs['dt']
        iterations = f.attrs['Iterations']
        time = np.linspace(0, (iterations - 1) * dt, num=iterations)

        fields = {}
        for comp in ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz']:
            fields[comp] = f['/rxs/rx1/' + comp][:]

    fig, axes = plt.subplots(3, 2, figsize=(14, 10), facecolor='white')
    comps = [('Ex', 'V/m', 'r'), ('Hx', 'A/m', 'b'),
             ('Ey', 'V/m', 'r'), ('Hy', 'A/m', 'b'),
             ('Ez', 'V/m', 'r'), ('Hz', 'A/m', 'b')]

    for ax, (comp, unit, color) in zip(axes.flat, comps):
        ax.plot(time * 1e9, fields[comp], color=color, linewidth=1.2)
        ax.set_xlabel('Time [ns]', fontsize=11)
        ax.set_ylabel(f'Field strength [{unit}]', fontsize=11)
        ax.set_title(comp, fontsize=13, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_xlim([0, time[-1] * 1e9])

    fig.suptitle('MATLAB / Python 读取 gprMax A-scan 输出 —— 6 个电磁场分量',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_Ascan.png'), dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print('Generated: fig_Ascan.png')


def fig_Bscan():
    """Generate B-scan plot similar to MATLAB plot_Bscan.m."""
    filename = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'user_models', 'cylinder_Bscan_2D_merged.out')
    if not os.path.exists(filename):
        print(f'Skip fig_Bscan.png: {filename} not found')
        return

    with h5py.File(filename, 'r') as f:
        dt = f.attrs['dt']
        iterations = f.attrs['Iterations']
        data = f['/rxs/rx1/Ez'][:]

    fig, ax = plt.subplots(figsize=(14, 7), facecolor='white')

    vmax = np.amax(np.abs(data))
    im = ax.imshow(data, extent=[0, data.shape[1], iterations * dt * 1e9, 0],
                   interpolation='nearest', aspect='auto', cmap='seismic',
                   vmin=-vmax, vmax=vmax)

    ax.set_xlabel('Trace number', fontsize=12)
    ax.set_ylabel('Time [ns]', fontsize=12)
    ax.set_title('MATLAB / Python 读取 gprMax B-scan 输出 —— Ez 分量（金属圆柱体双曲线反射）',
                 fontsize=14, fontweight='bold')
    ax.grid(which='both', axis='both', linestyle='-.', alpha=0.4)

    cb = plt.colorbar(im, ax=ax)
    cb.set_label('Field strength [V/m]', fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_Bscan.png'), dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print('Generated: fig_Bscan.png')


def fig_spectrum():
    """Generate frequency spectrum similar to outputfile_converter.m."""
    filename = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'user_models', 'cylinder_Bscan_2D_merged.out')
    if not os.path.exists(filename):
        print(f'Skip fig_spectrum.png: {filename} not found')
        return

    with h5py.File(filename, 'r') as f:
        dt = f.attrs['dt']
        iterations = f.attrs['Iterations']
        data = f['/rxs/rx1/Ez'][:]

    # FFT similar to MATLAB script
    m = 2 ** int(np.ceil(np.log2(iterations)))
    amp = np.fft.fft(data, n=m, axis=0)
    amp = (np.abs(amp[:m//2, :]) / m) * 2
    amp_mean = np.mean(amp, axis=1)

    samp_freq = (1 / dt) * 1e-6  # MHz
    freq = samp_freq * np.arange(m // 2) / m

    fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
    ax.fill_between(freq, amp_mean, alpha=0.7, color='black')
    ax.set_xlabel('Frequency [MHz]', fontsize=12)
    ax.set_ylabel('Amplitude', fontsize=12)
    ax.set_title('MATLAB / Python 频谱分析 —— Ricker 波形中心频率 1.5 GHz',
                 fontsize=14, fontweight='bold')
    ax.set_xlim([0, samp_freq / 2])
    ax.grid(True, linestyle='--', alpha=0.5)

    # Mark center frequency
    ax.axvline(x=1500, color='red', linestyle='--', linewidth=2, label='Center freq 1.5 GHz')
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_spectrum.png'), dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print('Generated: fig_spectrum.png')


def fig_matlab_code():
    """Generate MATLAB code example image."""
    fig, ax = plt.subplots(figsize=(10, 7), facecolor='#1e1e1e')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis('off')

    code_lines = [
        ("% MATLAB 读取 gprMax HDF5 输出", '#6A9955'),
        ("[filename, pathname] = uigetfile('*.out', 'Select gprMax output');", '#DCDCAA'),
        ("fullfile = strcat(pathname, filename);", '#DCDCAA'),
        ("", '#CCCCCC'),
        ("% 读取元数据", '#6A9955'),
        ("header.title = h5readatt(fullfile, '/', 'Title');", '#DCDCAA'),
        ("header.dt = h5readatt(fullfile, '/', 'dt');", '#DCDCAA'),
        ("header.iterations = h5readatt(fullfile, '/', 'Iterations');", '#DCDCAA'),
        ("", '#CCCCCC'),
        ("% 读取接收器场数据", '#6A9955'),
        ("data_ex = h5read(fullfile, '/rxs/rx1/Ex');", '#DCDCAA'),
        ("data_ez = h5read(fullfile, '/rxs/rx1/Ez');", '#DCDCAA'),
        ("", '#CCCCCC'),
        ("% 绘图", '#6A9955'),
        ("time = linspace(0, (header.iterations-1)*header.dt, header.iterations);", '#DCDCAA'),
        ("plot(time, data_ez, 'r', 'LineWidth', 2);", '#DCDCAA'),
        ("xlabel('Time [s]'); ylabel('Ez [V/m]');", '#DCDCAA'),
    ]

    y_start = 6.5
    line_height = 0.34
    for i, (line, color) in enumerate(code_lines):
        y = y_start - i * line_height
        ax.text(0.3, y, line, fontsize=11, color=color, ha='left', va='center',
                family='monospace')

    ax.text(5, 0.3, 'MATLAB 通过 h5read / h5readatt 直接读取 gprMax 输出',
            ha='center', fontsize=11, color='#CCCCCC', style='italic')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_matlab_code.png'), dpi=150, bbox_inches='tight',
                facecolor='#1e1e1e', edgecolor='none')
    plt.close()
    print('Generated: fig_matlab_code.png')


def fig_hdf5_structure():
    """Generate HDF5 file structure diagram."""
    fig, ax = plt.subplots(figsize=(10, 7), facecolor='white')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis('off')

    # Main file box
    main_box = FancyBboxPatch((1, 0.5), 8, 6, boxstyle="round,pad=0.1",
                              facecolor='#E7E6E6', edgecolor='black', linewidth=2)
    ax.add_patch(main_box)
    ax.text(5, 6.2, 'gprMax 输出文件 (.out) —— HDF5 格式', ha='center', fontsize=14,
            fontweight='bold')

    # Root attributes
    attr_box = FancyBboxPatch((1.5, 5.0), 3.5, 0.9, boxstyle="round,pad=0.05",
                              facecolor='#B4C7E7', edgecolor='black', linewidth=1.5)
    ax.add_patch(attr_box)
    ax.text(3.25, 5.45, 'Root Attributes\nTitle, dt, Iterations, nx_ny_nz...',
            ha='center', va='center', fontsize=10)

    # srcs
    src_box = FancyBboxPatch((1.5, 3.8), 3.5, 0.9, boxstyle="round,pad=0.05",
                             facecolor='#C5E0B4', edgecolor='black', linewidth=1.5)
    ax.add_patch(src_box)
    ax.text(3.25, 4.25, '/srcs/src1\nType, Position', ha='center', va='center', fontsize=10)

    # rxs
    rx_box = FancyBboxPatch((5.5, 3.8), 3.5, 0.9, boxstyle="round,pad=0.05",
                            facecolor='#F8CBAD', edgecolor='black', linewidth=1.5)
    ax.add_patch(rx_box)
    ax.text(7.25, 4.25, '/rxs/rx1\nPosition, Ex, Ey, Ez, Hx, Hy, Hz',
            ha='center', va='center', fontsize=10)

    # Data arrays detail
    data_box = FancyBboxPatch((5.5, 2.0), 3.5, 1.5, boxstyle="round,pad=0.05",
                              facecolor='#FFE699', edgecolor='black', linewidth=1.5)
    ax.add_patch(data_box)
    ax.text(7.25, 2.75, 'Field Data Arrays\nEx [iterations x 1]\nEz [iterations x 1]\n...',
            ha='center', va='center', fontsize=10)

    # Arrows
    ax.annotate('', xy=(3.25, 5.0), xytext=(3.25, 6.0),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.annotate('', xy=(3.25, 4.7), xytext=(3.25, 5.0),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.annotate('', xy=(7.25, 4.7), xytext=(7.25, 5.0),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.annotate('', xy=(7.25, 3.5), xytext=(7.25, 3.8),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    ax.text(5, 1.2, 'MATLAB 使用 h5read() 读取数组，h5readatt() 读取属性',
            ha='center', fontsize=11, style='italic', color='#333333')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_hdf5_structure.png'), dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print('Generated: fig_hdf5_structure.png')


def fig_ethics():
    """Generate ethics consideration diagram."""
    fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Title
    ax.text(6, 5.5, '工程伦理：仿真结果的使用责任', ha='center', fontsize=16,
            fontweight='bold', color='#1F4E78')

    boxes = [
        (1.0, 3.5, 2.5, 1.2, '#4472C4', '模型假设\n的局限性'),
        (4.0, 3.5, 2.5, 1.2, '#70AD47', '参数不确定性\n对结果的影响'),
        (7.0, 3.5, 2.5, 1.2, '#FFC000', '结果外推\n的风险评估'),
        (10.0, 3.5, 2.5, 1.2, '#C55A11', '数据真实性\n与可复现性'),
    ]

    for x, y, w, h, color, text in boxes:
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                             facecolor=color, edgecolor='black', linewidth=2, alpha=0.85)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=12, fontweight='bold', color='white')

    # Bottom text
    ax.text(6, 2.0, '正演仿真为工程决策提供依据，但决策者必须清楚模型的假设边界，\n'
            '避免将理想化仿真结果直接等同于现场实测数据。',
            ha='center', va='center', fontsize=12, color='#333333',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#F2F2F2', edgecolor='#999999'))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_ethics.png'), dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.close()
    print('Generated: fig_ethics.png')


if __name__ == '__main__':
    fig_workflow()
    fig_input_file()
    fig_Ascan()
    fig_Bscan()
    fig_spectrum()
    fig_matlab_code()
    fig_hdf5_structure()
    fig_ethics()
    print(f'\nAll images saved to: {OUTPUT_DIR}')
