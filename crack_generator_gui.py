#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPR 裂缝模型生成器 v2.0 - 支持弯曲裂缝、GUI界面
"""

import os
import sys
import numpy as np
from scipy.interpolate import interp1d
from skimage.draw import line, polygon
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

# PyQt5 for GUI
try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QLineEdit, QPushButton, QSpinBox, QDoubleSpinBox,
        QGroupBox, QTextEdit, QProgressBar, QFileDialog, QMessageBox,
        QComboBox, QCheckBox
    )
    from PyQt5.QtCore import Qt, QThread, pyqtSignal
    from PyQt5.QtGui import QPixmap, QImage
    HAS_GUI = True
except ImportError:
    HAS_GUI = False
    print("PyQt5 not available, running in CLI mode")


class CrackGenerator:
    """裂缝模型生成器核心类"""
    
    def __init__(self):
        self.default_params = {
            'model_x': 5.0,
            'model_y': 2.0,
            'dx': 0.005,
            'dy': 0.005,
            'water_table_depth': 1.0,
            'crack_width': 0.05,  # 5cm宽
            'crack_type': 'air',
            'n_traces': 100,
            'center_freq': 200e6,
            'time_window': 60e-9,
        }
    
    def generate_bent_crack(
        self,
        output_dir: str,
        output_filename: str = "bent_crack_model",
        model_x: float = 5.0,
        model_y: float = 2.0,
        dx: float = 0.005,
        dy: float = 0.005,
        # 弯曲裂缝控制点 [(x1,y1), (x2,y2), (x3,y3)...]
        crack_points: List[Tuple[float, float]] = None,
        crack_width: float = 0.05,
        crack_type: str = "air",
        water_table_depth: float = 1.0,
        n_traces: int = 100,
        random_seed: Optional[int] = None,
        progress_callback = None
    ) -> Dict[str, Any]:
        """生成弯曲裂缝模型"""
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # 默认弯曲裂缝（参考图形状）
        if crack_points is None:
            # 从左上（浅）到右下（深）的弯曲裂缝 - 与参考图一致
            crack_points = [
                (1.0, 1.5),   # 起点：左上（浅，Y大）
                (1.5, 1.3),   # 中间点1
                (2.0, 1.0),   # 中间点2（弯曲）
                (3.0, 0.7),   # 中间点3
                (4.0, 0.5),   # 终点：右下（深，Y小）
            ]
        
        # 创建输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(output_dir, f"bent_crack_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        
        nx = int(model_x / dx)
        ny = int(model_y / dy)
        
        # 材料 ID
        MAT_IDS = {
            "free_space": 0,
            "dry_soil": 1,
            "saturated_soil": 2,
            "rock": 3,
            "crack": 4,
        }
        
        # 初始化模型
        model = np.zeros((ny, nx), dtype=np.int16)
        
        # 1. 生成地表和土壤
        x_indices = np.arange(nx)
        surface_y_grid = int(ny * 0.85)
        surface_line = np.full(nx, surface_y_grid)
        
        water_table_y_grid = int(ny - (water_table_depth / dy))
        
        for ix in range(nx):
            surface_idx = int(surface_line[ix])
            dry_soil_end = min(surface_idx, water_table_y_grid)
            
            if surface_idx > dry_soil_end:
                model[dry_soil_end:surface_idx, ix] = MAT_IDS["dry_soil"]
            if dry_soil_end > 0:
                model[0:dry_soil_end, ix] = MAT_IDS["saturated_soil"]
        
        if progress_callback:
            progress_callback(20, "Soil generated")
        
        # 2. 生成弯曲裂缝
        # 插值生成平滑曲线
        points = np.array(crack_points)
        t = np.linspace(0, 1, len(points))
        t_fine = np.linspace(0, 1, 100)  # 100个插值点
        
        fx = interp1d(t, points[:, 0], kind='cubic')
        fy = interp1d(t, points[:, 1], kind='cubic')
        
        x_fine = fx(t_fine)
        y_fine = fy(t_fine)
        
        # 绘制裂缝
        width = int(crack_width / dx)
        
        for i in range(len(x_fine) - 1):
            x1, y1 = int(x_fine[i] / dx), int(y_fine[i] / dy)
            x2, y2 = int(x_fine[i+1] / dx), int(y_fine[i+1] / dy)
            
            rr, cc = line(y1, x1, y2, x2)
            
            # 扩展线宽
            for offset in range(-width//2, width//2 + 1):
                rr_off = np.clip(rr + offset, 0, ny - 1)
                cc_off = np.clip(cc + offset, 0, nx - 1)
                model[rr_off, cc_off] = MAT_IDS["crack"]
        
        if progress_callback:
            progress_callback(40, "Bent crack generated")
        
        # 3. 导出 HDF5
        hdf5_path = os.path.join(output_dir, f"{output_filename}.h5")
        model_3d = model[np.newaxis, :, :].astype(np.int16)
        with h5py.File(hdf5_path, "w") as f:
            f.create_dataset("data", data=model_3d, dtype=np.int16)
            f.attrs["dx_dy_dz"] = (dx, dy, dx)
        
        # 4. 生成材料文件
        materials_path = os.path.join(output_dir, f"{output_filename}_materials.txt")
        
        if crack_type == "air":
            crack_eps, crack_sigma = 1, 0
        else:
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
        
        if progress_callback:
            progress_callback(60, "Files saved")
        
        # 5. 生成预览图
        png_path = os.path.join(output_dir, f"{output_filename}.png")
        colors = {
            0: [255, 255, 255],
            1: [255, 215, 0],
            2: [255, 215, 0],
            3: [139, 0, 0],
            4: [0, 0, 0],  # 裂缝黑色
        }
        rgb = np.zeros((ny, nx, 3), dtype=np.uint8)
        for id_val, color in colors.items():
            rgb[model == id_val] = color
        
        plt.figure(figsize=(10, 5))
        plt.imshow(rgb, origin="lower", aspect="auto", extent=[0, model_x, 0, model_y])
        plt.plot(points[:, 0], points[:, 1], 'r-', linewidth=3, label='Crack path')
        plt.scatter(points[:, 0], points[:, 1], c='red', s=50, zorder=5)
        plt.axhline(y=model_y - water_table_depth, color="cyan", linestyle="--", label="Water Table")
        plt.title("GPR Bent Crack Model")
        plt.xlabel("X (m)")
        plt.ylabel("Depth Y (m)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(png_path, dpi=150)
        plt.close()
        
        # 6. 生成 gprMax .in 文件
        hdf5_basename = os.path.basename(hdf5_path)
        materials_basename = os.path.basename(materials_path)
        
        available_width = model_x - 0.35
        src_steps = available_width / (n_traces - 1) if n_traces > 1 else 0.1
        src_steps = max(0.02, min(0.1, src_steps))
        
        in_template = f"""#title: Bent crack model for GPR simulation
#domain: {model_x} {model_y} {dx}
#dx_dy_dz: {dx} {dy} {dx}
#time_window: {60e-9}

#material: 6 0.01 1 0 dry_soil
#material: 25 0.1 1 0 saturated_soil
#material: 5 0.001 1 0 rock
#material: {crack_eps} {crack_sigma} 1 0 crack

#geometry_objects_read: 0 0 0 {hdf5_basename} {materials_basename}

#waveform: ricker 1 {200e6} my_ricker
#hertzian_dipole: z 0.05 {model_y - 0.05} 0 my_ricker
#rx: 0.05 {model_y - 0.05} 0
#src_steps: {src_steps:.3f} 0 0
#rx_steps: {src_steps:.3f} 0 0
"""
        
        in_path = os.path.join(output_dir, f"{output_filename}.in")
        with open(in_path, "w", encoding='utf-8') as f:
            f.write(in_template)
        
        if progress_callback:
            progress_callback(80, "gprMax config created")
        
        return {
            "hdf5_path": hdf5_path,
            "png_path": png_path,
            "in_template_path": in_path,
            "output_dir": output_dir,
            "materials_path": materials_path,
        }


if HAS_GUI:
    class WorkerThread(QThread):
        """后台工作线程"""
        progress = pyqtSignal(int, str)
        finished = pyqtSignal(dict)
        error = pyqtSignal(str)
        
        def __init__(self, generator, params):
            super().__init__()
            self.generator = generator
            self.params = params
        
        def run(self):
            try:
                def callback(percent, msg):
                    self.progress.emit(percent, msg)
                
                self.params['progress_callback'] = callback
                result = self.generator.generate_bent_crack(**self.params)
                self.finished.emit(result)
            except Exception as e:
                self.error.emit(str(e))
    
    
    class CrackGeneratorGUI(QMainWindow):
        """裂缝生成器 GUI"""
        
        def __init__(self):
            super().__init__()
            self.generator = CrackGenerator()
            self.init_ui()
        
        def init_ui(self):
            self.setWindowTitle("GPR 裂缝模型生成器 v2.0")
            self.setGeometry(100, 100, 800, 600)
            
            # 主布局
            central = QWidget()
            self.setCentralWidget(central)
            layout = QHBoxLayout(central)
            
            # 左侧面板 - 参数设置
            left_panel = QWidget()
            left_layout = QVBoxLayout(left_panel)
            
            # 模型尺寸组
            size_group = QGroupBox("模型尺寸")
            size_layout = QVBoxLayout()
            
            self.model_x = QDoubleSpinBox()
            self.model_x.setRange(1, 20)
            self.model_x.setValue(5.0)
            self.model_x.setSuffix(" m")
            size_layout.addWidget(QLabel("宽度 X:"))
            size_layout.addWidget(self.model_x)
            
            self.model_y = QDoubleSpinBox()
            self.model_y.setRange(0.5, 10)
            self.model_y.setValue(2.0)
            self.model_y.setSuffix(" m")
            size_layout.addWidget(QLabel("深度 Y:"))
            size_layout.addWidget(self.model_y)
            
            size_group.setLayout(size_layout)
            left_layout.addWidget(size_group)
            
            # 裂缝参数组
            crack_group = QGroupBox("裂缝参数")
            crack_layout = QVBoxLayout()
            
            self.crack_width = QDoubleSpinBox()
            self.crack_width.setRange(0.01, 0.2)
            self.crack_width.setValue(0.05)
            self.crack_width.setSuffix(" m")
            self.crack_width.setSingleStep(0.01)
            crack_layout.addWidget(QLabel("裂缝宽度:"))
            crack_layout.addWidget(self.crack_width)
            
            self.crack_type = QComboBox()
            self.crack_type.addItems(["air", "water"])
            crack_layout.addWidget(QLabel("填充类型:"))
            crack_layout.addWidget(self.crack_type)
            
            crack_group.setLayout(crack_layout)
            left_layout.addWidget(crack_group)
            
            # GPR参数组
            gpr_group = QGroupBox("GPR参数")
            gpr_layout = QVBoxLayout()
            
            self.n_traces = QSpinBox()
            self.n_traces.setRange(10, 200)
            self.n_traces.setValue(100)
            gpr_layout.addWidget(QLabel("扫描道数:"))
            gpr_layout.addWidget(self.n_traces)
            
            self.water_table = QDoubleSpinBox()
            self.water_table.setRange(0.1, 5)
            self.water_table.setValue(1.0)
            self.water_table.setSuffix(" m")
            gpr_layout.addWidget(QLabel("潜水面深度:"))
            gpr_layout.addWidget(self.water_table)
            
            gpr_group.setLayout(gpr_layout)
            left_layout.addWidget(gpr_group)
            
            # 生成按钮
            self.generate_btn = QPushButton("生成模型")
            self.generate_btn.setStyleSheet("font-size: 14px; padding: 10px;")
            self.generate_btn.clicked.connect(self.generate_model)
            left_layout.addWidget(self.generate_btn)
            
            # 运行gprMax按钮
            self.run_gprmax_btn = QPushButton("运行 gprMax 正演")
            self.run_gprmax_btn.setStyleSheet("font-size: 14px; padding: 10px;")
            self.run_gprmax_btn.setEnabled(False)
            self.run_gprmax_btn.clicked.connect(self.run_gprmax)
            left_layout.addWidget(self.run_gprmax_btn)
            
            left_layout.addStretch()
            layout.addWidget(left_panel, 1)
            
            # 右侧面板 - 预览和日志
            right_panel = QWidget()
            right_layout = QVBoxLayout(right_panel)
            
            # 预览图
            self.preview_label = QLabel("模型预览将显示在这里")
            self.preview_label.setAlignment(Qt.AlignCenter)
            self.preview_label.setMinimumSize(400, 300)
            self.preview_label.setStyleSheet("border: 1px solid gray; background: #f0f0f0;")
            right_layout.addWidget(self.preview_label)
            
            # 进度条
            self.progress = QProgressBar()
            right_layout.addWidget(self.progress)
            
            # 日志输出
            self.log_output = QTextEdit()
            self.log_output.setReadOnly(True)
            self.log_output.setPlaceholderText("日志输出...")
            right_layout.addWidget(self.log_output)
            
            layout.addWidget(right_panel, 2)
            
            self.current_output_dir = None
        
        def log(self, msg):
            self.log_output.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
        
        def generate_model(self):
            params = {
                'output_dir': r'D:\ClawX-Data\sim\gprmax_outcsv',
                'output_filename': 'bent_crack_model',
                'model_x': self.model_x.value(),
                'model_y': self.model_y.value(),
                'crack_width': self.crack_width.value(),
                'crack_type': self.crack_type.currentText(),
                'water_table_depth': self.water_table.value(),
                'n_traces': self.n_traces.value(),
            }
            
            self.log("开始生成模型...")
            self.generate_btn.setEnabled(False)
            self.progress.setValue(0)
            
            # 创建工作线程
            self.worker = WorkerThread(self.generator, params)
            self.worker.progress.connect(self.update_progress)
            self.worker.finished.connect(self.on_generation_finished)
            self.worker.error.connect(self.on_generation_error)
            self.worker.start()
        
        def update_progress(self, percent, msg):
            self.progress.setValue(percent)
            self.log(msg)
        
        def on_generation_finished(self, result):
            self.log(f"模型生成完成！")
            self.log(f"输出目录: {result['output_dir']}")
            
            # 显示预览图
            pixmap = QPixmap(result['png_path'])
            scaled = pixmap.scaled(self.preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.preview_label.setPixmap(scaled)
            
            self.current_output_dir = result['output_dir']
            self.run_gprmax_btn.setEnabled(True)
            self.generate_btn.setEnabled(True)
        
        def on_generation_error(self, error_msg):
            self.log(f"错误: {error_msg}")
            self.generate_btn.setEnabled(True)
        
        def run_gprmax(self):
            if not self.current_output_dir:
                return
            
            in_file = os.path.join(self.current_output_dir, 'bent_crack_model.in')
            n_traces = self.n_traces.value()
            
            self.log(f"启动 gprMax 正演 ({n_traces} 道)...")
            self.log("这可能需要 10-30 分钟...")
            self.run_gprmax_btn.setEnabled(False)
            
            # TODO: 启动 gprMax 进程
            # 这里可以添加 subprocess 调用 gprMax


def main():
    if HAS_GUI and len(sys.argv) == 1:
        # GUI 模式
        app = QApplication(sys.argv)
        window = CrackGeneratorGUI()
        window.show()
        sys.exit(app.exec_())
    else:
        # CLI 模式
        gen = CrackGenerator()
        result = gen.generate_bent_crack(
            output_dir=r'D:\ClawX-Data\sim\gprmax_outcsv',
            output_filename='bent_crack_model',
            model_x=5.0,
            model_y=2.0,
            crack_width=0.05,
            n_traces=100,
        )
        print(f"\n模型已生成: {result['output_dir']}")


if __name__ == "__main__":
    main()
