#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (QMainWindow, QAction, QFileDialog, QLabel, 
                            QSlider, QVBoxLayout, QHBoxLayout, QWidget,
                            QPushButton, QMessageBox, QDockWidget, QScrollArea,
                            QGroupBox, QInputDialog)
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtCore import Qt, QSize

from core.file_ops import FileOperations
from core.preprocessing import ImagePreprocessing
from core.enhancement import ImageEnhancement
from core.morphology import MorphologyOperations
from core.segmentation import ImageSegmentation
from core.special_fx import SpecialEffects

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # 初始化变量
        self.image = None
        self.original_image = None
        self.current_image = None  # 当前状态的图像，用于连续操作
        self.filename = None
        
        # 初始化功能模块
        self.file_ops = FileOperations()
        self.preprocessing = ImagePreprocessing()
        self.enhancement = ImageEnhancement()
        self.morphology = MorphologyOperations()
        self.segmentation = ImageSegmentation()
        self.special_fx = SpecialEffects()
        
        # 设置窗口标题和大小 - 调整高度使界面更紧凑
        self.setWindowTitle("数字图像处理系统")
        self.resize(1024, 700)  # 减小窗口高度
        
        # 设置应用程序样式
        self.setStyleSheet("""
            QMainWindow {background-color: #f9f9f9;}
            QMenuBar {background-color: #f0f0f0; border-bottom: 1px solid #e0e0e0;}
            QMenuBar::item {padding: 4px 8px; background: transparent;}
            QMenuBar::item:selected {background: #e0e0e0; border-radius: 4px;}
            QMenu {background-color: #ffffff; border: 1px solid #e0e0e0;}
            QMenu::item {padding: 4px 16px 4px 20px;}
            QMenu::item:selected {background-color: #e5f1fb;}
            QStatusBar {background-color: #f0f0f0; color: #444; max-height: 20px;}
            QPushButton {padding: 4px 10px; border-radius: 4px;}
            QGroupBox {border: 1px solid #d0d0d0; border-radius: 6px; margin-top: 10px; font-weight: bold;}
            QGroupBox::title {subcontrol-origin: margin; subcontrol-position: top left; left: 8px; padding: 0 3px;}
        """)
        
        # 创建中央窗口部件
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # 创建主布局 - 减小边距
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(8, 8, 8, 8)
        self.main_layout.setSpacing(6)
        
        # 创建图像显示区域 - 减小高度
        self.image_label = QLabel("请打开一张图片")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(600, 400)  # 减小高度
        self.image_label.setStyleSheet("border: 1px solid #e0e0e0; background-color: white;")
        
        # 创建滚动区域用于大图像
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(self.image_label)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("QScrollArea {border: none;}")
        
        # 添加到主布局
        self.main_layout.addWidget(self.scroll_area)
        
        # 创建控制区域 - 减小高度
        control_layout = QHBoxLayout()
        control_layout.setSpacing(8)  # 减小按钮间距
        control_layout.setContentsMargins(0, 0, 0, 0)  # 减小边距
        
        # 创建Windows 11风格的按钮样式
        button_style = """
            QPushButton {
                background-color: #f0f0f0; 
                color: #202020; 
                border: 1px solid #d0d0d0; 
                border-radius: 4px; 
                padding: 4px 10px;
                min-height: 24px;
            }
            QPushButton:hover {
                background-color: #e5e5e5;
                border-color: #c0c0c0;
            }
            QPushButton:pressed {
                background-color: #d0d0d0;
            }
            QPushButton:disabled {
                background-color: #f5f5f5;
                color: #a0a0a0;
                border-color: #e0e0e0;
            }
        """
        
        # 打开图片按钮
        self.open_btn = QPushButton("打开图片")
        self.open_btn.clicked.connect(self.open_image)
        self.open_btn.setMinimumWidth(90)  # 减小宽度
        self.open_btn.setStyleSheet(button_style)
        
        # 保存图片按钮
        self.save_btn = QPushButton("保存图片")
        self.save_btn.clicked.connect(self.save_image)
        self.save_btn.setEnabled(False)
        self.save_btn.setMinimumWidth(90)  # 减小宽度
        self.save_btn.setStyleSheet(button_style)
        
        # 重置图片按钮
        self.reset_btn = QPushButton("重置图片")
        self.reset_btn.clicked.connect(self.reset_image)
        self.reset_btn.setEnabled(False)
        self.reset_btn.setMinimumWidth(90)  # 减小宽度
        self.reset_btn.setStyleSheet(button_style)
        
        # 添加按钮到控制布局
        control_layout.addWidget(self.open_btn)
        control_layout.addWidget(self.save_btn)
        control_layout.addWidget(self.reset_btn)
        control_layout.addStretch(1)  # 添加弹性空间
        
        # 添加控制布局到主布局
        self.main_layout.addLayout(control_layout)
        
        # 创建菜单
        self.create_menus()
        
        # 创建参数控制面板
        self.create_control_panel()
        
        # 创建状态栏并减小高度
        status_bar = self.statusBar()
        status_bar.setMaximumHeight(20)
        status_bar.showMessage("就绪")
    
    def create_menus(self):
        """创建菜单栏"""
        # 文件菜单
        file_menu = self.menuBar().addMenu("文件")
        
        open_action = QAction("打开", self)
        open_action.triggered.connect(self.open_image)
        file_menu.addAction(open_action)
        
        save_action = QAction("保存", self)
        save_action.triggered.connect(self.save_image)
        file_menu.addAction(save_action)
        
        save_as_action = QAction("另存为", self)
        save_as_action.triggered.connect(self.save_image_as)
        file_menu.addAction(save_as_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("退出", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 基本变换菜单
        basic_menu = self.menuBar().addMenu("基本变换")
        
        grayscale_action = QAction("灰度化", self)
        grayscale_action.triggered.connect(self.convert_to_grayscale)
        basic_menu.addAction(grayscale_action)
        
        binary_action = QAction("二值化", self)
        binary_action.triggered.connect(self.convert_to_binary)
        basic_menu.addAction(binary_action)
        
        # 预处理菜单
        preprocess_menu = self.menuBar().addMenu("图像预处理")
        
        # 噪声子菜单
        noise_menu = preprocess_menu.addMenu("噪声处理")
        
        add_gaussian_noise = QAction("添加高斯噪声", self)
        add_gaussian_noise.triggered.connect(lambda: self.add_noise("gaussian"))
        noise_menu.addAction(add_gaussian_noise)
        
        add_salt_pepper_noise = QAction("添加椒盐噪声", self)
        add_salt_pepper_noise.triggered.connect(lambda: self.add_noise("salt_pepper"))
        noise_menu.addAction(add_salt_pepper_noise)
        
        add_random_noise = QAction("添加随机噪声", self)
        add_random_noise.triggered.connect(lambda: self.add_noise("random"))
        noise_menu.addAction(add_random_noise)
        
        # 滤波子菜单
        filter_menu = preprocess_menu.addMenu("滤波处理")
        
        mean_filter = QAction("均值滤波", self)
        mean_filter.triggered.connect(lambda: self.apply_filter("mean"))
        filter_menu.addAction(mean_filter)
        
        median_filter = QAction("中值滤波", self)
        median_filter.triggered.connect(lambda: self.apply_filter("median"))
        filter_menu.addAction(median_filter)
        
        gaussian_filter = QAction("高斯滤波", self)
        gaussian_filter.triggered.connect(lambda: self.apply_filter("gaussian"))
        filter_menu.addAction(gaussian_filter)
        
        # 几何变换子菜单
        geometry_menu = preprocess_menu.addMenu("几何变换")
        
        rotate_action = QAction("旋转", self)
        rotate_action.triggered.connect(self.rotate_image)
        geometry_menu.addAction(rotate_action)
        
        resize_action = QAction("缩放", self)
        resize_action.triggered.connect(self.resize_image)
        geometry_menu.addAction(resize_action)
        
        translate_action = QAction("平移", self)
        translate_action.triggered.connect(self.translate_image)
        geometry_menu.addAction(translate_action)
        
        # 图像增强菜单
        enhance_menu = self.menuBar().addMenu("图像增强")
        
        # 直方图均衡化
        histogram_eq_action = QAction("直方图均衡化", self)
        histogram_eq_action.triggered.connect(self.histogram_equalization)
        enhance_menu.addAction(histogram_eq_action)
        
        # 对比度增强
        contrast_action = QAction("对比度增强", self)
        contrast_action.triggered.connect(self.enhance_contrast)
        enhance_menu.addAction(contrast_action)
        
        # 饱和度调整
        saturation_action = QAction("饱和度调整", self)
        saturation_action.triggered.connect(self.adjust_saturation)
        enhance_menu.addAction(saturation_action)
        
        # 亮度调整
        brightness_action = QAction("亮度调整", self)
        brightness_action.triggered.connect(self.adjust_brightness)
        enhance_menu.addAction(brightness_action)
        
        # 形态学操作菜单
        morphology_menu = self.menuBar().addMenu("形态学操作")
        
        erode_action = QAction("腐蚀", self)
        erode_action.triggered.connect(lambda: self.apply_morphology("erode"))
        morphology_menu.addAction(erode_action)
        
        dilate_action = QAction("膨胀", self)
        dilate_action.triggered.connect(lambda: self.apply_morphology("dilate"))
        morphology_menu.addAction(dilate_action)
        
        open_action = QAction("开运算", self)
        open_action.triggered.connect(lambda: self.apply_morphology("open"))
        morphology_menu.addAction(open_action)
        
        close_action = QAction("闭运算", self)
        close_action.triggered.connect(lambda: self.apply_morphology("close"))
        morphology_menu.addAction(close_action)
        
        # 图像分割菜单
        segment_menu = self.menuBar().addMenu("图像分割")
        
        threshold_action = QAction("阈值分割", self)
        threshold_action.triggered.connect(self.threshold_segmentation)
        segment_menu.addAction(threshold_action)
        
        region_growth_action = QAction("区域生长", self)
        region_growth_action.triggered.connect(self.region_growth_segmentation)
        segment_menu.addAction(region_growth_action)
        
        # 特效菜单
        effects_menu = self.menuBar().addMenu("特效")
        
        edge_detection_action = QAction("边缘检测", self)
        edge_detection_action.triggered.connect(self.edge_detection)
        effects_menu.addAction(edge_detection_action)
        
        mosaic_action = QAction("马赛克", self)
        mosaic_action.triggered.connect(self.apply_mosaic)
        effects_menu.addAction(mosaic_action)
        
        sketch_action = QAction("素描效果", self)
        sketch_action.triggered.connect(self.apply_sketch)
        effects_menu.addAction(sketch_action)
    
    def create_control_panel(self):
        """创建参数控制面板"""
        # 创建停靠窗口
        self.control_dock = QDockWidget("参数控制", self)
        self.control_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        # 设置控制面板宽度
        self.control_dock.setMinimumWidth(270)
        self.control_dock.setMaximumWidth(280)
        self.control_dock.setStyleSheet("QDockWidget::title {background-color: #f0f0f0; padding-left: 8px; padding-top: 3px; padding-bottom: 3px; max-height: 18px;}")
        
        # 创建控制面板容器
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        # 设置更紧凑的边距
        control_layout.setContentsMargins(6, 6, 6, 6)
        control_layout.setSpacing(6)
        
        # 旋转控制组（带应用按钮）
        rotation_group = self.create_slider_with_apply("旋转", -180, 180, 0, 1, self.on_rotation_changed)
        self.rotation_slider = rotation_group.findChild(QWidget)  # 获取滑块控件
        control_layout.addWidget(rotation_group)
        
        # 平移控制 - 使用更紧凑的布局
        translation_group = QGroupBox("平移")
        translation_layout = QVBoxLayout()
        translation_layout.setContentsMargins(6, 8, 6, 6)  # 更紧凑的内边距
        translation_layout.setSpacing(4)  # 减小间距
        
        # X轴平移
        x_layout = QHBoxLayout()
        x_layout.addWidget(QLabel("X轴:"))
        self.translation_x_slider = self.create_slider(-100, 100, 0, 1, self.on_translation_changed)
        x_layout.addWidget(self.translation_x_slider)
        translation_layout.addLayout(x_layout)
        
        # Y轴平移
        y_layout = QHBoxLayout()
        y_layout.addWidget(QLabel("Y轴:"))
        self.translation_y_slider = self.create_slider(-100, 100, 0, 1, self.on_translation_changed)
        y_layout.addWidget(self.translation_y_slider)
        translation_layout.addLayout(y_layout)
        
        translation_group.setLayout(translation_layout)
        control_layout.addWidget(translation_group)
        
        # 对比度控制（带应用按钮）
        contrast_group = self.create_slider_with_apply("对比度", 1, 300, 100, 1, self.on_contrast_changed)
        self.contrast_slider = contrast_group.findChild(QWidget)  # 获取滑块控件
        control_layout.addWidget(contrast_group)
        
        # 饱和度控制（带应用按钮）
        saturation_group = self.create_slider_with_apply("饱和度", 0, 300, 100, 1, self.on_saturation_changed)
        self.saturation_slider = saturation_group.findChild(QWidget)  # 获取滑块控件
        control_layout.addWidget(saturation_group)
        
        # 滤波核大小控制 - 更紧凑的布局
        filter_group = QGroupBox("滤波核大小")
        filter_layout = QVBoxLayout()
        filter_layout.setContentsMargins(6, 8, 6, 6)  # 更紧凑的内边距
        filter_layout.setSpacing(4)  # 减小间距
        
        self.filter_slider = self.create_slider(3, 25, 5, 2, None)  # 只允许奇数值，暂不设置回调
        filter_layout.addWidget(self.filter_slider)
        
        # 滤波类型按钮 - 更紧凑的布局
        filter_btns_layout = QHBoxLayout()
        filter_btns_layout.setSpacing(4)  # 减小按钮间距
        
        filter_btn_style = """
            QPushButton {
                background-color: #f0f0f0; 
                color: #202020; 
                border: 1px solid #d0d0d0; 
                border-radius: 3px; 
                padding: 3px 6px;
                min-height: 22px;
            }
            QPushButton:hover {
                background-color: #e5e5e5;
            }
            QPushButton:pressed {
                background-color: #d0d0d0;
            }
        """
        
        self.mean_filter_btn = QPushButton("均值滤波")
        self.mean_filter_btn.clicked.connect(lambda: self.apply_filter_with_slider("mean"))
        self.mean_filter_btn.setStyleSheet(filter_btn_style)
        
        self.median_filter_btn = QPushButton("中值滤波")
        self.median_filter_btn.clicked.connect(lambda: self.apply_filter_with_slider("median"))
        self.median_filter_btn.setStyleSheet(filter_btn_style)
        
        self.gaussian_filter_btn = QPushButton("高斯滤波")
        self.gaussian_filter_btn.clicked.connect(lambda: self.apply_filter_with_slider("gaussian"))
        self.gaussian_filter_btn.setStyleSheet(filter_btn_style)
        
        filter_btns_layout.addWidget(self.mean_filter_btn)
        filter_btns_layout.addWidget(self.median_filter_btn)
        filter_btns_layout.addWidget(self.gaussian_filter_btn)
        
        filter_layout.addLayout(filter_btns_layout)
        filter_group.setLayout(filter_layout)
        control_layout.addWidget(filter_group)
        
        # 阈值分割控制（带应用按钮）
        threshold_group = self.create_slider_with_apply("阈值分割", 0, 255, 128, 1, self.on_threshold_changed)
        self.threshold_slider = threshold_group.findChild(QWidget)  # 获取滑块控件
        control_layout.addWidget(threshold_group)
        
        # 添加空白区域
        control_layout.addStretch()
        
        # 设置dock widget的内容
        self.control_dock.setWidget(control_panel)
        
        # 将dock widget添加到主窗口
        self.addDockWidget(Qt.RightDockWidgetArea, self.control_dock)

    def create_slider(self, min_value, max_value, default_value, step, on_change_func):
        """创建滑块控件"""
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(min_value)
        slider.setMaximum(max_value)
        slider.setValue(default_value)
        slider.setSingleStep(step)
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval((max_value - min_value) // 10)
        # 增加滑块高度和样式
        slider.setFixedHeight(16)  # 减小高度
        slider.setStyleSheet("QSlider::groove:horizontal {height: 6px; background: #ddd; border-radius: 3px;} "
                             "QSlider::handle:horizontal {width: 14px; height: 14px; margin: -4px 0; border-radius: 7px; background: #6b7280;}")
        
        # 添加值显示标签
        value_label = QLabel(str(default_value))
        value_label.setFixedWidth(40)  # 减小宽度
        value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)  # 右对齐文本
        
        # 添加最小值和最大值标签
        min_label = QLabel(str(min_value))
        min_label.setFixedWidth(30)  # 减小宽度
        min_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        
        max_label = QLabel(str(max_value))
        max_label.setFixedWidth(30)  # 减小宽度
        max_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        
        # 创建滑块和当前值的水平布局
        slider_layout = QHBoxLayout()
        slider_layout.setSpacing(4)  # 减小间距
        slider_layout.addWidget(slider)
        slider_layout.addWidget(value_label)
        
        # 创建最小值和最大值的水平布局
        range_layout = QHBoxLayout()
        range_layout.setSpacing(4)
        range_layout.addWidget(min_label)
        range_layout.addStretch()
        range_layout.addWidget(max_label)
        
        # 创建垂直布局包含所有元素
        layout = QVBoxLayout()
        layout.setSpacing(2)  # 减小间距
        layout.addLayout(slider_layout)
        layout.addLayout(range_layout)
        
        # 创建容器widget
        container = QWidget()
        container.setLayout(layout)
        
        # 保存滑块引用到容器的属性中，方便后续直接访问
        container.slider = slider
        
        # 连接滑块值变化信号到回调函数和标签更新
        slider.valueChanged.connect(lambda v: value_label.setText(str(v)))
        if on_change_func:
            # 使用传入的回调函数
            slider.valueChanged.connect(on_change_func)
        
        return container
    
    # 为滑块控件组添加提交按钮
    def create_slider_with_apply(self, title, min_value, max_value, default_value, step, on_change_func):
        """创建滑块组控件，移除应用按钮，直接应用效果"""
        group = QGroupBox(title)
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 8, 6, 6)  # 更紧凑的内边距
        layout.setSpacing(4)  # 减小间距
        
        # 创建滑块
        slider_widget = self.create_slider(min_value, max_value, default_value, step, on_change_func)
        layout.addWidget(slider_widget)
        
        group.setLayout(layout)
        return group

    def open_image(self):
        """打开图像文件"""
        options = QFileDialog.Options()
        self.filename, _ = QFileDialog.getOpenFileName(
            self, "打开图像", "", 
            "图像文件 (*.png *.jpg *.bmp *.jpeg *.tif);;所有文件 (*)",
            options=options
        )
        
        if self.filename:
            self.image = self.file_ops.open_image(self.filename)
            if self.image is not None:
                self.original_image = self.image.copy()
                self.current_image = self.image.copy()  # 初始化当前状态图像
                self.display_image(self.image)
                self.save_btn.setEnabled(True)
                self.reset_btn.setEnabled(True)
                
                self.statusBar().showMessage(f"已打开: {self.filename}")
            else:
                QMessageBox.critical(self, "错误", "无法打开该图像文件！")
    
    def save_image(self):
        """保存当前图像"""
        if self.image is not None and self.filename:
            success = self.file_ops.save_image(self.filename, self.image)
            if success:
                self.statusBar().showMessage(f"图像已保存至: {self.filename}")
            else:
                self.save_image_as()
        else:
            self.save_image_as()
    
    def save_image_as(self):
        """另存为新文件"""
        if self.image is not None:
            options = QFileDialog.Options()
            filename, _ = QFileDialog.getSaveFileName(
                self, "保存图像", "", 
                "PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp);;TIFF (*.tif)",
                options=options
            )
            
            if filename:
                success = self.file_ops.save_image(filename, self.image)
                if success:
                    self.filename = filename
                    self.statusBar().showMessage(f"图像已保存至: {self.filename}")
                else:
                    QMessageBox.critical(self, "错误", "保存图像失败！")
    
    def reset_image(self):
        """重置图像到原始状态"""
        if self.original_image is not None:
            self.image = self.original_image.copy()
            self.current_image = self.original_image.copy()  # 重置当前状态图像
            self.display_image(self.image)
            
            # 重置所有滑块到默认值
            if hasattr(self, 'rotation_slider'):
                self.rotation_slider.slider.setValue(0)
            if hasattr(self, 'translation_x_slider'):
                self.translation_x_slider.slider.setValue(0)
            if hasattr(self, 'translation_y_slider'):
                self.translation_y_slider.slider.setValue(0)
            if hasattr(self, 'contrast_slider'):
                self.contrast_slider.slider.setValue(100)
            if hasattr(self, 'saturation_slider'):
                self.saturation_slider.slider.setValue(100)
            if hasattr(self, 'threshold_slider'):
                self.threshold_slider.slider.setValue(128)
            if hasattr(self, 'filter_slider'):
                self.filter_slider.slider.setValue(5)
            
            self.statusBar().showMessage("图像已重置到原始状态")
    
    def display_image(self, image):
        """在界面上显示图像"""
        if image is None:
            return
            
        # 将OpenCV的BGR格式转换为RGB格式
        if len(image.shape) == 3:  # 彩色图像
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, channels = rgb_image.shape
            bytes_per_line = channels * w
            q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        else:  # 灰度图像
            h, w = image.shape
            q_img = QImage(image.data, w, h, w, QImage.Format_Grayscale8)
        
        pixmap = QPixmap.fromImage(q_img)
        self.image_label.setPixmap(pixmap)
        self.image_label.setMinimumSize(1, 1)  # 允许图像正常缩放
        
        # 更新窗口标题显示图像尺寸
        if self.filename:
            base_filename = os.path.basename(self.filename)
            if len(image.shape) == 3:
                h, w, c = image.shape
                self.setWindowTitle(f"数字图像处理系统 - {base_filename} ({w}x{h}, {c}通道)")
            else:
                h, w = image.shape
                self.setWindowTitle(f"数字图像处理系统 - {base_filename} ({w}x{h}, 灰度)")
    
    # 基本变换功能
    def convert_to_grayscale(self):
        """将图像转换为灰度图"""
        if self.image is not None:
            self.image = self.preprocessing.to_grayscale(self.image)
            self.display_image(self.image)
            self.statusBar().showMessage("已转换为灰度图")
    
    def convert_to_binary(self):
        """将图像二值化"""
        if self.image is not None:
            self.image = self.preprocessing.to_binary(self.image)
            self.display_image(self.image)
            self.statusBar().showMessage("已转换为二值图")
    
    # 噪声处理功能
    def add_noise(self, noise_type):
        """向图像添加噪声"""
        if self.image is not None:
            self.image = self.preprocessing.add_noise(self.image, noise_type)
            self.display_image(self.image)
            self.statusBar().showMessage(f"已添加{noise_type}噪声")
    
    # 滤波处理功能
    def apply_filter(self, filter_type):
        """应用滤波器"""
        if self.image is not None:
            self.image = self.preprocessing.apply_filter(self.image, filter_type)
            self.display_image(self.image)
            self.statusBar().showMessage(f"已应用{filter_type}滤波")
    
    # 几何变换功能
    def rotate_image(self):
        """旋转图像"""
        if self.image is not None:
            angle, ok = QInputDialog.getInt(self, "旋转图像", "请输入旋转角度:", 0, -360, 360, 1)
            if ok:
                self.image = self.preprocessing.rotate(self.image, angle)
                self.display_image(self.image)
                self.statusBar().showMessage(f"图像已旋转{angle}度")
    
    def resize_image(self):
        """调整图像大小"""
        if self.image is not None:
            scale, ok = QInputDialog.getDouble(self, "缩放图像", "请输入缩放比例:", 1.0, 0.1, 10.0, 1)
            if ok:
                self.image = self.preprocessing.resize(self.image, scale)
                self.current_image = self.image.copy()  # 更新当前状态图像，使后续操作基于缩放后的图像
                self.display_image(self.image)
                self.statusBar().showMessage(f"图像已缩放，比例: {scale}")
    
    def translate_image(self):
        """平移图像"""
        if self.image is not None:
            x, ok1 = QInputDialog.getInt(self, "平移图像", "请输入X轴平移量:", 0, -1000, 1000, 1)
            if ok1:
                y, ok2 = QInputDialog.getInt(self, "平移图像", "请输入Y轴平移量:", 0, -1000, 1000, 1)
                if ok2:
                    self.image = self.preprocessing.translate(self.image, x, y)
                    self.display_image(self.image)
                    self.statusBar().showMessage(f"图像已平移，X: {x}, Y: {y}")
    
    # 图像增强功能
    def histogram_equalization(self):
        """直方图均衡化"""
        if self.image is not None:
            self.image = self.enhancement.histogram_equalization(self.image)
            self.display_image(self.image)
            self.statusBar().showMessage("已应用直方图均衡化")
    
    def enhance_contrast(self):
        """增强对比度"""
        if self.image is not None:
            alpha, ok = QInputDialog.getDouble(self, "对比度增强", "请输入对比度增强系数:", 1.0, 0.1, 3.0, 2)
            if ok:
                self.image = self.enhancement.adjust_contrast(self.image, alpha)
                self.display_image(self.image)
                self.statusBar().showMessage(f"已增强对比度，系数: {alpha}")
    
    def adjust_brightness(self):
        """调整亮度"""
        if self.image is not None:
            beta, ok = QInputDialog.getInt(self, "亮度调整", "请输入亮度调整值:", 0, -255, 255, 1)
            if ok:
                self.image = self.enhancement.adjust_brightness(self.image, beta)
                self.display_image(self.image)
                self.statusBar().showMessage(f"已调整亮度，值: {beta}")
    
    def adjust_saturation(self):
        """调整饱和度"""
        if self.image is not None:
            saturation, ok = QInputDialog.getDouble(self, "饱和度调整", "请输入饱和度系数:", 1.0, 0.0, 3.0, 2)
            if ok:
                self.image = self.enhancement.adjust_saturation(self.image, saturation)
                self.display_image(self.image)
                self.statusBar().showMessage(f"已调整饱和度，系数: {saturation}")
    
    # 形态学处理功能
    def apply_morphology(self, operation):
        """应用形态学操作"""
        if self.image is not None:
            kernel_size, ok = QInputDialog.getInt(self, "形态学操作", "请输入核大小:", 3, 1, 21, 2)
            if ok and kernel_size % 2 == 1:  # 确保核大小是奇数
                self.image = self.morphology.apply_operation(self.image, operation, kernel_size)
                self.display_image(self.image)
                self.statusBar().showMessage(f"已应用{operation}操作，核大小: {kernel_size}")
            elif ok:
                QMessageBox.warning(self, "警告", "核大小必须是奇数！")
    
    # 图像分割功能
    def threshold_segmentation(self):
        """阈值分割"""
        if self.image is not None:
            threshold, ok = QInputDialog.getInt(self, "阈值分割", "请输入阈值:", 128, 0, 255, 1)
            if ok:
                self.image = self.segmentation.threshold(self.image, threshold)
                self.display_image(self.image)
                self.statusBar().showMessage(f"已应用阈值分割，阈值: {threshold}")
    
    def region_growth_segmentation(self):
        """区域生长分割"""
        if self.image is not None:
            QMessageBox.information(self, "区域生长分割", "请在图像上点击种子点位置")
            # 此处需要实现鼠标点击事件获取种子点，简化起见暂时使用图像中心点
            h, w = self.image.shape[:2]
            seed_point = (w//2, h//2)
            self.image = self.segmentation.region_growing(self.image, seed_point)
            self.display_image(self.image)
            self.statusBar().showMessage(f"已应用区域生长分割，种子点: {seed_point}")
    
    # 特效功能
    def edge_detection(self):
        """边缘检测"""
        if self.image is not None:
            self.image = self.special_fx.edge_detection(self.image)
            self.display_image(self.image)
            self.statusBar().showMessage("已应用边缘检测")
    
    def apply_mosaic(self):
        """应用马赛克效果"""
        if self.image is not None:
            block_size, ok = QInputDialog.getInt(self, "马赛克效果", "请输入马赛克块大小:", 10, 2, 50, 1)
            if ok:
                self.image = self.special_fx.mosaic(self.image, block_size)
                self.display_image(self.image)
                self.statusBar().showMessage(f"已应用马赛克效果，块大小: {block_size}")
    
    def apply_sketch(self):
        """应用素描效果"""
        if self.image is not None:
            self.image = self.special_fx.sketch(self.image)
            self.display_image(self.image)
            self.statusBar().showMessage("已应用素描效果")
            
    # 添加滑块调整回调方法
    def on_rotation_changed(self):
        """旋转滑块值改变回调"""
        if self.image is not None and hasattr(self, 'rotation_slider'):
            # 使用直接引用访问滑块值
            angle = self.rotation_slider.slider.value()
            rotated_img = self.preprocessing.rotate(self.current_image.copy(), angle)
            if rotated_img is not None:
                self.image = rotated_img
                self.display_image(self.image)
                # 更新状态栏
                self.statusBar().showMessage(f"图像已旋转{angle}度")
    
    def on_translation_changed(self):
        """平移滑块值改变回调"""
        if self.image is not None and hasattr(self, 'translation_x_slider') and hasattr(self, 'translation_y_slider'):
            # 使用直接引用访问滑块值
            tx = self.translation_x_slider.slider.value()
            ty = self.translation_y_slider.slider.value()
            # 平移操作基于当前图像状态
            translated_img = self.preprocessing.translate(self.current_image.copy(), tx, ty)
            if translated_img is not None:
                self.image = translated_img
                self.display_image(self.image)
                # 更新状态栏
                self.statusBar().showMessage(f"图像已平移，X: {tx}, Y: {ty}")
    
    def on_contrast_changed(self):
        """对比度滑块值改变回调"""
        if self.image is not None and hasattr(self, 'contrast_slider'):
            # 使用直接引用访问滑块值
            contrast = self.contrast_slider.slider.value() / 100.0
            contrast_img = self.enhancement.adjust_contrast(self.current_image.copy(), contrast)
            if contrast_img is not None:
                self.image = contrast_img
                self.display_image(self.image)
                # 更新状态栏
                self.statusBar().showMessage(f"已调整对比度，系数: {contrast}")
    
    def on_saturation_changed(self):
        """饱和度滑块值改变回调"""
        if self.image is not None and hasattr(self, 'saturation_slider'):
            # 使用直接引用访问滑块值
            saturation = self.saturation_slider.slider.value() / 100.0
            sat_img = self.enhancement.adjust_saturation(self.current_image.copy(), saturation)
            if sat_img is not None:
                self.image = sat_img
                self.display_image(self.image)
                # 更新状态栏
                self.statusBar().showMessage(f"已调整饱和度，系数: {saturation}")
    
    def on_threshold_changed(self):
        """阈值分割滑块值改变回调"""
        if self.image is not None and hasattr(self, 'threshold_slider'):
            # 使用直接引用访问滑块值
            threshold = self.threshold_slider.slider.value()
            thresh_img = self.segmentation.threshold(self.current_image.copy(), threshold)
            if thresh_img is not None:
                self.image = thresh_img
                self.display_image(self.image)
                # 更新状态栏
                self.statusBar().showMessage(f"已应用阈值分割，阈值: {threshold}")
    
    def apply_filter_with_slider(self, filter_type):
        """应用带滑块参数的滤波器"""
        if self.image is not None and hasattr(self, 'filter_slider'):
            # 使用直接引用访问滑块值
            kernel_size = self.filter_slider.slider.value()
            # 确保kernel_size是奇数
            if kernel_size % 2 == 0:
                kernel_size += 1
                self.filter_slider.slider.setValue(kernel_size)
                
            filtered_img = self.preprocessing.apply_filter(self.current_image.copy(), filter_type, kernel_size)
            if filtered_img is not None:
                self.image = filtered_img
                self.display_image(self.image)
                self.statusBar().showMessage(f"已应用{filter_type}滤波，核大小: {kernel_size}") 