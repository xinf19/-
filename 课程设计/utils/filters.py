#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np

def apply_gaussian_filter(image, kernel_size=5, sigma=0):
    """
    应用高斯滤波器
    
    参数:
        image: 输入图像
        kernel_size: 核大小，必须为正奇数，默认5
        sigma: 标准差，默认0
    
    返回:
        滤波后的图像
    """
    # 确保kernel_size为正奇数
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

def apply_median_filter(image, kernel_size=5):
    """
    应用中值滤波器
    
    参数:
        image: 输入图像
        kernel_size: 核大小，必须为正奇数，默认5
    
    返回:
        滤波后的图像
    """
    # 确保kernel_size为正奇数
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    return cv2.medianBlur(image, kernel_size)

def apply_bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):
    """
    应用双边滤波器，保留边缘的同时平滑区域
    
    参数:
        image: 输入图像
        d: 滤波过程中的邻域直径，默认9
        sigma_color: 颜色空间的标准差，默认75
        sigma_space: 坐标空间的标准差，默认75
    
    返回:
        滤波后的图像
    """
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

def apply_box_filter(image, kernel_size=5, normalize=True):
    """
    应用均值滤波器（方框滤波器）
    
    参数:
        image: 输入图像
        kernel_size: 核大小，默认5
        normalize: 是否归一化，默认True
    
    返回:
        滤波后的图像
    """
    # 确保kernel_size为正奇数
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    return cv2.boxFilter(image, -1, (kernel_size, kernel_size), normalize=normalize)

def apply_custom_filter(image, kernel):
    """
    应用自定义滤波器/卷积核
    
    参数:
        image: 输入图像
        kernel: 自定义卷积核，numpy数组
    
    返回:
        滤波后的图像
    """
    return cv2.filter2D(image, -1, kernel)

def create_sharpening_kernel(amount=1.0):
    """
    创建锐化卷积核
    
    参数:
        amount: 锐化强度，默认1.0
    
    返回:
        锐化卷积核
    """
    kernel = np.array([[-amount, -amount, -amount],
                       [-amount, 1 + 8*amount, -amount],
                       [-amount, -amount, -amount]], dtype=np.float32)
    return kernel

def create_edge_detection_kernel(edge_type='sobel_x'):
    """
    创建边缘检测卷积核
    
    参数:
        edge_type: 边缘检测类型，'sobel_x', 'sobel_y', 'laplacian'，默认'sobel_x'
    
    返回:
        边缘检测卷积核
    """
    if edge_type == 'sobel_x':
        # 水平方向Sobel算子
        kernel = np.array([[-1, 0, 1],
                          [-2, 0, 2],
                          [-1, 0, 1]], dtype=np.float32)
    elif edge_type == 'sobel_y':
        # 垂直方向Sobel算子
        kernel = np.array([[-1, -2, -1],
                          [0, 0, 0],
                          [1, 2, 1]], dtype=np.float32)
    elif edge_type == 'laplacian':
        # Laplacian算子
        kernel = np.array([[0, 1, 0],
                          [1, -4, 1],
                          [0, 1, 0]], dtype=np.float32)
    else:
        # 默认为水平方向Sobel算子
        kernel = np.array([[-1, 0, 1],
                          [-2, 0, 2],
                          [-1, 0, 1]], dtype=np.float32)
    
    return kernel 