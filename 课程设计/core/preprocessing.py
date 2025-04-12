#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import random

class ImagePreprocessing:
    """图像预处理类：负责图像的基本变换、噪声处理、滤波和几何变换"""
    
    def __init__(self):
        """初始化"""
        pass
    
    # 基本变换功能
    def to_grayscale(self, image):
        """
        将图像转换为灰度图
        
        参数:
            image: 输入图像
            
        返回:
            灰度图像
        """
        if image is None:
            return None
            
        # 如果已经是灰度图，直接返回
        if len(image.shape) == 2:
            return image
            
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    def to_binary(self, image, threshold=127, max_value=255):
        """
        将图像二值化
        
        参数:
            image: 输入图像
            threshold: 二值化阈值，默认127
            max_value: 超过阈值时的值，默认255
            
        返回:
            二值化图像
        """
        if image is None:
            return None
            
        # 如果是彩色图像，先转为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        _, binary = cv2.threshold(gray, threshold, max_value, cv2.THRESH_BINARY)
        return binary
    
    # 噪声处理功能
    def add_noise(self, image, noise_type='gaussian', amount=0.05):
        """
        向图像添加噪声
        
        参数:
            image: 输入图像
            noise_type: 噪声类型，'gaussian'高斯噪声，'salt_pepper'椒盐噪声，'random'随机噪声
            amount: 噪声强度，默认0.05
            
        返回:
            添加噪声后的图像
        """
        if image is None:
            return None
            
        # 复制原图，避免修改原图
        noisy = image.copy()
        
        if noise_type == 'gaussian':
            # 高斯噪声
            row, col, ch = noisy.shape if len(noisy.shape) == 3 else (*noisy.shape, 1)
            mean = 0
            var = 0.1
            sigma = var ** 0.5
            gauss = np.random.normal(mean, sigma, (row, col, ch))
            gauss = gauss.reshape(row, col, ch)
            noisy = noisy + gauss * 255
            noisy = np.clip(noisy, 0, 255)
            return noisy.astype(np.uint8)
            
        elif noise_type == 'salt_pepper':
            # 椒盐噪声
            s_vs_p = 0.5  # 盐噪声比例
            row, col, ch = noisy.shape if len(noisy.shape) == 3 else (*noisy.shape, 1)
            
            # 添加盐噪声
            num_salt = np.ceil(amount * row * col * s_vs_p)
            salt_coords = [np.random.randint(0, i - 1, int(num_salt)) for i in (row, col)]
            
            if len(noisy.shape) == 3:
                noisy[salt_coords[0], salt_coords[1], :] = 255
            else:
                noisy[salt_coords[0], salt_coords[1]] = 255
                
            # 添加椒噪声
            num_pepper = np.ceil(amount * row * col * (1. - s_vs_p))
            pepper_coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in (row, col)]
            
            if len(noisy.shape) == 3:
                noisy[pepper_coords[0], pepper_coords[1], :] = 0
            else:
                noisy[pepper_coords[0], pepper_coords[1]] = 0
                
            return noisy
            
        elif noise_type == 'random':
            # 随机噪声
            row, col = noisy.shape[:2]
            if len(noisy.shape) == 3:
                ch = noisy.shape[2]
                for i in range(row):
                    for j in range(col):
                        if random.random() < amount:
                            noisy[i, j] = [random.randint(0, 255) for _ in range(ch)]
            else:
                for i in range(row):
                    for j in range(col):
                        if random.random() < amount:
                            noisy[i, j] = random.randint(0, 255)
            return noisy
        
        return image
    
    # 滤波功能
    def apply_filter(self, image, filter_type='gaussian', kernel_size=5):
        """
        应用滤波器
        
        参数:
            image: 输入图像
            filter_type: 滤波器类型，'gaussian'高斯滤波，'median'中值滤波，'mean'均值滤波
            kernel_size: 滤波核大小，默认5
            
        返回:
            滤波后的图像
        """
        if image is None:
            return None
            
        # 确保kernel_size是奇数
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        if filter_type == 'gaussian':
            # 高斯滤波
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
            
        elif filter_type == 'median':
            # 中值滤波
            return cv2.medianBlur(image, kernel_size)
            
        elif filter_type == 'mean':
            # 均值滤波
            return cv2.blur(image, (kernel_size, kernel_size))
            
        return image
    
    # 几何变换功能
    def rotate(self, image, angle, center=None, scale=1.0):
        """
        旋转图像
        
        参数:
            image: 输入图像
            angle: 旋转角度，正值为逆时针
            center: 旋转中心，默认为图像中心
            scale: 旋转后的缩放因子，默认1.0
            
        返回:
            旋转后的图像
        """
        if image is None:
            return None
            
        h, w = image.shape[:2]
        if center is None:
            center = (w // 2, h // 2)
            
        # 计算旋转矩阵
        M = cv2.getRotationMatrix2D(center, angle, scale)
        
        # 应用旋转变换
        rotated = cv2.warpAffine(image, M, (w, h))
        return rotated
    
    def resize(self, image, scale_factor=1.0):
        """
        调整图像大小
        
        参数:
            image: 输入图像
            scale_factor: 缩放因子，大于1表示放大，小于1表示缩小
            
        返回:
            调整大小后的图像
        """
        if image is None or scale_factor <= 0:
            return None
            
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    def translate(self, image, tx, ty):
        """
        平移图像
        
        参数:
            image: 输入图像
            tx: X轴平移量，正值向右
            ty: Y轴平移量，正值向下
            
        返回:
            平移后的图像
        """
        if image is None:
            return None
            
        h, w = image.shape[:2]
        
        # 平移矩阵
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        
        # 应用平移变换
        translated = cv2.warpAffine(image, M, (w, h))
        return translated 