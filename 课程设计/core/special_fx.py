#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np

class SpecialEffects:
    """特效类：负责实现边缘检测、马赛克、素描等特殊效果"""
    
    def __init__(self):
        """初始化"""
        pass
    
    def edge_detection(self, image, method='canny', threshold1=100, threshold2=200):
        """
        边缘检测
        
        参数:
            image: 输入图像
            method: 边缘检测方法，'canny'、'sobel'、'laplacian'，默认'canny'
            threshold1, threshold2: Canny边缘检测的阈值
            
        返回:
            边缘检测后的图像
        """
        if image is None:
            return None
            
        # 转为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        if method == 'canny':
            # Canny边缘检测
            edges = cv2.Canny(gray, threshold1, threshold2)
        elif method == 'sobel':
            # Sobel边缘检测
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            
            # 计算边缘梯度和方向
            edges = cv2.magnitude(sobel_x, sobel_y)
            
            # 归一化到0-255
            edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        elif method == 'laplacian':
            # Laplacian边缘检测
            edges = cv2.Laplacian(gray, cv2.CV_64F)
            
            # 取绝对值，并归一化到0-255
            edges = np.absolute(edges)
            edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        else:
            edges = gray
            
        return edges
    
    def mosaic(self, image, block_size=10):
        """
        马赛克效果
        
        参数:
            image: 输入图像
            block_size: 马赛克块大小，默认10
            
        返回:
            马赛克化后的图像
        """
        if image is None or block_size <= 0:
            return None
            
        # 复制原图
        mosaic_img = image.copy()
        
        # 获取图像宽高
        height, width = image.shape[:2]
        
        # 处理每个块
        for y in range(0, height, block_size):
            for x in range(0, width, block_size):
                # 确定块的范围
                x_end = min(x + block_size, width)
                y_end = min(y + block_size, height)
                
                # 获取块区域
                region = image[y:y_end, x:x_end]
                
                # 计算块内像素的平均值
                if len(image.shape) == 3:  # 彩色图像
                    b, g, r = cv2.mean(region)[:3]
                    color = (int(b), int(g), int(r))
                else:  # 灰度图像
                    color = int(cv2.mean(region)[0])
                
                # 填充块区域
                mosaic_img[y:y_end, x:x_end] = color
        
        return mosaic_img
    
    def sketch(self, image, ksize=7, sigma=0):
        """
        素描效果
        
        参数:
            image: 输入图像
            ksize: 高斯模糊核大小，默认7
            sigma: 高斯模糊核标准差，默认0
            
        返回:
            素描效果图像
        """
        if image is None:
            return None
            
        # 转为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # 对灰度图像进行高斯模糊
        blurred = cv2.GaussianBlur(gray, (ksize, ksize), sigma)
        
        # 颜色反转
        inverted = 255 - blurred
        
        # 混合图像
        sketch_img = cv2.divide(gray, inverted, scale=256.0)
        
        return sketch_img
    
    def cartoon(self, image, blur_ksize=5, threshold1=100, threshold2=200):
        """
        卡通化效果
        
        参数:
            image: 输入图像
            blur_ksize: 边缘提取前的模糊处理核大小，默认5
            threshold1, threshold2: Canny边缘检测的阈值
            
        返回:
            卡通化后的图像
        """
        if image is None or len(image.shape) != 3:
            return None
            
        # 复制原图
        img = image.copy()
        
        # 提取边缘
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, blur_ksize)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                      cv2.THRESH_BINARY, 9, 9)
        
        # 应用双边滤波减少细节
        color = cv2.bilateralFilter(img, 9, 300, 300)
        
        # 将边缘与处理后的图像结合
        cartoon = cv2.bitwise_and(color, color, mask=edges)
        
        return cartoon
    
    def emboss(self, image):
        """
        浮雕效果
        
        参数:
            image: 输入图像
            
        返回:
            浮雕效果图像
        """
        if image is None:
            return None
            
        # 定义浮雕卷积核
        kernel = np.array([[-2, -1, 0],
                          [-1, 1, 1],
                          [0, 1, 2]], dtype=np.float32)
        
        # 转为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # 应用滤波器
        emboss_img = cv2.filter2D(gray, -1, kernel)
        
        # 将结果调整到合适的范围
        emboss_img = cv2.normalize(emboss_img, None, 0, 255, cv2.NORM_MINMAX)
        
        return emboss_img.astype(np.uint8)
    
    def sepia(self, image):
        """
        棕褐色调效果
        
        参数:
            image: 输入图像
            
        返回:
            棕褐色调效果图像
        """
        if image is None or len(image.shape) != 3:
            return None
            
        # 棕褐色变换矩阵
        sepia_matrix = np.array([
            [0.393, 0.769, 0.189],
            [0.349, 0.686, 0.168],
            [0.272, 0.534, 0.131]
        ])
        
        # 复制原图
        sepia_img = image.copy()
        
        # 应用变换
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                b, g, r = image[y, x]
                new_pixel = np.dot(sepia_matrix, [r, g, b])
                new_pixel = np.clip(new_pixel, 0, 255)
                sepia_img[y, x] = (new_pixel[2], new_pixel[1], new_pixel[0])
        
        return sepia_img
    
    def vignette(self, image, amount=0.5):
        """
        晕影效果
        
        参数:
            image: 输入图像
            amount: 晕影强度，0-1之间，默认0.5
            
        返回:
            晕影效果图像
        """
        if image is None:
            return None
            
        # 复制原图
        vignette_img = image.copy()
        
        # 获取图像宽高
        height, width = image.shape[:2]
        
        # 创建晕影掩码
        mask = np.ones_like(image) * 255
        
        # 图像中心
        center_x, center_y = width // 2, height // 2
        
        # 计算最大半径（从中心到角落）
        max_radius = np.sqrt(center_x ** 2 + center_y ** 2)
        
        # 应用晕影效果
        for y in range(height):
            for x in range(width):
                # 计算到中心的距离
                distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                
                # 基于距离计算晕影系数
                vignette_coef = 1 - amount * (distance / max_radius)
                vignette_coef = max(0, vignette_coef)
                
                # 应用晕影效果
                if len(image.shape) == 3:  # 彩色图像
                    vignette_img[y, x] = vignette_img[y, x] * vignette_coef
                else:  # 灰度图像
                    vignette_img[y, x] = int(vignette_img[y, x] * vignette_coef)
        
        return vignette_img 