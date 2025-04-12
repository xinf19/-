#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from collections import deque

class ImageSegmentation:
    """图像分割类：负责图像的阈值分割、区域生长分割等操作"""
    
    def __init__(self):
        """初始化"""
        pass
    
    def threshold(self, image, threshold_value=127, max_value=255, method=cv2.THRESH_BINARY):
        """
        阈值分割
        
        参数:
            image: 输入图像
            threshold_value: 阈值，默认127
            max_value: 超过阈值时的值，默认255
            method: 阈值处理方法，默认cv2.THRESH_BINARY
            
        返回:
            分割后的图像
        """
        if image is None:
            return None
            
        # 如果是彩色图像，先转为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        _, thresholded = cv2.threshold(gray, threshold_value, max_value, method)
        return thresholded
    
    def adaptive_threshold(self, image, max_value=255, method=cv2.THRESH_BINARY, 
                           adaptive_method=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                           block_size=11, C=2):
        """
        自适应阈值分割
        
        参数:
            image: 输入图像
            max_value: 超过阈值时的值，默认255
            method: 阈值处理方法，默认cv2.THRESH_BINARY
            adaptive_method: 自适应方法，默认cv2.ADAPTIVE_THRESH_GAUSSIAN_C
            block_size: 计算阈值的区域大小，默认11
            C: 常数，阈值等于平均值或加权平均值减去该值，默认2
            
        返回:
            分割后的图像
        """
        if image is None:
            return None
            
        # 确保图像为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # 确保block_size为奇数
        if block_size % 2 == 0:
            block_size += 1
            
        return cv2.adaptiveThreshold(gray, max_value, adaptive_method, method, block_size, C)
    
    def otsu_threshold(self, image, max_value=255):
        """
        Otsu阈值分割
        
        参数:
            image: 输入图像
            max_value: 超过阈值时的值，默认255
            
        返回:
            分割后的图像和计算的阈值
        """
        if image is None:
            return None
            
        # 确保图像为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # 应用Otsu阈值法
        _, thresholded = cv2.threshold(gray, 0, max_value, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresholded
    
    def region_growing(self, image, seed_point, threshold=10):
        """
        区域生长分割
        
        参数:
            image: 输入图像
            seed_point: 种子点坐标 (x, y)
            threshold: 阈值，判断相邻像素是否属于同一区域，默认10
            
        返回:
            分割后的图像
        """
        if image is None or not isinstance(seed_point, tuple) or len(seed_point) != 2:
            return None
            
        # 确保图像为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        height, width = gray.shape
        x, y = seed_point
        
        # 检查种子点是否在图像范围内
        if x < 0 or x >= width or y < 0 or y >= height:
            return None
            
        # 创建标记矩阵，0表示未分割，255表示已分割
        segmented = np.zeros_like(gray)
        
        # 种子点的像素值
        seed_value = gray[y, x]
        
        # 使用队列进行区域生长
        queue = deque([(x, y)])
        segmented[y, x] = 255
        
        # 四邻域或八邻域，这里使用八邻域
        neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        
        while queue:
            cur_x, cur_y = queue.popleft()
            
            for dx, dy in neighbors:
                nx, ny = cur_x + dx, cur_y + dy
                
                # 检查是否在图像范围内且未被分割
                if (0 <= nx < width and 0 <= ny < height and 
                    segmented[ny, nx] == 0 and 
                    abs(int(gray[ny, nx]) - int(seed_value)) <= threshold):
                    segmented[ny, nx] = 255
                    queue.append((nx, ny))
        
        return segmented
    
    def watershed(self, image):
        """
        分水岭分割
        
        参数:
            image: 输入图像
            
        返回:
            分割后的图像
        """
        if image is None:
            return None
            
        # 如果是灰度图像，转换为BGR
        if len(image.shape) == 2:
            img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            img = image.copy()
            
        # 转为灰度图并进行阈值处理
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 噪声去除
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # 确定背景区域
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        # 确定前景区域
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        
        # 找到未知区域
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # 标记
        _, markers = cv2.connectedComponents(sure_fg)
        
        # 为所有的标记加1，保证背景是0而不是1
        markers = markers + 1
        
        # 标记未知区域为0
        markers[unknown == 255] = 0
        
        # 应用分水岭算法
        markers = cv2.watershed(img, markers)
        
        # 标记边界
        img[markers == -1] = [0, 0, 255]  # 边界标记为红色
        
        return img
    
    def grab_cut(self, image, rect=None):
        """
        GrabCut分割
        
        参数:
            image: 输入图像
            rect: 矩形区域，格式为(x, y, width, height)，默认为整个图像
            
        返回:
            分割后的图像
        """
        if image is None or len(image.shape) != 3:
            return None
            
        # 复制原图
        img = image.copy()
        
        # 如果没有指定矩形区域，使用整个图像
        if rect is None:
            height, width = img.shape[:2]
            rect = (0, 0, width, height)
            
        # 创建掩码
        mask = np.zeros(img.shape[:2], np.uint8)
        
        # 创建临时数组
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        # 执行GrabCut
        cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        
        # 修改掩码，其中0和2做为背景，1和3做为前景
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        
        # 使用修改后的掩码获取前景
        result = img * mask2[:, :, np.newaxis]
        
        return result 