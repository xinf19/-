#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np

class MorphologyOperations:
    """形态学操作类：负责图像的腐蚀、膨胀、开运算、闭运算等操作"""
    
    def __init__(self):
        """初始化"""
        pass
    
    def apply_operation(self, image, operation, kernel_size=3):
        """
        应用形态学操作
        
        参数:
            image: 输入图像
            operation: 操作类型，'erode'腐蚀, 'dilate'膨胀, 'open'开运算, 'close'闭运算
            kernel_size: 结构元素大小，默认3x3
            
        返回:
            处理后的图像
        """
        if image is None:
            return None
            
        # 创建结构元素
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        # 根据操作类型选择相应的形态学操作
        if operation == 'erode':
            return self.erode(image, kernel)
        elif operation == 'dilate':
            return self.dilate(image, kernel)
        elif operation == 'open':
            return self.open(image, kernel)
        elif operation == 'close':
            return self.close(image, kernel)
        else:
            return image
    
    def erode(self, image, kernel):
        """
        腐蚀操作
        
        参数:
            image: 输入图像
            kernel: 结构元素
            
        返回:
            腐蚀后的图像
        """
        if image is None:
            return None
            
        return cv2.erode(image, kernel, iterations=1)
    
    def dilate(self, image, kernel):
        """
        膨胀操作
        
        参数:
            image: 输入图像
            kernel: 结构元素
            
        返回:
            膨胀后的图像
        """
        if image is None:
            return None
            
        return cv2.dilate(image, kernel, iterations=1)
    
    def open(self, image, kernel):
        """
        开运算（先腐蚀后膨胀）
        
        参数:
            image: 输入图像
            kernel: 结构元素
            
        返回:
            开运算后的图像
        """
        if image is None:
            return None
            
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    
    def close(self, image, kernel):
        """
        闭运算（先膨胀后腐蚀）
        
        参数:
            image: 输入图像
            kernel: 结构元素
            
        返回:
            闭运算后的图像
        """
        if image is None:
            return None
            
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    
    def gradient(self, image, kernel_size=3):
        """
        形态学梯度（膨胀图像减去腐蚀图像）
        
        参数:
            image: 输入图像
            kernel_size: 结构元素大小，默认3x3
            
        返回:
            形态学梯度图像
        """
        if image is None:
            return None
            
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
    
    def top_hat(self, image, kernel_size=9):
        """
        顶帽操作（原图像减去开运算结果）
        
        参数:
            image: 输入图像
            kernel_size: 结构元素大小，默认9x9
            
        返回:
            顶帽操作后的图像
        """
        if image is None:
            return None
            
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
    
    def black_hat(self, image, kernel_size=9):
        """
        黑帽操作（闭运算结果减去原图像）
        
        参数:
            image: 输入图像
            kernel_size: 结构元素大小，默认9x9
            
        返回:
            黑帽操作后的图像
        """
        if image is None:
            return None
            
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
    
    def skeleton(self, image):
        """
        骨架提取
        
        参数:
            image: 输入图像（应为二值图像）
            
        返回:
            骨架图像
        """
        if image is None:
            return None
            
        # 确保图像为二值图像
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        else:
            _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        
        # 创建结构元素
        kernel = np.ones((3, 3), np.uint8)
        
        # 骨架图像初始化为全黑
        skeleton = np.zeros(binary.shape, np.uint8)
        
        # 原图像副本
        img = binary.copy()
        
        # 迭代提取骨架
        while True:
            # 开运算
            opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
            # 原图减去开运算结果
            temp = cv2.subtract(img, opened)
            # 腐蚀
            eroded = cv2.erode(img, kernel)
            # 更新骨架
            skeleton = cv2.bitwise_or(skeleton, temp)
            # 更新图像
            img = eroded.copy()
            
            # 如果图像全为0，则停止迭代
            if cv2.countNonZero(img) == 0:
                break
        
        return skeleton 