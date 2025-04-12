#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np

class ImageEnhancement:
    """图像增强类：负责图像的直方图均衡化、对比度增强、亮度调整等操作"""
    
    def __init__(self):
        """初始化"""
        pass
    
    def histogram_equalization(self, image):
        """
        直方图均衡化
        
        参数:
            image: 输入图像
            
        返回:
            均衡化后的图像
        """
        if image is None:
            return None
            
        # 根据图像类型进行不同处理
        if len(image.shape) == 2:  # 灰度图像
            return cv2.equalizeHist(image)
        elif len(image.shape) == 3:  # 彩色图像
            # 转换到HSV空间
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            # 对亮度通道进行均衡化
            hsv[:,:,2] = cv2.equalizeHist(hsv[:,:,2])
            # 转换回BGR空间
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return image
    
    def adjust_contrast(self, image, alpha=1.0):
        """
        调整图像对比度
        
        参数:
            image: 输入图像
            alpha: 对比度系数，>1增加对比度，<1降低对比度
            
        返回:
            调整后的图像
        """
        if image is None or alpha < 0:
            return None
            
        # 对图像进行线性变换 new_img = alpha*img + beta
        # 其中beta保持为0以保持亮度不变
        return cv2.convertScaleAbs(image, alpha=alpha, beta=0)
    
    def adjust_brightness(self, image, beta=0):
        """
        调整图像亮度
        
        参数:
            image: 输入图像
            beta: 亮度调整值，>0增加亮度，<0降低亮度
            
        返回:
            调整后的图像
        """
        if image is None:
            return None
            
        # 对图像进行线性变换 new_img = alpha*img + beta
        # 其中alpha保持为1以保持对比度不变
        return cv2.convertScaleAbs(image, alpha=1, beta=beta)
    
    def adjust_saturation(self, image, saturation=1.0):
        """
        调整图像饱和度
        
        参数:
            image: 输入图像
            saturation: 饱和度系数，>1增加饱和度，<1降低饱和度
            
        返回:
            调整后的图像
        """
        if image is None or saturation < 0:
            return None
            
        # 如果是灰度图像，无法调整饱和度
        if len(image.shape) == 2:
            return image
            
        # 转换到HSV空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # 调整饱和度通道
        hsv[:,:,1] = hsv[:,:,1] * saturation
        
        # 饱和度通道限制在0-255范围内
        hsv[:,:,1] = np.clip(hsv[:,:,1], 0, 255)
        
        # 转换回BGR空间
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    def adjust_gamma(self, image, gamma=1.0):
        """
        调整图像的gamma值
        
        参数:
            image: 输入图像
            gamma: gamma值，>1使图像变暗，<1使图像变亮
            
        返回:
            调整后的图像
        """
        if image is None or gamma <= 0:
            return None
            
        # 创建gamma校正查找表
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                          for i in np.arange(0, 256)]).astype(np.uint8)
        
        # 应用查找表
        if len(image.shape) == 3:
            # 彩色图像，分别应用于每个通道
            return cv2.LUT(image, table)
        else:
            # 灰度图像
            return cv2.LUT(image, table)
    
    def sharpen(self, image, amount=1.0):
        """
        锐化图像
        
        参数:
            image: 输入图像
            amount: 锐化程度，值越大锐化效果越明显
            
        返回:
            锐化后的图像
        """
        if image is None or amount <= 0:
            return None
            
        # 使用高斯模糊创建模糊版本
        blurred = cv2.GaussianBlur(image, (0, 0), 3)
        
        # 计算锐化图像: img + amount*(img - blurred)
        sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
        
        return sharpened
    
    def denoise(self, image, strength=10):
        """
        去噪处理
        
        参数:
            image: 输入图像
            strength: 去噪强度，值越大去噪效果越明显
            
        返回:
            去噪后的图像
        """
        if image is None or strength <= 0:
            return None
            
        # 根据图像类型选择去噪方法
        if len(image.shape) == 2:  # 灰度图像
            return cv2.fastNlMeansDenoising(image, None, strength, 7, 21)
        elif len(image.shape) == 3:  # 彩色图像
            return cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)
        
        return image 