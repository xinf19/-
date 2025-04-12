#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np

class FileOperations:
    """文件操作类：负责图像的打开、保存等操作"""
    
    def __init__(self):
        """初始化"""
        # 支持的图像格式
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        # 最后一次错误信息
        self.last_error = ""
    
    def open_image(self, filepath):
        """
        打开图像文件
        
        参数:
            filepath: 图像文件路径
            
        返回:
            成功返回图像数据，失败返回None
        """
        self.last_error = ""
        
        try:
            # 检查文件是否存在
            if not os.path.exists(filepath):
                self.last_error = f"文件不存在: {filepath}"
                print(self.last_error)
                return None
            
            # 检查文件扩展名
            _, ext = os.path.splitext(filepath)
            if ext.lower() not in self.supported_formats:
                self.last_error = f"不支持的文件格式: {ext}"
                print(self.last_error)
                return None
            
            # 尝试使用OpenCV读取图像
            img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
            if img is None:
                self.last_error = f"OpenCV无法读取图像: {filepath}"
                print(self.last_error)
                
                # 尝试使用其他方式读取
                try:
                    # 确保文件路径格式正确（特别是对中文路径）
                    abs_path = os.path.abspath(filepath)
                    img = cv2.imdecode(np.fromfile(abs_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                    if img is not None:
                        return img
                    else:
                        self.last_error = "备用方法也无法读取图像"
                        print(self.last_error)
                except Exception as e:
                    self.last_error = f"备用方法读取图像出错: {str(e)}"
                    print(self.last_error)
                
                return None
                
            return img
        except Exception as e:
            self.last_error = f"打开图像时出错: {str(e)}"
            print(self.last_error)
            return None
    
    def get_last_error(self):
        """获取最后一次错误信息"""
        return self.last_error
    
    def save_image(self, filepath, image):
        """
        保存图像文件
        
        参数:
            filepath: 保存的文件路径
            image: 要保存的图像数据
            
        返回:
            保存成功返回True，失败返回False
        """
        self.last_error = ""
        
        if image is None:
            self.last_error = "没有可保存的图像数据"
            print(self.last_error)
            return False
        
        _, ext = os.path.splitext(filepath)
        if not ext:
            self.last_error = "文件扩展名缺失"
            print(self.last_error)
            return False
            
        if ext.lower() not in self.supported_formats:
            self.last_error = f"不支持的文件格式: {ext}"
            print(self.last_error)
            return False
        
        try:
            # 使用OpenCV保存图像
            result = cv2.imwrite(filepath, image)
            if not result:
                self.last_error = f"保存图像失败: {filepath}"
                print(self.last_error)
                
                # 尝试使用其他方式保存（特别是对中文路径）
                try:
                    abs_path = os.path.abspath(filepath)
                    _, ext = os.path.splitext(abs_path)
                    is_success, buf = cv2.imencode(ext, image)
                    if is_success:
                        with open(abs_path, "wb") as f:
                            f.write(buf)
                        return True
                    else:
                        self.last_error = "备用方法无法保存图像"
                        print(self.last_error)
                        return False
                except Exception as e:
                    self.last_error = f"备用方法保存图像出错: {str(e)}"
                    print(self.last_error)
                    return False
                
                return False
                
            return True
        except Exception as e:
            self.last_error = f"保存图像时出错: {str(e)}"
            print(self.last_error)
            return False
    
    def convert_format(self, image, target_format):
        """
        转换图像格式
        
        参数:
            image: 原始图像数据
            target_format: 目标格式（'jpg', 'png', 'bmp'等）
            
        返回:
            转换后的图像数据
        """
        # 对于OpenCV，图像格式转换主要在保存时通过文件扩展名体现
        # 此处仅对图像进行预处理，如JPEG需要去除alpha通道等
        if image is None:
            return None
        
        target_format = target_format.lower()
        
        # 如果是JPEG格式，确保图像是BGR或灰度，没有alpha通道
        if target_format in ['jpg', 'jpeg']:
            if len(image.shape) == 3 and image.shape[2] == 4:
                # 去除alpha通道
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        
        return image 