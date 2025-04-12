#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_histogram(image, channels=[0], mask=None, hist_size=[256], ranges=[0, 256]):
    """
    计算图像直方图
    
    参数:
        image: 输入图像
        channels: 要计算的通道，默认[0]（灰度图或BGR图像的蓝色通道）
        mask: 掩码，默认None
        hist_size: 直方图大小，默认[256]
        ranges: 值的范围，默认[0, 256]
    
    返回:
        计算得到的直方图
    """
    hist = cv2.calcHist([image], channels, mask, hist_size, ranges)
    return hist

def plot_histogram(image, title="Histogram", save_path=None):
    """
    绘制并显示图像直方图
    
    参数:
        image: 输入图像
        title: 图表标题，默认"Histogram"
        save_path: 保存路径，如果提供则保存图表，默认None
    """
    plt.figure(figsize=(10, 8))
    plt.title(title)
    
    if len(image.shape) == 2:  # 灰度图像
        hist = calculate_histogram(image)
        plt.plot(hist, color='gray')
        plt.xlim([0, 256])
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
    else:  # 彩色图像
        colors = ('b', 'g', 'r')
        for i, color in enumerate(colors):
            hist = calculate_histogram(image, [i])
            plt.plot(hist, color=color)
        plt.xlim([0, 256])
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
    
    if save_path:
        plt.savefig(save_path)
    
    plt.close()

def equalize_histogram(image):
    """
    对图像进行直方图均衡化
    
    参数:
        image: 输入图像
    
    返回:
        均衡化后的图像
    """
    if len(image.shape) == 2:  # 灰度图像
        return cv2.equalizeHist(image)
    else:  # 彩色图像
        # 转换为HSV颜色空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # 对亮度通道进行均衡化
        hsv[:,:,2] = cv2.equalizeHist(hsv[:,:,2])
        # 转换回BGR颜色空间
        equalized = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return equalized

def compare_histograms(hist1, hist2, method=cv2.HISTCMP_CORREL):
    """
    比较两个直方图的相似度
    
    参数:
        hist1: 第一个直方图
        hist2: 第二个直方图
        method: 比较方法，默认cv2.HISTCMP_CORREL（相关性）
    
    返回:
        相似度或距离值，取决于所选方法
    """
    return cv2.compareHist(hist1, hist2, method) 