#!/usr/bin/env python3
"""
libcbdet的Python包装器
用于连接我们的C++库和PyCBD
"""

import ctypes
import numpy as np
import os
import sys
from typing import Tuple, Optional
import cv2

class LibCBDetect:
    """libcbdet的Python包装器"""
    
    def __init__(self):
        # 尝试加载库文件
        lib_paths = [
            "build/libcbdetect.dylib",  # macOS
            "build/libcbdetect.so",     # Linux
            "build/cbdetect.dll",       # Windows
            "3rdparty/libcbdetCpp/lib/libcbdetect.dylib",  # 第三方库
        ]
        
        self.lib = None
        for lib_path in lib_paths:
            if os.path.exists(lib_path):
                try:
                    self.lib = ctypes.CDLL(lib_path)
                    print(f"✅ 加载库成功: {lib_path}")
                    self._setup_function_signatures()
                    break
                except Exception as e:
                    print(f"❌ 加载库失败 {lib_path}: {e}")
        
        if self.lib is None:
            print("❌ 无法加载任何库文件")
    
    def _setup_function_signatures(self):
        """设置函数签名"""
        if self.lib is None:
            return
        
        try:
            # 设置函数参数和返回类型
            # 这里需要根据实际的C++接口定义
            self.lib.detect_checkerboard.argtypes = [
                ctypes.c_void_p,  # image data
                ctypes.c_int,     # height
                ctypes.c_int,     # width
                ctypes.c_void_p,  # corners output
                ctypes.c_void_p,  # board output
            ]
            self.lib.detect_checkerboard.restype = ctypes.c_bool
            
            print("✅ 函数签名设置成功")
        except Exception as e:
            print(f"⚠️ 函数签名设置失败: {e}")
    
    def detect_checkerboard(self, image: np.ndarray) -> Tuple[bool, np.ndarray, np.ndarray]:
        """
        检测棋盘
        
        Args:
            image: 输入图像 (BGR格式)
            
        Returns:
            success: 是否成功
            board_uv: 棋盘角点坐标 (u, v)
            corners_uv: 所有检测到的角点坐标
        """
        if self.lib is None:
            print("❌ 库未加载")
            return False, np.array([]), np.array([])
        
        try:
            # 转换为灰度图
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            height, width = gray.shape
            
            # 准备输出数组
            max_corners = 1000
            corners_uv = np.zeros((max_corners, 2), dtype=np.float64)
            board_uv = np.zeros((max_corners, 2), dtype=np.float64)
            
            # 调用C++函数
            success = self.lib.detect_checkerboard(
                gray.ctypes.data_as(ctypes.c_void_p),
                height,
                width,
                corners_uv.ctypes.data_as(ctypes.c_void_p),
                board_uv.ctypes.data_as(ctypes.c_void_p)
            )
            
            if success:
                # 过滤掉零值
                valid_corners = corners_uv[np.any(corners_uv != 0, axis=1)]
                valid_board = board_uv[np.any(board_uv != 0, axis=1)]
                
                return True, valid_board, valid_corners
            else:
                return False, np.array([]), np.array([])
                
        except Exception as e:
            print(f"❌ 检测过程中出错: {e}")
            return False, np.array([]), np.array([])

# 创建全局实例
libcbdet = LibCBDetect()

# 为了兼容PyCBD，创建一个Checkerboard类
class Checkerboard:
    """兼容PyCBD的Checkerboard类"""
    
    def __init__(self):
        self.detector = libcbdet
        self.rows = 0
        self.cols = 0
        self.number_of_corners = 0
        self.norm = True
        self.score_thr = 0.01
        self.strict_grow = False
        self.show_grow_processing = False
        self.overlay = True
        self.show_debug_image = False
    
    def array_norm_to_image(self, image_array, height, width):
        """将数组标准化为图像"""
        self.image_array = image_array
        self.height = height
        self.width = width
    
    def find_corners(self):
        """查找角点"""
        # 这里可以调用我们的C++库
        pass
    
    def find_board_from_corners(self):
        """从角点查找棋盘"""
        # 这里可以调用我们的C++库
        pass
    
    def get_corners(self, corners_u, corners_v):
        """获取角点坐标"""
        # 这里可以调用我们的C++库
        pass
    
    def get_board_corners(self, board_u, board_v):
        """获取棋盘角点坐标"""
        # 这里可以调用我们的C++库
        pass

# 导出模块
__all__ = ['LibCBDetect', 'libcbdet', 'Checkerboard']
