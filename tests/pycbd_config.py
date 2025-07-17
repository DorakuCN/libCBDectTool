#!/usr/bin/env python3
"""
PyCBD配置
使用我们的C++程序作为后端
"""

import os
import sys
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import tempfile
import subprocess
import time


class Checkerboard:
    """PyCBD兼容的Checkerboard类"""
    
    def __init__(self):
        self.norm = True
        self.score_thr = 0.01
        self.strict_grow = False
        self.show_grow_processing = False
        self.overlay = True
        self.show_debug_image = False
        self.cols = 0
        self.rows = 0
        self.number_of_corners = 0
        
        # 使用我们的C++程序
        self.cpp_program = "build/demo"
        if not os.path.exists(self.cpp_program):
            raise FileNotFoundError(f"C++程序不存在: {self.cpp_program}")
    
    def array_norm_to_image(self, image_array: np.ndarray, height: int, width: int):
        """将数组转换为图像"""
        self.image_array = image_array
        self.height = height
        self.width = width
        
    def find_corners(self):
        """查找角点"""
        # 保存图像到临时文件
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            # 将归一化数组转换回图像
            image = (self.image_array * 255).astype(np.uint8)
            cv2.imwrite(tmp_file.name, image)
            image_path = tmp_file.name
        
        try:
            # 运行C++程序
            result = subprocess.run(
                [self.cpp_program, image_path],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # 解析结果
            if result.returncode == 0:
                self._parse_cpp_output(result.stdout)
            else:
                print(f"C++程序执行失败: {result.stderr}")
                
        except Exception as e:
            print(f"执行错误: {e}")
        finally:
            # 清理临时文件
            if os.path.exists(image_path):
                os.unlink(image_path)
        
    def find_board_from_corners(self):
        """从角点查找棋盘"""
        # 这个功能已经在find_corners中完成
        pass
        
    def get_corners(self, corners_u: np.ndarray, corners_v: np.ndarray):
        """获取角点坐标"""
        if hasattr(self, 'detected_corners') and len(self.detected_corners) > 0:
            corners = np.array(self.detected_corners)
            n_corners = min(len(corners), len(corners_u))
            corners_u[:n_corners] = corners[:n_corners, 0]
            corners_v[:n_corners] = corners[:n_corners, 1]
            self.number_of_corners = n_corners
        
    def get_board_corners(self, board_u: np.ndarray, board_v: np.ndarray):
        """获取棋盘角点坐标"""
        if hasattr(self, 'board_corners') and len(self.board_corners) > 0:
            corners = np.array(self.board_corners)
            n_corners = min(len(corners), len(board_u))
            board_u[:n_corners] = corners[:n_corners, 0]
            board_v[:n_corners] = corners[:n_corners, 1]
            self.rows = int(np.sqrt(n_corners))
            self.cols = n_corners // self.rows
    
    def _parse_cpp_output(self, output: str):
        """解析C++程序输出"""
        lines = output.split('\n')
        
        # 提取角点数量
        corners_detected = 0
        chessboards_detected = 0
        
        for line in lines:
            if 'Detected' in line and 'corners' in line:
                try:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == 'Detected':
                            if i + 2 < len(parts) and parts[i + 2] == 'corners':
                                corners_detected = int(parts[i + 1])
                                break
                except:
                    pass
            elif 'Detected' in line and 'chessboards' in line:
                try:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == 'Detected':
                            if i + 2 < len(parts) and parts[i + 2] == 'chessboards':
                                chessboards_detected = int(parts[i + 1])
                                break
                except:
                    pass
        
        # 提取角点坐标
        corners = []
        corner_section = False
        
        for line in lines:
            if 'DEBUG: final corners' in line:
                corner_section = True
                continue
                
            if corner_section and line.strip():
                try:
                    if '(' in line and ')' in line:
                        coord_part = line.split('(')[1].split(')')[0]
                        x, y = map(float, coord_part.split(','))
                        corners.append([x, y])
                except:
                    pass
                    
                if len(corners) >= corners_detected:
                    break
        
        self.detected_corners = corners
        
        # 如果有棋盘检测到，使用所有角点作为棋盘角点
        if chessboards_detected > 0 and len(corners) > 0:
            self.board_corners = corners.copy()
        else:
            self.board_corners = []


class CheckerboardDetector:
    """PyCBD兼容的棋盘检测器"""
    
    def __init__(self):
        self.detector = Checkerboard()
    
    def detect_checkerboard(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        检测棋盘
        
        Args:
            image: 输入图像（numpy数组）
            
        Returns:
            board_uv: 棋盘角点的图像坐标 (u, v)
            board_xy: 棋盘角点的局部坐标 (x, y)
            corners_uv: 所有检测到的角点坐标 (u, v)
        """
        # 准备图像
        prepared_image = self._prepare_image(image)
        
        # 设置检测器参数
        self.detector.array_norm_to_image(prepared_image, prepared_image.shape[0], prepared_image.shape[1])
        
        # 执行检测
        self.detector.find_corners()
        self.detector.find_board_from_corners()
        
        # 提取结果
        if self.detector.rows > 0 and self.detector.cols > 0:
            board_uv, corners_uv = self._extract_corners()
            board_xy = self._calculate_local_coordinates(board_uv)
            return board_uv, board_xy, corners_uv
        else:
            return np.array([]), np.array([]), np.array([])
    
    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        """
        准备图像
        
        Args:
            image: 输入图像
            
        Returns:
            处理后的图像
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("输入必须是numpy数组")
        
        image = image.copy()
        
        # 转换为灰度图
        if len(image.shape) == 3:
            if image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif image.shape[2] != 1:
                raise ValueError("图像应该有1或3个通道")
        elif image.ndim != 2:
            raise ValueError("图像数组应该是2D或3D")
        
        # 转换为float并归一化
        image = image.astype(np.float64)
        image = (image - np.amin(image)) / np.ptp(image)
        
        return image
    
    def _extract_corners(self) -> Tuple[np.ndarray, np.ndarray]:
        """提取检测到的角点"""
        corners_uv = np.array(self.detector.detected_corners) if self.detector.detected_corners else np.array([])
        board_uv = np.array(self.detector.board_corners) if self.detector.board_corners else np.array([])
        
        return board_uv, corners_uv
    
    def _calculate_local_coordinates(self, board_uv: np.ndarray) -> np.ndarray:
        """计算局部坐标"""
        if board_uv.size == 0:
            return np.array([])
        
        n_corners = len(board_uv)
        board_xy = np.zeros((n_corners, 2))
        
        # 假设是矩形网格，计算行列数
        if n_corners > 0:
            cols = int(np.sqrt(n_corners))
            rows = n_corners // cols
            
            for i in range(n_corners):
                row = i // cols
                col = i % cols
                board_xy[i] = [row, col]
        
        return board_xy


class CBDPipeline:
    """PyCBD兼容的管道类"""
    
    def __init__(self, detector=None, expand: bool = False, predict: bool = False):
        """
        初始化管道
        
        Args:
            detector: 检测器实例
            expand: 是否扩展棋盘
            predict: 是否预测角点
        """
        if detector is None:
            detector = CheckerboardDetector()
        
        self.checkerboard_detector = detector
        self.expand = expand
        self.predict = predict
    
    def detect_checkerboard(self, image: np.ndarray, size: Optional[Tuple[int, int]] = None) -> Tuple[int, np.ndarray, np.ndarray]:
        """
        检测棋盘
        
        Args:
            image: 输入图像
            size: 棋盘尺寸 (rows, cols)
            
        Returns:
            result: 检测结果状态
            board_uv: 棋盘角点图像坐标
            board_xy: 棋盘角点局部坐标
        """
        board_uv, board_xy, corners_uv = self.checkerboard_detector.detect_checkerboard(image)
        
        # 验证结果
        if board_uv.size > 0:
            if size is not None:
                expected_corners = size[0] * size[1]
                if len(board_uv) == expected_corners:
                    result = 2  # 绝对坐标
                else:
                    result = 1  # 相对坐标
            else:
                result = 1  # 相对坐标
        else:
            result = 0  # 检测失败
        
        return result, board_uv, board_xy


def main():
    """测试函数"""
    print("PyCBD配置测试")
    print("=" * 50)
    
    # 检查C++程序
    if not os.path.exists("build/demo"):
        print("错误: C++程序不存在，请先编译")
        return
    
    # 创建检测器
    detector = CheckerboardDetector()
    
    # 测试图像
    test_image = "data/04.png"
    if os.path.exists(test_image):
        # 读取图像
        image = cv2.imread(test_image)
        print(f"图像尺寸: {image.shape}")
        
        # 检测棋盘
        board_uv, board_xy, corners_uv = detector.detect_checkerboard(image)
        
        print(f"检测结果:")
        print(f"  棋盘角点数量: {len(board_uv)}")
        print(f"  总角点数量: {len(corners_uv)}")
        
        if len(board_uv) > 0:
            print(f"  前5个棋盘角点:")
            for i, (uv, xy) in enumerate(zip(board_uv[:5], board_xy[:5])):
                print(f"    {i}: UV({uv[0]:.1f}, {uv[1]:.1f}) -> XY({xy[0]:.0f}, {xy[1]:.0f})")
    
    # 测试管道
    pipeline = CBDPipeline()
    result, board_uv, board_xy = pipeline.detect_checkerboard(image)
    
    print(f"\n管道检测结果:")
    print(f"  结果状态: {result}")
    print(f"  棋盘角点数量: {len(board_uv)}")


if __name__ == "__main__":
    main() 