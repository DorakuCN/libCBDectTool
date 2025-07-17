#!/usr/bin/env python3
"""
libcbdetect Python Wrapper
使用subprocess调用C++可执行文件进行棋盘检测
"""

import os
import subprocess
import json
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import time
import tempfile


class LibCBDetect:
    """libcbdetect Python包装器"""
    
    def __init__(self, build_dir: str = "build"):
        """
        初始化libcbdetect包装器
        
        Args:
            build_dir: 编译目录路径
        """
        self.build_dir = Path(build_dir)
        self.available_programs = self._discover_programs()
        
    def _discover_programs(self) -> Dict[str, Path]:
        """发现可用的C++程序"""
        programs = {}
        
        # 主要的检测程序
        main_programs = [
            "demo", "pipeline_demo", "perfect_detection", 
            "fine_tuned", "optimized_debug", "debug_comparison"
        ]
        
        for prog in main_programs:
            prog_path = self.build_dir / prog
            if prog_path.exists() and os.access(prog_path, os.X_OK):
                programs[prog] = prog_path
                
        return programs
    
    def list_available_programs(self) -> List[str]:
        """列出可用的程序"""
        return list(self.available_programs.keys())
    
    def detect_checkerboard(self, 
                           image: Union[str, np.ndarray], 
                           program: str = "demo",
                           save_result: bool = True,
                           timeout: int = 30) -> Dict:
        """
        检测棋盘
        
        Args:
            image: 图像路径或numpy数组
            program: 使用的程序名称
            save_result: 是否保存结果图像
            timeout: 超时时间（秒）
            
        Returns:
            检测结果字典
        """
        if program not in self.available_programs:
            raise ValueError(f"程序 {program} 不可用。可用程序: {self.list_available_programs()}")
            
        prog_path = self.available_programs[program]
        
        # 处理输入图像
        if isinstance(image, np.ndarray):
            # 创建临时文件
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                cv2.imwrite(tmp_file.name, image)
                image_path = tmp_file.name
        else:
            image_path = str(image)
        
        # 准备命令
        cmd = [str(prog_path), image_path]
        
        try:
            # 运行程序
            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            end_time = time.time()
            
            # 解析输出
            output = result.stdout
            error = result.stderr
            
            # 提取关键信息
            detection_info = self._parse_output(output, error)
            detection_info.update({
                'program': program,
                'execution_time': end_time - start_time,
                'return_code': result.returncode,
                'command': ' '.join(cmd)
            })
            
            return detection_info
            
        except subprocess.TimeoutExpired:
            return {
                'error': f'程序执行超时 ({timeout}秒)',
                'program': program,
                'command': ' '.join(cmd)
            }
        except Exception as e:
            return {
                'error': f'执行错误: {str(e)}',
                'program': program,
                'command': ' '.join(cmd)
            }
        finally:
            # 清理临时文件
            if isinstance(image, np.ndarray) and os.path.exists(image_path):
                os.unlink(image_path)
    
    def _parse_output(self, output: str, error: str) -> Dict:
        """解析程序输出"""
        info = {
            'corners_detected': 0,
            'chessboards_detected': 0,
            'corners': [],
            'chessboards': [],
            'debug_info': {},
            'raw_output': output,
            'raw_error': error
        }
        
        lines = output.split('\n')
        
        # 提取角点数量
        for line in lines:
            if 'Detected' in line and 'corners' in line:
                try:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == 'Detected':
                            if i + 2 < len(parts) and parts[i + 2] == 'corners':
                                info['corners_detected'] = int(parts[i + 1])
                                break
                except:
                    pass
                    
            elif 'Detected' in line and 'chessboards' in line:
                try:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == 'Detected':
                            if i + 2 < len(parts) and parts[i + 2] == 'chessboards':
                                info['chessboards_detected'] = int(parts[i + 1])
                                break
                except:
                    pass
        
        # 提取角点坐标
        corner_section = False
        for line in lines:
            if 'DEBUG: final corners' in line or 'Corner' in line and ':' in line:
                corner_section = True
                continue
                
            if corner_section and line.strip():
                try:
                    # 解析角点坐标
                    if '(' in line and ')' in line:
                        coord_part = line.split('(')[1].split(')')[0]
                        x, y = map(float, coord_part.split(','))
                        info['corners'].append([x, y])
                except:
                    pass
                    
                if len(info['corners']) >= info['corners_detected']:
                    break
        
        return info
    
    def compare_programs(self, image: Union[str, np.ndarray], programs: Optional[List[str]] = None) -> Dict:
        """
        比较多个程序的检测结果
        
        Args:
            image: 图像路径或numpy数组
            programs: 要比较的程序列表，如果为None则使用所有可用程序
            
        Returns:
            比较结果
        """
        if programs is None:
            programs = self.list_available_programs()
        
        results = {}
        
        for program in programs:
            if program in self.available_programs:
                print(f"\n测试程序: {program}")
                result = self.detect_checkerboard(image, program, save_result=False)
                results[program] = result
                
        return results
    
    def batch_detect(self, image_dir: str, program: str = "demo") -> Dict:
        """
        批量检测目录中的图像
        
        Args:
            image_dir: 图像目录
            program: 使用的程序
            
        Returns:
            批量检测结果
        """
        image_dir = Path(image_dir)
        results = {}
        
        # 支持的图像格式
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
        
        for image_file in image_dir.iterdir():
            if image_file.suffix.lower() in image_extensions:
                print(f"\n处理图像: {image_file.name}")
                result = self.detect_checkerboard(str(image_file), program)
                results[image_file.name] = result
                
        return results


# 为了兼容PyCBD，创建一个Checkerboard类
class Checkerboard:
    """兼容PyCBD的Checkerboard类"""
    
    def __init__(self):
        self.detector = LibCBDetect()
        self.norm = True
        self.score_thr = 0.01
        self.strict_grow = False
        self.show_grow_processing = False
        self.overlay = True
        self.show_debug_image = False
        self.cols = 0
        self.rows = 0
        self.number_of_corners = 0
        
    def array_norm_to_image(self, image_array: np.ndarray, height: int, width: int):
        """将数组转换为图像"""
        self.image_array = image_array
        self.height = height
        self.width = width
        
    def find_corners(self):
        """查找角点"""
        # 这里可以调用C++程序进行角点检测
        pass
        
    def find_board_from_corners(self):
        """从角点查找棋盘"""
        # 这里可以调用C++程序进行棋盘检测
        pass
        
    def get_corners(self, corners_u: np.ndarray, corners_v: np.ndarray):
        """获取角点坐标"""
        # 返回检测到的角点坐标
        pass
        
    def get_board_corners(self, board_u: np.ndarray, board_v: np.ndarray):
        """获取棋盘角点坐标"""
        # 返回棋盘角点坐标
        pass


def main():
    """主函数 - 演示用法"""
    detector = LibCBDetect()
    
    print("可用的C++检测程序:")
    for prog in detector.list_available_programs():
        print(f"  - {prog}")
    
    # 测试单个图像
    test_image = "data/04.png"
    if os.path.exists(test_image):
        print(f"\n测试图像: {test_image}")
        
        # 使用默认程序检测
        result = detector.detect_checkerboard(test_image)
        print(f"检测结果: {result['corners_detected']} 个角点, {result['chessboards_detected']} 个棋盘")
        
        # 比较多个程序
        print("\n比较不同程序的结果:")
        comparison = detector.compare_programs(test_image, ["demo", "pipeline_demo", "optimized_debug"])
        
        for prog, res in comparison.items():
            if 'error' not in res:
                print(f"  {prog}: {res['corners_detected']} 角点, {res['chessboards_detected']} 棋盘, {res['execution_time']:.2f}s")
            else:
                print(f"  {prog}: 错误 - {res['error']}")


if __name__ == "__main__":
    main() 