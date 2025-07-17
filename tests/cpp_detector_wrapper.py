#!/usr/bin/env python3
"""
C++ Detector Wrapper
使用编译好的C++程序进行棋盘检测
"""

import os
import subprocess
import json
import numpy as np
from pathlib import Path
import cv2
from typing import List, Dict, Tuple, Optional, Union
import time


class CppDetectorWrapper:
    """C++检测器包装器"""
    
    def __init__(self, build_dir: str = "build"):
        """
        初始化C++检测器包装器
        
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
    
    def detect_chessboard(self, 
                         image_path: str, 
                         program: str = "demo",
                         save_result: bool = True,
                         timeout: int = 30) -> Dict:
        """
        使用指定的C++程序检测棋盘
        
        Args:
            image_path: 图像路径
            program: 使用的程序名称
            save_result: 是否保存结果图像
            timeout: 超时时间（秒）
            
        Returns:
            检测结果字典
        """
        if program not in self.available_programs:
            raise ValueError(f"程序 {program} 不可用。可用程序: {self.list_available_programs()}")
            
        prog_path = self.available_programs[program]
        
        # 准备命令
        cmd = [str(prog_path), image_path]
        
        print(f"运行程序: {program}")
        print(f"命令: {' '.join(cmd)}")
        
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
    
    def compare_programs(self, image_path: str, programs: Optional[List[str]] = None) -> Dict:
        """
        比较多个程序的检测结果
        
        Args:
            image_path: 图像路径
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
                result = self.detect_chessboard(image_path, program, save_result=False)
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
                result = self.detect_chessboard(str(image_file), program)
                results[image_file.name] = result
                
        return results


def main():
    """主函数 - 演示用法"""
    wrapper = CppDetectorWrapper()
    
    print("可用的C++检测程序:")
    for prog in wrapper.list_available_programs():
        print(f"  - {prog}")
    
    # 测试单个图像
    test_image = "data/04.png"
    if os.path.exists(test_image):
        print(f"\n测试图像: {test_image}")
        
        # 使用默认程序检测
        result = wrapper.detect_chessboard(test_image)
        print(f"检测结果: {result['corners_detected']} 个角点, {result['chessboards_detected']} 个棋盘")
        
        # 比较多个程序
        print("\n比较不同程序的结果:")
        comparison = wrapper.compare_programs(test_image, ["demo", "pipeline_demo", "optimized_debug"])
        
        for prog, res in comparison.items():
            if 'error' not in res:
                print(f"  {prog}: {res['corners_detected']} 角点, {res['chessboards_detected']} 棋盘, {res['execution_time']:.2f}s")
            else:
                print(f"  {prog}: 错误 - {res['error']}")


if __name__ == "__main__":
    main() 