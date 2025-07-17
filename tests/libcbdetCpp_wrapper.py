#!/usr/bin/env python3
"""
libcbdetCpp Python Wrapper
使用libcbdetCpp库进行棋盘检测
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


class LibCBDetCppWrapper:
    """libcbdetCpp Python包装器"""
    
    def __init__(self, libcbdetCpp_dir: str = "3rdparty/libcbdetCpp"):
        """
        初始化libcbdetCpp包装器
        
        Args:
            libcbdetCpp_dir: libcbdetCpp目录路径
        """
        self.libcbdetCpp_dir = Path(libcbdetCpp_dir)
        self.example_path = self.libcbdetCpp_dir / "bin" / "example"
        self.example_data_dir = self.libcbdetCpp_dir / "example_data"
        
        if not self.example_path.exists():
            raise FileNotFoundError(f"example程序不存在: {self.example_path}")
            
        if not os.access(self.example_path, os.X_OK):
            raise PermissionError(f"example程序无执行权限: {self.example_path}")
    
    def detect_checkerboard(self, 
                           image: Union[str, np.ndarray], 
                           save_result: bool = True,
                           timeout: int = 30) -> Dict:
        """
        检测棋盘
        
        Args:
            image: 图像路径或numpy数组
            save_result: 是否保存结果图像
            timeout: 超时时间（秒）
            
        Returns:
            检测结果字典
        """
        # 处理输入图像
        if isinstance(image, np.ndarray):
            # 创建临时文件
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                cv2.imwrite(tmp_file.name, image)
                image_path = tmp_file.name
        else:
            image_path = str(image)
        
        # 准备命令
        cmd = [str(self.example_path), image_path]
        
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
                'program': 'libcbdetCpp_example',
                'execution_time': end_time - start_time,
                'return_code': result.returncode,
                'command': ' '.join(cmd)
            })
            
            return detection_info
            
        except subprocess.TimeoutExpired:
            return {
                'error': f'程序执行超时 ({timeout}秒)',
                'program': 'libcbdetCpp_example',
                'command': ' '.join(cmd)
            }
        except Exception as e:
            return {
                'error': f'执行错误: {str(e)}',
                'program': 'libcbdetCpp_example',
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
            'raw_error': error,
            'timing': {}
        }
        
        lines = output.split('\n')
        
        # 提取时间信息
        for line in lines:
            if 'Find corners took:' in line:
                try:
                    time_ms = float(line.split(':')[1].strip().split()[0])
                    info['timing']['find_corners_ms'] = time_ms
                except:
                    pass
            elif 'Find boards took:' in line:
                try:
                    time_ms = float(line.split(':')[1].strip().split()[0])
                    info['timing']['find_boards_ms'] = time_ms
                except:
                    pass
            elif 'Total took:' in line:
                try:
                    time_ms = float(line.split(':')[1].strip().split()[0])
                    info['timing']['total_ms'] = time_ms
                except:
                    pass
        
        return info
    
    def test_example_data(self) -> Dict:
        """
        测试example_data目录中的图像
        
        Returns:
            测试结果
        """
        results = {}
        
        if not self.example_data_dir.exists():
            return {'error': f'example_data目录不存在: {self.example_data_dir}'}
        
        # 支持的图像格式
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
        
        for image_file in self.example_data_dir.iterdir():
            if image_file.suffix.lower() in image_extensions:
                print(f"\n测试图像: {image_file.name}")
                result = self.detect_checkerboard(str(image_file))
                results[image_file.name] = result
        
        return results
    
    def compare_with_our_cpp(self, image_path: str) -> Dict:
        """
        与我们的C++程序比较
        
        Args:
            image_path: 图像路径
            
        Returns:
            比较结果
        """
        # 使用libcbdetCpp检测
        libcbdet_result = self.detect_checkerboard(image_path)
        
        # 使用我们的C++程序检测
        from cpp_detector_wrapper import CppDetectorWrapper
        our_cpp = CppDetectorWrapper()
        our_result = our_cpp.detect_checkerboard(image_path, "demo")
        
        return {
            'libcbdetCpp': libcbdet_result,
            'our_cpp': our_result
        }


def main():
    """主函数 - 演示用法"""
    try:
        wrapper = LibCBDetCppWrapper()
        print("libcbdetCpp包装器初始化成功")
        
        # 测试example_data
        print("\n=== 测试example_data目录 ===")
        results = wrapper.test_example_data()
        
        if 'error' in results:
            print(f"错误: {results['error']}")
        else:
            print(f"测试了 {len(results)} 个图像")
            for image_name, result in results.items():
                if 'error' in result:
                    print(f"  {image_name}: 错误 - {result['error']}")
                else:
                    timing = result.get('timing', {})
                    print(f"  {image_name}: 角点检测 {timing.get('find_corners_ms', 0):.1f}ms, "
                          f"棋盘检测 {timing.get('find_boards_ms', 0):.1f}ms")
        
        # 与我们的C++程序比较
        print("\n=== 与我们的C++程序比较 ===")
        test_image = "data/04.png"
        if os.path.exists(test_image):
            comparison = wrapper.compare_with_our_cpp(test_image)
            
            print("比较结果:")
            for program, result in comparison.items():
                if 'error' in result:
                    print(f"  {program}: 错误 - {result['error']}")
                else:
                    timing = result.get('timing', {})
                    print(f"  {program}: 总时间 {timing.get('total_ms', 0):.1f}ms")
        
    except Exception as e:
        print(f"初始化失败: {e}")


if __name__ == "__main__":
    main() 