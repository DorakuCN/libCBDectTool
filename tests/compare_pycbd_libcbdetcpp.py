#!/usr/bin/env python3
"""
PyCBD和libcbdetCpp处理结果对比分析
"""

import os
import sys
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any, List
import tempfile
import subprocess
import json
from pathlib import Path

# 导入我们的PyCBD配置
from pycbd_example_config import CheckerboardDetector as OurDetector, CBDPipeline as OurPipeline


class ResultComparison:
    """结果对比分析类"""
    
    def __init__(self):
        self.results = {}
        self.comparison_data = {}
        
    def add_result(self, method: str, image_name: str, result: Dict[str, Any]):
        """添加检测结果"""
        if method not in self.results:
            self.results[method] = {}
        self.results[method][image_name] = result
    
    def compare_results(self, image_name: str) -> Dict[str, Any]:
        """对比两种方法的结果"""
        if 'libcbdetCpp' not in self.results or 'PyCBD' not in self.results:
            return {}
        
        libcbdet_result = self.results['libcbdetCpp'].get(image_name, {})
        pycbd_result = self.results['PyCBD'].get(image_name, {})
        
        comparison = {
            'image_name': image_name,
            'libcbdet_corners': libcbdet_result.get('corners_detected', 0),
            'pycbd_corners': pycbd_result.get('corners_detected', 0),
            'libcbdet_boards': libcbdet_result.get('boards_detected', 0),
            'pycbd_boards': pycbd_result.get('boards_detected', 0),
            'libcbdet_time': libcbdet_result.get('execution_time', 0),
            'pycbd_time': pycbd_result.get('execution_time', 0),
            'corner_difference': abs(libcbdet_result.get('corners_detected', 0) - pycbd_result.get('corners_detected', 0)),
            'board_difference': abs(libcbdet_result.get('boards_detected', 0) - pycbd_result.get('boards_detected', 0)),
            'time_ratio': pycbd_result.get('execution_time', 1) / max(libcbdet_result.get('execution_time', 1), 0.001),
            'success_rate_libcbdet': 1 if libcbdet_result.get('success', False) else 0,
            'success_rate_pycbd': 1 if pycbd_result.get('success', False) else 0
        }
        
        return comparison
    
    def generate_summary(self) -> Dict[str, Any]:
        """生成对比总结"""
        if not self.results:
            return {}
        
        comparisons = []
        for image_name in set().union(*[set(method_results.keys()) for method_results in self.results.values()]):
            comparison = self.compare_results(image_name)
            if comparison:
                comparisons.append(comparison)
        
        if not comparisons:
            return {}
        
        # 计算统计信息
        summary = {
            'total_images': len(comparisons),
            'libcbdet_success_rate': np.mean([c['success_rate_libcbdet'] for c in comparisons]),
            'pycbd_success_rate': np.mean([c['success_rate_pycbd'] for c in comparisons]),
            'avg_corner_difference': np.mean([c['corner_difference'] for c in comparisons]),
            'avg_board_difference': np.mean([c['board_difference'] for c in comparisons]),
            'avg_time_ratio': np.mean([c['time_ratio'] for c in comparisons]),
            'libcbdet_avg_time': np.mean([c['libcbdet_time'] for c in comparisons]),
            'pycbd_avg_time': np.mean([c['pycbd_time'] for c in comparisons]),
            'libcbdet_avg_corners': np.mean([c['libcbdet_corners'] for c in comparisons]),
            'pycbd_avg_corners': np.mean([c['pycbd_corners'] for c in comparisons]),
            'libcbdet_avg_boards': np.mean([c['libcbdet_boards'] for c in comparisons]),
            'pycbd_avg_boards': np.mean([c['pycbd_boards'] for c in comparisons])
        }
        
        return summary


def test_libcbdetCpp(image_path: str) -> Dict[str, Any]:
    """测试libcbdetCpp"""
    start_time = time.time()
    
    try:
        # 运行修复版本的example程序
        result = subprocess.run(
            ["3rdparty/libcbdetCpp/bin/example_fixed", image_path],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        execution_time = time.time() - start_time
        
        if result.returncode == 0:
            # 解析输出
            output = result.stdout
            corners_detected = 0
            boards_detected = 0
            
            for line in output.split('\n'):
                if 'Detected' in line and 'corners' in line and 'boards' in line:
                    try:
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part == 'corners':
                                corners_detected = int(parts[i-1])
                            elif part == 'boards':
                                boards_detected = int(parts[i-1])
                    except:
                        pass
            
            return {
                'success': True,
                'corners_detected': corners_detected,
                'boards_detected': boards_detected,
                'execution_time': execution_time,
                'output': output
            }
        else:
            return {
                'success': False,
                'corners_detected': 0,
                'boards_detected': 0,
                'execution_time': execution_time,
                'error': result.stderr
            }
            
    except Exception as e:
        return {
            'success': False,
            'corners_detected': 0,
            'boards_detected': 0,
            'execution_time': time.time() - start_time,
            'error': str(e)
        }


def test_pycbd(image_path: str) -> Dict[str, Any]:
    """测试PyCBD（我们的配置）"""
    start_time = time.time()
    
    try:
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            return {
                'success': False,
                'corners_detected': 0,
                'boards_detected': 0,
                'execution_time': time.time() - start_time,
                'error': f"无法读取图像: {image_path}"
            }
        
        # 使用我们的PyCBD配置
        detector = OurDetector()
        board_uv, board_xy, corners_uv = detector.detect_checkerboard(image)
        
        execution_time = time.time() - start_time
        
        return {
            'success': len(board_uv) > 0,
            'corners_detected': len(corners_uv),
            'boards_detected': 1 if len(board_uv) > 0 else 0,
            'execution_time': execution_time,
            'board_uv': board_uv,
            'board_xy': board_xy,
            'corners_uv': corners_uv
        }
        
    except Exception as e:
        return {
            'success': False,
            'corners_detected': 0,
            'boards_detected': 0,
            'execution_time': time.time() - start_time,
            'error': str(e)
        }


def test_pycbd_enhanced(image_path: str) -> Dict[str, Any]:
    """测试PyCBD增强功能"""
    start_time = time.time()
    
    try:
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            return {
                'success': False,
                'corners_detected': 0,
                'boards_detected': 0,
                'execution_time': time.time() - start_time,
                'error': f"无法读取图像: {image_path}"
            }
        
        # 使用增强的PyCBD管道
        pipeline = OurPipeline(expand=True, predict=True)
        result, board_uv, board_xy = pipeline.detect_checkerboard(image)
        
        execution_time = time.time() - start_time
        
        return {
            'success': result > 0,
            'corners_detected': len(board_uv),
            'boards_detected': 1 if len(board_uv) > 0 else 0,
            'execution_time': execution_time,
            'result_code': result,
            'board_uv': board_uv,
            'board_xy': board_xy
        }
        
    except Exception as e:
        return {
            'success': False,
            'corners_detected': 0,
            'boards_detected': 0,
            'execution_time': time.time() - start_time,
            'error': str(e)
        }


def run_comparison_tests():
    """运行对比测试"""
    print("PyCBD和libcbdetCpp处理结果对比分析")
    print("=" * 60)
    
    # 初始化对比器
    comparison = ResultComparison()
    
    # 测试图像列表
    test_images = [
        "data/04.png",
        "3rdparty/libcbdetCpp/example_data/04.png",
        "3rdparty/libcbdetCpp/example_data/e1.png",
        "3rdparty/libcbdetCpp/example_data/e2.png",
        "3rdparty/libcbdetCpp/example_data/e3.png",
        "3rdparty/libcbdetCpp/example_data/e4.png",
        "3rdparty/libcbdetCpp/example_data/e5.png",
        "3rdparty/libcbdetCpp/example_data/e6.png",
        "3rdparty/libcbdetCpp/example_data/e7.png"
    ]
    
    # 过滤存在的图像
    existing_images = [img for img in test_images if os.path.exists(img)]
    print(f"找到 {len(existing_images)} 个测试图像")
    
    for i, image_path in enumerate(existing_images):
        image_name = os.path.basename(image_path)
        print(f"\n[{i+1}/{len(existing_images)}] 测试图像: {image_name}")
        print("-" * 40)
        
        # 测试libcbdetCpp
        print("测试libcbdetCpp...")
        libcbdet_result = test_libcbdetCpp(image_path)
        comparison.add_result('libcbdetCpp', image_name, libcbdet_result)
        
        print(f"  角点数量: {libcbdet_result['corners_detected']}")
        print(f"  棋盘数量: {libcbdet_result['boards_detected']}")
        print(f"  执行时间: {libcbdet_result['execution_time']:.3f}s")
        print(f"  成功: {libcbdet_result['success']}")
        
        # 测试PyCBD基础版本
        print("测试PyCBD基础版本...")
        pycbd_result = test_pycbd(image_path)
        comparison.add_result('PyCBD', image_name, pycbd_result)
        
        print(f"  角点数量: {pycbd_result['corners_detected']}")
        print(f"  棋盘数量: {pycbd_result['boards_detected']}")
        print(f"  执行时间: {pycbd_result['execution_time']:.3f}s")
        print(f"  成功: {pycbd_result['success']}")
        
        # 测试PyCBD增强版本
        print("测试PyCBD增强版本...")
        pycbd_enhanced_result = test_pycbd_enhanced(image_path)
        comparison.add_result('PyCBD_Enhanced', image_name, pycbd_enhanced_result)
        
        print(f"  角点数量: {pycbd_enhanced_result['corners_detected']}")
        print(f"  棋盘数量: {pycbd_enhanced_result['boards_detected']}")
        print(f"  执行时间: {pycbd_enhanced_result['execution_time']:.3f}s")
        print(f"  成功: {pycbd_enhanced_result['success']}")
        
        # 对比结果
        comp = comparison.compare_results(image_name)
        if comp:
            print(f"\n对比结果:")
            print(f"  角点差异: {comp['corner_difference']}")
            print(f"  棋盘差异: {comp['board_difference']}")
            print(f"  时间比: {comp['time_ratio']:.2f}x")
    
    # 生成总结
    print("\n" + "=" * 60)
    print("总体对比总结")
    print("=" * 60)
    
    summary = comparison.generate_summary()
    if summary:
        print(f"测试图像总数: {summary['total_images']}")
        print(f"\n成功率:")
        print(f"  libcbdetCpp: {summary['libcbdet_success_rate']:.1%}")
        print(f"  PyCBD基础: {summary['pycbd_success_rate']:.1%}")
        
        print(f"\n平均检测结果:")
        print(f"  libcbdetCpp - 角点: {summary['libcbdet_avg_corners']:.1f}, 棋盘: {summary['libcbdet_avg_boards']:.1f}")
        print(f"  PyCBD基础 - 角点: {summary['pycbd_avg_corners']:.1f}, 棋盘: {summary['pycbd_avg_boards']:.1f}")
        
        print(f"\n平均执行时间:")
        print(f"  libcbdetCpp: {summary['libcbdet_avg_time']:.3f}s")
        print(f"  PyCBD基础: {summary['pycbd_avg_time']:.3f}s")
        print(f"  时间比: {summary['avg_time_ratio']:.2f}x")
        
        print(f"\n平均差异:")
        print(f"  角点差异: {summary['avg_corner_difference']:.1f}")
        print(f"  棋盘差异: {summary['avg_board_difference']:.1f}")
    
    # 保存详细结果
    save_detailed_results(comparison, summary)


def save_detailed_results(comparison: ResultComparison, summary: Dict[str, Any]):
    """保存详细结果到文件"""
    results_file = "comparison_results.json"
    
    # 准备保存的数据
    save_data = {
        'summary': summary,
        'detailed_results': comparison.results,
        'comparisons': {}
    }
    
    # 添加对比数据
    for image_name in set().union(*[set(method_results.keys()) for method_results in comparison.results.values()]):
        save_data['comparisons'][image_name] = comparison.compare_results(image_name)
    
    # 保存到文件
    with open(results_file, 'w', encoding='utf-8') as f:
        # 转换numpy数组为列表以便JSON序列化
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        save_data = convert_numpy(save_data)
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n详细结果已保存到: {results_file}")


def create_visualization():
    """创建可视化对比图表"""
    try:
        # 读取结果文件
        with open("comparison_results.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        summary = data['summary']
        comparisons = data['comparisons']
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('PyCBD vs libcbdetCpp 对比分析', fontsize=16)
        
        # 1. 成功率对比
        ax1 = axes[0, 0]
        methods = ['libcbdetCpp', 'PyCBD基础']
        success_rates = [summary['libcbdet_success_rate'], summary['pycbd_success_rate']]
        bars1 = ax1.bar(methods, success_rates, color=['blue', 'orange'])
        ax1.set_ylabel('成功率')
        ax1.set_title('检测成功率对比')
        ax1.set_ylim(0, 1)
        for bar, rate in zip(bars1, success_rates):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{rate:.1%}', ha='center', va='bottom')
        
        # 2. 平均角点数量对比
        ax2 = axes[0, 1]
        avg_corners = [summary['libcbdet_avg_corners'], summary['pycbd_avg_corners']]
        bars2 = ax2.bar(methods, avg_corners, color=['blue', 'orange'])
        ax2.set_ylabel('平均角点数量')
        ax2.set_title('平均检测角点数量')
        for bar, count in zip(bars2, avg_corners):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{count:.1f}', ha='center', va='bottom')
        
        # 3. 执行时间对比
        ax3 = axes[1, 0]
        avg_times = [summary['libcbdet_avg_time'], summary['pycbd_avg_time']]
        bars3 = ax3.bar(methods, avg_times, color=['blue', 'orange'])
        ax3.set_ylabel('平均执行时间 (秒)')
        ax3.set_title('平均执行时间对比')
        for bar, time_val in zip(bars3, avg_times):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                    f'{time_val:.3f}s', ha='center', va='bottom')
        
        # 4. 角点差异分布
        ax4 = axes[1, 1]
        corner_diffs = [comp['corner_difference'] for comp in comparisons.values()]
        ax4.hist(corner_diffs, bins=10, alpha=0.7, color='green', edgecolor='black')
        ax4.set_xlabel('角点数量差异')
        ax4.set_ylabel('图像数量')
        ax4.set_title('角点检测差异分布')
        ax4.axvline(np.mean(corner_diffs), color='red', linestyle='--', 
                   label=f'平均差异: {np.mean(corner_diffs):.1f}')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig('comparison_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("可视化图表已保存到: comparison_visualization.png")
        
    except Exception as e:
        print(f"创建可视化时出错: {e}")


def main():
    """主函数"""
    print("开始PyCBD和libcbdetCpp对比测试...")
    
    # 检查必要的文件
    if not os.path.exists("3rdparty/libcbdetCpp/bin/example_fixed"):
        print("错误: 找不到example_fixed程序，请先编译")
        return
    
    # 运行对比测试
    run_comparison_tests()
    
    # 创建可视化
    if os.path.exists("comparison_results.json"):
        create_visualization()


if __name__ == "__main__":
    main() 