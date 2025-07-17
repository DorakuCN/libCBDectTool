#!/usr/bin/env python3
"""
测试PyCBD兼容的检测器
"""

import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pycbd_compatible_detector import CheckerboardDetector, CBDPipeline


def test_basic_detection():
    """测试基本检测功能"""
    print("=== 测试基本检测功能 ===")
    
    # 创建检测器
    detector = CheckerboardDetector()
    
    # 测试图像
    test_image = "data/04.png"
    if not os.path.exists(test_image):
        print(f"测试图像不存在: {test_image}")
        return
    
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
    
    return board_uv, board_xy, corners_uv


def test_pipeline():
    """测试管道功能"""
    print("\n=== 测试管道功能 ===")
    
    # 创建管道
    pipeline = CBDPipeline()
    
    # 测试图像
    test_image = "data/04.png"
    if not os.path.exists(test_image):
        print(f"测试图像不存在: {test_image}")
        return
    
    # 读取图像
    image = cv2.imread(test_image)
    
    # 检测棋盘
    result, board_uv, board_xy = pipeline.detect_checkerboard(image)
    
    print(f"管道检测结果:")
    print(f"  结果状态: {result} (0=失败, 1=相对坐标, 2=绝对坐标)")
    print(f"  棋盘角点数量: {len(board_uv)}")
    
    return result, board_uv, board_xy


def test_different_programs():
    """测试不同的C++程序"""
    print("\n=== 测试不同的C++程序 ===")
    
    # 可用的程序
    programs = ["demo", "pipeline_demo", "optimized_debug"]
    
    test_image = "data/04.png"
    if not os.path.exists(test_image):
        print(f"测试图像不存在: {test_image}")
        return
    
    image = cv2.imread(test_image)
    
    results = {}
    
    for program in programs:
        try:
            print(f"\n测试程序: {program}")
            detector = CheckerboardDetector(program=program)
            board_uv, board_xy, corners_uv = detector.detect_checkerboard(image)
            
            results[program] = {
                'board_corners': len(board_uv),
                'total_corners': len(corners_uv),
                'success': len(board_uv) > 0
            }
            
            print(f"  棋盘角点: {len(board_uv)}")
            print(f"  总角点: {len(corners_uv)}")
            
        except Exception as e:
            print(f"  错误: {e}")
            results[program] = {'error': str(e)}
    
    # 比较结果
    print(f"\n程序比较结果:")
    print("-" * 50)
    print(f"{'程序':<20} {'棋盘角点':<10} {'总角点':<10} {'状态':<10}")
    print("-" * 50)
    
    for program, result in results.items():
        if 'error' in result:
            status = "错误"
            board_corners = "-"
            total_corners = "-"
        else:
            status = "成功" if result['success'] else "失败"
            board_corners = str(result['board_corners'])
            total_corners = str(result['total_corners'])
        
        print(f"{program:<20} {board_corners:<10} {total_corners:<10} {status:<10}")


def test_visualization():
    """测试可视化功能"""
    print("\n=== 测试可视化功能 ===")
    
    test_image = "data/04.png"
    if not os.path.exists(test_image):
        print(f"测试图像不存在: {test_image}")
        return
    
    # 读取图像
    image = cv2.imread(test_image)
    
    # 检测棋盘
    detector = CheckerboardDetector()
    board_uv, board_xy, corners_uv = detector.detect_checkerboard(image)
    
    # 创建可视化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 原始图像
    ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax1.set_title("原始图像")
    ax1.axis('off')
    
    # 检测结果
    ax2.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # 绘制所有角点
    if len(corners_uv) > 0:
        ax2.plot(corners_uv[:, 0], corners_uv[:, 1], 'bo', markersize=4, label='所有角点')
    
    # 绘制棋盘角点
    if len(board_uv) > 0:
        ax2.plot(board_uv[:, 0], board_uv[:, 1], 'ro', markersize=6, label='棋盘角点')
        
        # 连接棋盘角点
        if len(board_uv) > 1:
            ax2.plot(board_uv[:, 0], board_uv[:, 1], 'r-', linewidth=2)
    
    ax2.set_title(f"检测结果 (棋盘: {len(board_uv)}, 总角点: {len(corners_uv)})")
    ax2.legend()
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig('detection_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("可视化结果已保存到: detection_visualization.png")


def test_batch_processing():
    """测试批量处理"""
    print("\n=== 测试批量处理 ===")
    
    data_dir = "data"
    if not os.path.exists(data_dir):
        print(f"数据目录不存在: {data_dir}")
        return
    
    # 支持的图像格式
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    
    # 获取所有图像文件
    image_files = []
    for file in os.listdir(data_dir):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(os.path.join(data_dir, file))
    
    print(f"找到 {len(image_files)} 个图像文件")
    
    # 创建检测器
    detector = CheckerboardDetector()
    
    results = {}
    
    for image_file in image_files:
        print(f"\n处理: {os.path.basename(image_file)}")
        
        try:
            image = cv2.imread(image_file)
            board_uv, board_xy, corners_uv = detector.detect_checkerboard(image)
            
            results[os.path.basename(image_file)] = {
                'board_corners': len(board_uv),
                'total_corners': len(corners_uv),
                'success': len(board_uv) > 0
            }
            
            print(f"  棋盘角点: {len(board_uv)}")
            print(f"  总角点: {len(corners_uv)}")
            
        except Exception as e:
            print(f"  错误: {e}")
            results[os.path.basename(image_file)] = {'error': str(e)}
    
    # 总结结果
    print(f"\n批量处理总结:")
    print("-" * 50)
    print(f"{'图像':<20} {'棋盘角点':<10} {'总角点':<10} {'状态':<10}")
    print("-" * 50)
    
    success_count = 0
    for image_name, result in results.items():
        if 'error' in result:
            status = "错误"
            board_corners = "-"
            total_corners = "-"
        else:
            status = "成功" if result['success'] else "失败"
            board_corners = str(result['board_corners'])
            total_corners = str(result['total_corners'])
            if result['success']:
                success_count += 1
        
        print(f"{image_name:<20} {board_corners:<10} {total_corners:<10} {status:<10}")
    
    print(f"\n成功率: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")


def main():
    """主函数"""
    print("PyCBD兼容检测器测试")
    print("=" * 50)
    
    # 检查build目录
    if not os.path.exists("build"):
        print("错误: build目录不存在，请先编译C++程序")
        return
    
    # 运行各种测试
    test_basic_detection()
    test_pipeline()
    test_different_programs()
    test_visualization()
    test_batch_processing()
    
    print("\n测试完成!")


if __name__ == "__main__":
    main() 