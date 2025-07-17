#!/usr/bin/env python3
"""
测试libcbdet包装器
"""

import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(__file__))

def test_wrapper_import():
    """
    测试包装器导入
    """
    print("🔍 测试包装器导入...")
    
    try:
        from libcbdet_wrapper import libcbdet, Checkerboard
        print("✅ libcbdet_wrapper 导入成功")
        return True
    except ImportError as e:
        print(f"❌ libcbdet_wrapper 导入失败: {e}")
        return False

def test_library_loading():
    """
    测试库加载
    """
    print("\n🔍 测试库加载...")
    
    try:
        from libcbdet_wrapper import libcbdet
        
        if libcbdet.lib is not None:
            print("✅ 库加载成功")
            return True
        else:
            print("❌ 库加载失败")
            return False
    except Exception as e:
        print(f"❌ 库加载出错: {e}")
        return False

def test_checkerboard_class():
    """
    测试Checkerboard类
    """
    print("\n🔍 测试Checkerboard类...")
    
    try:
        from libcbdet_wrapper import Checkerboard
        
        checkerboard = Checkerboard()
        print("✅ Checkerboard类创建成功")
        
        # 测试属性
        print(f"📊 rows: {checkerboard.rows}")
        print(f"📊 cols: {checkerboard.cols}")
        print(f"📊 number_of_corners: {checkerboard.number_of_corners}")
        
        return True
    except Exception as e:
        print(f"❌ Checkerboard类测试失败: {e}")
        return False

def test_image_processing():
    """
    测试图像处理
    """
    print("\n🔍 测试图像处理...")
    
    test_image = "data/04.png"
    if not os.path.exists(test_image):
        print(f"❌ 测试图像不存在: {test_image}")
        return False
    
    try:
        image = cv2.imread(test_image)
        print(f"✅ 图像读取成功: {image.shape}")
        
        # 测试包装器的检测功能
        from libcbdet_wrapper import libcbdet
        
        success, board_uv, corners_uv = libcbdet.detect_checkerboard(image)
        
        print(f"📊 检测结果: {success}")
        print(f"📊 棋盘角点数量: {len(board_uv)}")
        print(f"📊 总角点数量: {len(corners_uv)}")
        
        if success and len(board_uv) > 0:
            print(f"📊 第一个棋盘角点: {board_uv[0]}")
            print(f"📊 最后一个棋盘角点: {board_uv[-1]}")
        
        return success
        
    except Exception as e:
        print(f"❌ 图像处理失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pycbd_integration():
    """
    测试PyCBD集成
    """
    print("\n🔍 测试PyCBD集成...")
    
    try:
        # 添加PyCBD路径
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '3rdparty', 'pyCBD', 'src'))
        
        # 尝试导入PyCBD
        from PyCBD.checkerboard_detection.checkerboard_detector import CheckerboardDetector
        print("✅ PyCBD导入成功")
        
        # 创建检测器
        detector = CheckerboardDetector()
        print("✅ PyCBD检测器创建成功")
        
        return True
        
    except ImportError as e:
        print(f"❌ PyCBD导入失败: {e}")
        return False
    except Exception as e:
        print(f"❌ PyCBD集成测试失败: {e}")
        return False

def visualize_results(image, board_uv, corners_uv, title="检测结果"):
    """
    可视化检测结果
    """
    if len(board_uv) == 0:
        print("❌ 没有检测到棋盘角点，无法可视化")
        return
    
    try:
        if len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(image_rgb)
        
        # 绘制棋盘角点
        if len(board_uv) > 0:
            ax.plot(board_uv[:, 0], board_uv[:, 1], 'ro', markersize=8, label='棋盘角点')
        
        # 绘制所有角点
        if len(corners_uv) > 0:
            ax.plot(corners_uv[:, 0], corners_uv[:, 1], 'b.', markersize=4, alpha=0.5, label='所有角点')
        
        ax.set_title(title, fontsize=14, weight='bold')
        ax.legend()
        ax.axis('off')
        
        output_path = "result/libcbdet_wrapper_test.png"
        os.makedirs("result", exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"💾 结果已保存到: {output_path}")
        
        plt.show()
        
    except Exception as e:
        print(f"❌ 可视化过程中出错: {e}")

def main():
    """
    主函数
    """
    print("🎯 libcbdet包装器测试")
    print("=" * 50)
    
    # 测试包装器导入
    wrapper_ok = test_wrapper_import()
    
    # 测试库加载
    library_ok = test_library_loading()
    
    # 测试Checkerboard类
    checkerboard_ok = test_checkerboard_class()
    
    # 测试图像处理
    image_ok = test_image_processing()
    
    # 测试PyCBD集成
    pycbd_ok = test_pycbd_integration()
    
    # 如果图像处理成功，进行可视化
    if image_ok:
        test_image = "data/04.png"
        image = cv2.imread(test_image)
        from libcbdet_wrapper import libcbdet
        success, board_uv, corners_uv = libcbdet.detect_checkerboard(image)
        
        if success:
            visualize_results(image, board_uv, corners_uv, "libcbdet包装器检测结果")
    
    print(f"\n{'='*60}")
    print(f"📊 测试总结:")
    print(f"   包装器导入: {'✅' if wrapper_ok else '❌'}")
    print(f"   库加载: {'✅' if library_ok else '❌'}")
    print(f"   Checkerboard类: {'✅' if checkerboard_ok else '❌'}")
    print(f"   图像处理: {'✅' if image_ok else '❌'}")
    print(f"   PyCBD集成: {'✅' if pycbd_ok else '❌'}")
    
    if wrapper_ok and library_ok:
        print(f"\n✅ libcbdet包装器基本功能正常")
    else:
        print(f"\n❌ libcbdet包装器存在问题，需要进一步调试")

if __name__ == "__main__":
    main() 