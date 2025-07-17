#!/usr/bin/env python3
"""
测试libcbdet包装器 v2
"""

import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(__file__))

def test_wrapper_v2():
    """
    测试v2包装器
    """
    print("🔍 测试v2包装器...")
    
    try:
        from libcbdet_wrapper_v2 import libcbdet_v2, CheckerboardV2
        print("✅ libcbdet_wrapper_v2 导入成功")
        return True
    except ImportError as e:
        print(f"❌ libcbdet_wrapper_v2 导入失败: {e}")
        return False

def test_cpp_demo_integration():
    """
    测试C++ demo集成
    """
    print("\n🔍 测试C++ demo集成...")
    
    test_image = "data/04.png"
    if not os.path.exists(test_image):
        print(f"❌ 测试图像不存在: {test_image}")
        return False
    
    try:
        image = cv2.imread(test_image)
        print(f"✅ 图像读取成功: {image.shape}")
        
        # 测试包装器的检测功能
        from libcbdet_wrapper_v2 import libcbdet_v2
        
        success, board_uv, corners_uv = libcbdet_v2.detect_checkerboard(image)
        
        print(f"📊 检测结果: {success}")
        print(f"📊 棋盘角点数量: {len(board_uv)}")
        print(f"📊 总角点数量: {len(corners_uv)}")
        
        if success and len(board_uv) > 0:
            print(f"📊 第一个棋盘角点: {board_uv[0]}")
            print(f"📊 最后一个棋盘角点: {board_uv[-1]}")
            
            # 可视化结果
            visualize_results(image, board_uv, corners_uv, "v2包装器检测结果")
        
        return success
        
    except Exception as e:
        print(f"❌ C++ demo集成测试失败: {e}")
        import traceback
        traceback.print_exc()
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
        
        output_path = "result/libcbdet_wrapper_v2_test.png"
        os.makedirs("result", exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"💾 结果已保存到: {output_path}")
        
        plt.show()
        
    except Exception as e:
        print(f"❌ 可视化过程中出错: {e}")

def test_pycbd_integration_v2():
    """
    测试PyCBD集成 v2
    """
    print("\n🔍 测试PyCBD集成 v2...")
    
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

def main():
    """
    主函数
    """
    print("🎯 libcbdet包装器 v2 测试")
    print("=" * 50)
    
    # 测试包装器导入
    wrapper_ok = test_wrapper_v2()
    
    # 测试C++ demo集成
    demo_ok = test_cpp_demo_integration()
    
    # 测试PyCBD集成
    pycbd_ok = test_pycbd_integration_v2()
    
    print(f"\n{'='*60}")
    print(f"📊 测试总结:")
    print(f"   包装器导入: {'✅' if wrapper_ok else '❌'}")
    print(f"   C++ demo集成: {'✅' if demo_ok else '❌'}")
    print(f"   PyCBD集成: {'✅' if pycbd_ok else '❌'}")
    
    if wrapper_ok and demo_ok:
        print(f"\n✅ libcbdet包装器 v2 基本功能正常")
        print(f"💡 可以通过调用C++ demo程序进行棋盘检测")
    else:
        print(f"\n❌ libcbdet包装器 v2 存在问题，需要进一步调试")

if __name__ == "__main__":
    main() 