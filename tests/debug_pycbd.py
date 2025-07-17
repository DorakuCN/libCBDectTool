#!/usr/bin/env python3
"""
PyCBD调试脚本
使用项目中的测试图像进行棋盘检测测试
"""

import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from pathlib import Path

# 添加PyCBD到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '3rdparty', 'pyCBD', 'src'))

try:
    from PyCBD.pipelines import CBDPipeline
    from PyCBD.checkerboard_detection.checkerboard_detector import CheckerboardDetector
    from PyCBD.checkerboard_enhancement.checkerboard_enhancer import CheckerboardEnhancer
    from PyCBD.pipelines import prepare_image
    print("✅ PyCBD导入成功")
except ImportError as e:
    print(f"❌ PyCBD导入失败: {e}")
    print("请确保已安装PyCBD依赖: pip install -r 3rdparty/pyCBD/requirements.txt")
    sys.exit(1)

def test_pycbd_detection(image_path, checkerboard_size=None):
    """
    测试PyCBD棋盘检测
    
    Args:
        image_path: 图像路径
        checkerboard_size: 棋盘尺寸 (rows, cols)，可选
    """
    print(f"\n🔍 测试图像: {image_path}")
    
    # 检查图像文件是否存在
    if not os.path.exists(image_path):
        print(f"❌ 图像文件不存在: {image_path}")
        return None
    
    # 读取图像
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ 无法读取图像: {image_path}")
            return None
        
        print(f"📊 图像尺寸: {image.shape}")
        print(f"📊 图像类型: {image.dtype}")
        
    except Exception as e:
        print(f"❌ 读取图像时出错: {e}")
        return None
    
    # 创建检测器
    try:
        print("\n🚀 创建PyCBD检测器...")
        detector = CBDPipeline(expand=True, predict=True)
        print("✅ 检测器创建成功")
    except Exception as e:
        print(f"❌ 创建检测器失败: {e}")
        return None
    
    # 执行检测
    try:
        print(f"\n🔍 开始检测棋盘...")
        if checkerboard_size:
            print(f"📏 指定棋盘尺寸: {checkerboard_size}")
            result, board_uv, board_xy = detector.detect_checkerboard(image, checkerboard_size)
        else:
            result, board_uv, board_xy = detector.detect_checkerboard(image)
        
        print(f"✅ 检测完成")
        print(f"📊 检测结果: {result}")
        print(f"📊 棋盘UV坐标数量: {len(board_uv) if board_uv is not None else 0}")
        print(f"📊 棋盘XY坐标数量: {len(board_xy) if board_xy is not None else 0}")
        
        if board_uv is not None and len(board_uv) > 0:
            print(f"📊 第一个角点UV: {board_uv[0]}")
            print(f"📊 最后一个角点UV: {board_uv[-1]}")
            print(f"📊 第一个角点XY: {board_xy[0]}")
            print(f"📊 最后一个角点XY: {board_xy[-1]}")
        
        return result, board_uv, board_xy, image
        
    except Exception as e:
        print(f"❌ 检测过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return None

def visualize_results(image, board_uv, board_xy, title="PyCBD检测结果"):
    """
    可视化检测结果
    """
    if board_uv is None or len(board_uv) == 0:
        print("❌ 没有检测到棋盘角点，无法可视化")
        return
    
    try:
        # 转换图像颜色空间用于显示
        if len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # 创建图像
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(image_rgb)
        
        # 绘制角点
        ax.plot(board_uv[:, 0], board_uv[:, 1], 'r-o', markeredgecolor='k', markersize=6, linewidth=2)
        
        # 添加坐标标签
        if len(board_uv) > 0:
            trans_offset = mtransforms.offset_copy(ax.transData, fig=fig, x=-0.4, y=-0.20, units='inches')
            ax.text(board_uv[0, 0], board_uv[0, 1], 
                   f'({int(board_xy[0, 0])}, {int(board_xy[0, 1])})',
                   color="red", transform=trans_offset, fontsize=10, weight='bold')
            
            trans_offset = mtransforms.offset_copy(ax.transData, fig=fig, x=0.05, y=0.05, units='inches')
            ax.text(board_uv[-1, 0], board_uv[-1, 1], 
                   f'({int(board_xy[-1, 0])}, {int(board_xy[-1, 1])})',
                   color="red", transform=trans_offset, fontsize=10, weight='bold')
        
        ax.set_title(title, fontsize=14, weight='bold')
        ax.axis('off')
        
        # 保存结果
        output_path = f"result/pycbd_{Path(title).stem}.png"
        os.makedirs("result", exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"💾 结果已保存到: {output_path}")
        
        plt.show()
        
    except Exception as e:
        print(f"❌ 可视化过程中出错: {e}")
        import traceback
        traceback.print_exc()

def test_dewarping(detector, image, board_uv, board_xy):
    """
    测试图像去扭曲
    """
    if board_uv is None or len(board_uv) == 0:
        print("❌ 没有检测到棋盘，无法进行去扭曲")
        return
    
    try:
        print("\n🔄 开始图像去扭曲...")
        dewarped = detector.dewarp_image(image, board_uv, board_xy)
        
        if dewarped is not None:
            print("✅ 去扭曲完成")
            
            # 保存去扭曲结果
            output_path = "result/pycbd_dewarped.png"
            os.makedirs("result", exist_ok=True)
            cv2.imwrite(output_path, dewarped)
            print(f"💾 去扭曲结果已保存到: {output_path}")
            
            # 显示去扭曲结果
            plt.figure(figsize=(12, 8))
            if len(dewarped.shape) == 3:
                plt.imshow(cv2.cvtColor(dewarped, cv2.COLOR_BGR2RGB))
            else:
                plt.imshow(dewarped, cmap='gray')
            plt.title("PyCBD去扭曲结果", fontsize=14, weight='bold')
            plt.axis('off')
            plt.show()
        else:
            print("❌ 去扭曲失败")
            
    except Exception as e:
        print(f"❌ 去扭曲过程中出错: {e}")
        import traceback
        traceback.print_exc()

def main():
    """
    主函数
    """
    print("🎯 PyCBD调试脚本")
    print("=" * 50)
    
    # 测试图像列表
    test_images = [
        ("data/04.png", (9, 14)),  # 标准棋盘
        ("data/00.png", None),     # 其他测试图像
        ("data/01.png", None),
        ("data/02.png", None),
        ("data/03.png", None),
        ("data/05.png", None),
    ]
    
    successful_detections = 0
    total_tests = len(test_images)
    
    for image_path, checkerboard_size in test_images:
        print(f"\n{'='*60}")
        print(f"🧪 测试 {test_images.index((image_path, checkerboard_size)) + 1}/{total_tests}")
        
        # 执行检测
        result = test_pycbd_detection(image_path, checkerboard_size)
        
        if result is not None:
            result_flag, board_uv, board_xy, image = result
            
            if result_flag and board_uv is not None and len(board_uv) > 0:
                print(f"✅ 检测成功! 找到 {len(board_uv)} 个角点")
                successful_detections += 1
                
                # 可视化结果
                title = f"PyCBD检测结果 - {os.path.basename(image_path)}"
                visualize_results(image, board_uv, board_xy, title)
                
                # 测试去扭曲（仅对第一个成功检测的图像）
                if successful_detections == 1:
                    detector = CBDPipeline(expand=True, predict=True)
                    test_dewarping(detector, image, board_uv, board_xy)
            else:
                print(f"❌ 检测失败或未找到棋盘")
        else:
            print(f"❌ 检测过程出错")
    
    # 总结
    print(f"\n{'='*60}")
    print(f"📊 测试总结:")
    print(f"   总测试数: {total_tests}")
    print(f"   成功检测: {successful_detections}")
    print(f"   成功率: {successful_detections/total_tests*100:.1f}%")
    
    if successful_detections > 0:
        print(f"\n✅ PyCBD调试完成，检测到 {successful_detections} 个棋盘")
    else:
        print(f"\n❌ PyCBD调试失败，未检测到任何棋盘")

if __name__ == "__main__":
    main() 