#!/usr/bin/env python3
"""
测试C++检测器包装器
"""

import os
import json
from cpp_detector_wrapper import CppDetectorWrapper


def test_single_detection():
    """测试单个图像检测"""
    print("=== 测试单个图像检测 ===")
    
    wrapper = CppDetectorWrapper()
    
    # 列出可用程序
    print("可用的C++程序:")
    for prog in wrapper.list_available_programs():
        print(f"  - {prog}")
    
    # 测试图像
    test_image = "data/04.png"
    if not os.path.exists(test_image):
        print(f"测试图像不存在: {test_image}")
        return
    
    print(f"\n测试图像: {test_image}")
    
    # 使用不同程序检测
    programs_to_test = ["demo", "pipeline_demo", "optimized_debug"]
    
    for program in programs_to_test:
        if program in wrapper.list_available_programs():
            print(f"\n--- 使用 {program} 检测 ---")
            result = wrapper.detect_chessboard(test_image, program)
            
            if 'error' in result:
                print(f"错误: {result['error']}")
            else:
                print(f"角点数量: {result['corners_detected']}")
                print(f"棋盘数量: {result['chessboards_detected']}")
                print(f"执行时间: {result['execution_time']:.2f}秒")
                
                if result['corners']:
                    print(f"前5个角点坐标:")
                    for i, corner in enumerate(result['corners'][:5]):
                        print(f"  {i}: ({corner[0]:.1f}, {corner[1]:.1f})")


def test_program_comparison():
    """测试程序比较"""
    print("\n=== 测试程序比较 ===")
    
    wrapper = CppDetectorWrapper()
    test_image = "data/04.png"
    
    if not os.path.exists(test_image):
        print(f"测试图像不存在: {test_image}")
        return
    
    # 比较多个程序
    comparison = wrapper.compare_programs(test_image, ["demo", "pipeline_demo", "optimized_debug"])
    
    print("\n程序比较结果:")
    print("-" * 60)
    print(f"{'程序':<20} {'角点':<8} {'棋盘':<8} {'时间(s)':<10} {'状态':<10}")
    print("-" * 60)
    
    for program, result in comparison.items():
        if 'error' in result:
            status = "错误"
            corners = "-"
            boards = "-"
            time_taken = "-"
        else:
            status = "成功"
            corners = str(result['corners_detected'])
            boards = str(result['chessboards_detected'])
            time_taken = f"{result['execution_time']:.2f}"
        
        print(f"{program:<20} {corners:<8} {boards:<8} {time_taken:<10} {status:<10}")


def test_batch_detection():
    """测试批量检测"""
    print("\n=== 测试批量检测 ===")
    
    wrapper = CppDetectorWrapper()
    data_dir = "data"
    
    if not os.path.exists(data_dir):
        print(f"数据目录不存在: {data_dir}")
        return
    
    # 批量检测
    results = wrapper.batch_detect(data_dir, "demo")
    
    print(f"\n批量检测结果 (共{len(results)}个图像):")
    print("-" * 50)
    print(f"{'图像':<15} {'角点':<8} {'棋盘':<8} {'状态':<10}")
    print("-" * 50)
    
    for image_name, result in results.items():
        if 'error' in result:
            status = "错误"
            corners = "-"
            boards = "-"
        else:
            status = "成功"
            corners = str(result['corners_detected'])
            boards = str(result['chessboards_detected'])
        
        print(f"{image_name:<15} {corners:<8} {boards:<8} {status:<10}")


def test_detailed_analysis():
    """测试详细分析"""
    print("\n=== 测试详细分析 ===")
    
    wrapper = CppDetectorWrapper()
    test_image = "data/04.png"
    
    if not os.path.exists(test_image):
        print(f"测试图像不存在: {test_image}")
        return
    
    # 使用optimized_debug进行详细分析
    if "optimized_debug" in wrapper.list_available_programs():
        print("使用 optimized_debug 进行详细分析...")
        result = wrapper.detect_chessboard(test_image, "optimized_debug")
        
        if 'error' not in result:
            print(f"\n详细结果:")
            print(f"程序: {result['program']}")
            print(f"角点数量: {result['corners_detected']}")
            print(f"棋盘数量: {result['chessboards_detected']}")
            print(f"执行时间: {result['execution_time']:.2f}秒")
            
            if result['corners']:
                print(f"\n角点坐标 (前10个):")
                for i, corner in enumerate(result['corners'][:10]):
                    print(f"  {i:2d}: ({corner[0]:6.1f}, {corner[1]:6.1f})")
            
            # 保存详细结果到文件
            output_file = "detailed_detection_result.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\n详细结果已保存到: {output_file}")


def main():
    """主函数"""
    print("C++检测器包装器测试")
    print("=" * 50)
    
    # 检查build目录
    if not os.path.exists("build"):
        print("错误: build目录不存在，请先编译C++程序")
        return
    
    # 运行各种测试
    test_single_detection()
    test_program_comparison()
    test_batch_detection()
    test_detailed_analysis()
    
    print("\n测试完成!")


if __name__ == "__main__":
    main() 