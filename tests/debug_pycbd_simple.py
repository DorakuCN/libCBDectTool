#!/usr/bin/env python3
"""
PyCBD简化调试脚本
检查源码结构和基本功能
"""

import sys
import os
from pathlib import Path

# 添加PyCBD到Python路径
pycbd_path = os.path.join(os.path.dirname(__file__), '3rdparty', 'pyCBD', 'src')
sys.path.insert(0, pycbd_path)

def check_pycbd_structure():
    """
    检查PyCBD源码结构
    """
    print("🔍 检查PyCBD源码结构...")
    
    # 检查主要目录
    directories = [
        '3rdparty/pyCBD/src/PyCBD',
        '3rdparty/pyCBD/src/PyCBD/checkerboard_detection',
        '3rdparty/pyCBD/src/PyCBD/checkerboard_enhancement',
        '3rdparty/pyCBD/examples',
        '3rdparty/pyCBD/data'
    ]
    
    for dir_path in directories:
        if os.path.exists(dir_path):
            print(f"✅ {dir_path}")
            # 列出目录内容
            try:
                files = os.listdir(dir_path)
                for file in files[:5]:  # 只显示前5个文件
                    print(f"   📄 {file}")
                if len(files) > 5:
                    print(f"   ... 还有 {len(files) - 5} 个文件")
            except Exception as e:
                print(f"   ❌ 无法读取目录: {e}")
        else:
            print(f"❌ {dir_path} - 不存在")
    
    print()

def check_pycbd_imports():
    """
    检查PyCBD模块导入
    """
    print("🔍 检查PyCBD模块导入...")
    
    try:
        # 尝试导入基本模块
        import PyCBD
        print("✅ PyCBD包导入成功")
        
        # 检查子模块
        modules_to_check = [
            'PyCBD.pipelines',
            'PyCBD.checkerboard_detection',
            'PyCBD.checkerboard_enhancement',
            'PyCBD.logger_configuration'
        ]
        
        for module_name in modules_to_check:
            try:
                __import__(module_name)
                print(f"✅ {module_name} 导入成功")
            except ImportError as e:
                print(f"❌ {module_name} 导入失败: {e}")
        
    except ImportError as e:
        print(f"❌ PyCBD包导入失败: {e}")
    
    print()

def check_dependencies():
    """
    检查依赖包
    """
    print("🔍 检查依赖包...")
    
    dependencies = [
        'numpy',
        'cv2',
        'matplotlib',
        'sklearn',
        'scipy',
        'gpy',
        'PIL',
        'h5py',
        'albumentations'
    ]
    
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✅ {dep} 已安装")
        except ImportError:
            print(f"❌ {dep} 未安装")
    
    print()

def analyze_pycbd_code():
    """
    分析PyCBD代码结构
    """
    print("🔍 分析PyCBD代码结构...")
    
    # 读取主要文件
    files_to_analyze = [
        '3rdparty/pyCBD/src/PyCBD/pipelines.py',
        '3rdparty/pyCBD/src/PyCBD/checkerboard_detection/checkerboard_detector.py',
        '3rdparty/pyCBD/src/PyCBD/checkerboard_enhancement/checkerboard_enhancer.py'
    ]
    
    for file_path in files_to_analyze:
        if os.path.exists(file_path):
            print(f"📄 分析 {file_path}")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')
                    print(f"   行数: {len(lines)}")
                    
                    # 查找主要类和方法
                    classes = [line.strip() for line in lines if line.strip().startswith('class ')]
                    methods = [line.strip() for line in lines if line.strip().startswith('def ')]
                    
                    print(f"   类数量: {len(classes)}")
                    if classes:
                        print(f"   主要类: {classes[:3]}")
                    
                    print(f"   方法数量: {len(methods)}")
                    if methods:
                        print(f"   主要方法: {methods[:5]}")
                        
            except Exception as e:
                print(f"   ❌ 读取文件失败: {e}")
        else:
            print(f"❌ {file_path} - 不存在")
    
    print()

def check_example_images():
    """
    检查示例图像
    """
    print("🔍 检查示例图像...")
    
    example_images = [
        '3rdparty/pyCBD/examples/images/thermal.tiff',
        '3rdparty/pyCBD/examples/images/flare.jpg',
        '3rdparty/pyCBD/examples/images/warped.jpg'
    ]
    
    for image_path in example_images:
        if os.path.exists(image_path):
            print(f"✅ {image_path}")
        else:
            print(f"❌ {image_path} - 不存在")
    
    # 检查我们项目的测试图像
    print("\n🔍 检查项目测试图像...")
    project_images = [
        'data/04.png',
        'data/00.png',
        'data/01.png',
        'data/02.png',
        'data/03.png',
        'data/05.png'
    ]
    
    for image_path in project_images:
        if os.path.exists(image_path):
            print(f"✅ {image_path}")
        else:
            print(f"❌ {image_path} - 不存在")
    
    print()

def main():
    """
    主函数
    """
    print("🎯 PyCBD简化调试脚本")
    print("=" * 50)
    
    check_pycbd_structure()
    check_pycbd_imports()
    check_dependencies()
    analyze_pycbd_code()
    check_example_images()
    
    print("=" * 50)
    print("📋 调试总结:")
    print("1. 检查了PyCBD源码结构")
    print("2. 尝试导入PyCBD模块")
    print("3. 检查了依赖包状态")
    print("4. 分析了主要代码文件")
    print("5. 检查了示例图像")
    print("\n💡 建议:")
    print("- 如果依赖包缺失，请安装: pip install numpy opencv-python matplotlib scikit-learn scipy")
    print("- 如果PyCBD导入失败，可能需要安装: pip install -e 3rdparty/pyCBD/")
    print("- 确保Python环境正确配置")

if __name__ == "__main__":
    main() 