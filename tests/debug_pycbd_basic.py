#!/usr/bin/env python3
"""
PyCBD基本调试脚本
尝试使用我们的libcbdetCpp库
"""

import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 添加我们的C++库路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'build'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# 添加PyCBD路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '3rdparty', 'pyCBD', 'src'))

def test_libcbdet_import():
    """
    测试libcbdet导入
    """
    print("🔍 测试libcbdet导入...")
    
    try:
        # 尝试导入我们的C++库
        import cbdetect
        print("✅ cbdetect 导入成功")
        return True
    except ImportError as e:
        print(f"❌ cbdetect 导入失败: {e}")
        
        # 尝试其他可能的模块名
        try:
            import libcbdet
            print("✅ libcbdet 导入成功")
            return True
        except ImportError as e2:
            print(f"❌ libcbdet 导入失败: {e2}")
    
    return False

def test_pycbd_with_custom_detector():
    """
    测试PyCBD与自定义检测器
    """
    print("\n🔍 测试PyCBD与自定义检测器...")
    
    try:
        from PyCBD.pipelines import CBDPipeline
        print("✅ PyCBD.pipelines 导入成功")
        
        # 创建一个简单的自定义检测器
        class CustomDetector:
            def __init__(self):
                self.name = "CustomDetector"
            
            def detect_checkerboard(self, image):
                print("🔍 使用自定义检测器进行检测...")
                # 这里可以调用我们的C++库
                return True, np.array([]), np.array([])
        
        # 使用自定义检测器
        detector = CBDPipeline(CustomDetector())
        print("✅ 自定义检测器创建成功")
        
        return True
        
    except ImportError as e:
        print(f"❌ PyCBD导入失败: {e}")
        return False

def test_our_cpp_library():
    """
    测试我们的C++库
    """
    print("\n🔍 测试我们的C++库...")
    
    # 检查build目录
    build_dir = "build"
    if os.path.exists(build_dir):
        print(f"✅ build目录存在")
        files = os.listdir(build_dir)
        print(f"📄 build目录文件: {files}")
    else:
        print(f"❌ build目录不存在")
    
    # 检查是否有编译好的库文件
    lib_files = []
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.endswith(('.so', '.dylib', '.dll', '.a')):
                lib_files.append(os.path.join(root, file))
    
    if lib_files:
        print(f"📄 找到库文件: {lib_files}")
    else:
        print("❌ 未找到库文件")
    
    return len(lib_files) > 0

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
        
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print(f"✅ 灰度转换成功: {gray.shape}")
        
        # 显示图像
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("原始图像")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(gray, cmap='gray')
        plt.title("灰度图像")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig("result/test_image_processing.png", dpi=150, bbox_inches='tight')
        print("💾 图像处理结果已保存")
        plt.show()
        
        return True
        
    except Exception as e:
        print(f"❌ 图像处理失败: {e}")
        return False

def create_libcbdet_wrapper():
    """
    创建libcbdet包装器
    """
    print("\n🔍 创建libcbdet包装器...")
    
    wrapper_code = '''
import ctypes
import numpy as np
import os

class LibCBDetect:
    """libcbdet的Python包装器"""
    
    def __init__(self):
        # 尝试加载库文件
        lib_paths = [
            "build/libcbdetect.dylib",  # macOS
            "build/libcbdetect.so",     # Linux
            "build/cbdetect.dll",       # Windows
        ]
        
        self.lib = None
        for lib_path in lib_paths:
            if os.path.exists(lib_path):
                try:
                    self.lib = ctypes.CDLL(lib_path)
                    print(f"✅ 加载库成功: {lib_path}")
                    break
                except Exception as e:
                    print(f"❌ 加载库失败 {lib_path}: {e}")
        
        if self.lib is None:
            print("❌ 无法加载任何库文件")
    
    def detect_checkerboard(self, image):
        """检测棋盘"""
        if self.lib is None:
            return False, np.array([]), np.array([])
        
        # 这里需要根据实际的C++接口定义
        # 暂时返回空结果
        return True, np.array([]), np.array([])

# 创建全局实例
libcbdet = LibCBDetect()
'''
    
    with open("libcbdet_wrapper.py", "w") as f:
        f.write(wrapper_code)
    
    print("✅ libcbdet包装器创建成功")

def main():
    """
    主函数
    """
    print("🎯 PyCBD基本调试脚本")
    print("=" * 50)
    
    # 测试libcbdet导入
    libcbdet_ok = test_libcbdet_import()
    
    # 测试我们的C++库
    cpp_lib_ok = test_our_cpp_library()
    
    # 测试PyCBD
    pycbd_ok = test_pycbd_with_custom_detector()
    
    # 测试图像处理
    image_ok = test_image_processing()
    
    # 创建包装器
    create_libcbdet_wrapper()
    
    print(f"\n{'='*60}")
    print(f"📊 调试总结:")
    print(f"   libcbdet导入: {'✅' if libcbdet_ok else '❌'}")
    print(f"   C++库检查: {'✅' if cpp_lib_ok else '❌'}")
    print(f"   PyCBD测试: {'✅' if pycbd_ok else '❌'}")
    print(f"   图像处理: {'✅' if image_ok else '❌'}")
    
    print(f"\n💡 建议:")
    if not cpp_lib_ok:
        print("- 需要编译C++库: mkdir build && cd build && cmake .. && make")
    if not pycbd_ok:
        print("- 需要安装PyCBD依赖或修复导入问题")
    print("- 可以尝试使用创建的libcbdet_wrapper.py")

if __name__ == "__main__":
    main() 