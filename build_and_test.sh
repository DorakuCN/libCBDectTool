#!/bin/bash

# libcbdetect 快速编译和测试脚本
# 用法: ./build_and_test.sh

set -e  # 遇到错误立即退出

echo "========================================"
echo "libcbdetect C++ 编译和测试脚本"
echo "========================================"

# 检查依赖
echo "检查依赖项..."

# 检查cmake
if ! command -v cmake &> /dev/null; then
    echo "错误: 未找到cmake，请先安装cmake"
    exit 1
fi

# 检查OpenCV (通过pkg-config)
if ! pkg-config --exists opencv4 && ! pkg-config --exists opencv; then
    echo "警告: 未找到OpenCV，请确保已正确安装OpenCV"
    echo "Ubuntu/Debian: sudo apt install libopencv-dev"
    echo "macOS: brew install opencv"
fi

# 创建build目录
echo "创建构建目录..."
if [ -d "build" ]; then
    echo "清理现有构建目录..."
    rm -rf build
fi
mkdir build
cd build

# 配置项目
echo "配置CMake项目..."
cmake .. -DCMAKE_BUILD_TYPE=Release

# 编译项目
echo "编译项目..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    make -j$(sysctl -n hw.ncpu)
else
    # Linux
    make -j$(nproc)
fi

echo "编译完成！"

# 检查是否有测试数据
if [ ! -f "../data/04.png" ]; then
    echo "警告: 未找到测试图像 data/04.png"
    echo "请将棋盘格图像放置在 data/ 目录下"
    exit 0
fi

# 创建结果目录
mkdir -p ../result

# 运行demo
echo "运行demo程序..."
echo "输入图像: ../data/04.png"
./demo ../data/04.png

echo "========================================"
echo "构建和测试完成！"
echo "可执行文件位置: build/demo"
echo "库文件位置: build/libcbdetect.a"
echo "结果图像保存在: result/ 目录"
echo "========================================"

# 返回项目根目录
cd ..

echo "使用方法:"
echo "1. 将棋盘格图像放在 data/ 目录"
echo "2. 运行: build/demo [图像路径]"
echo "3. 查看结果图像在 result/ 目录" 