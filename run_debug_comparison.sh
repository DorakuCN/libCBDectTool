#!/bin/bash

# MATLAB vs C++ 详细调试对比一键运行脚本
# 此脚本将运行所有调试工具并生成完整的对比分析报告

echo "======================================================="
echo "    MATLAB vs C++ 详细调试对比分析套件"
echo "======================================================="
echo "此脚本将运行以下分析工具："
echo "1. C++ 详细调试版本 (多配置对比)"
echo "2. C++ vs MATLAB 综合对比分析"
echo "3. 生成完整的分析报告和可视化图表"
echo "======================================================="

# 检查是否在正确的目录
if [ ! -f "CMakeLists.txt" ]; then
    echo "❌ 错误：请在项目根目录运行此脚本"
    exit 1
fi

# 创建build目录
echo "🔧 准备构建环境..."
mkdir -p build
cd build

# 检查图像文件
IMAGE_PATH="../data/04.png"
if [ ! -f "$IMAGE_PATH" ]; then
    echo "❌ 错误：找不到测试图像 $IMAGE_PATH"
    exit 1
fi

echo "✅ 测试图像: $IMAGE_PATH"

# 构建项目
echo "🔨 编译调试工具..."
cmake .. > cmake.log 2>&1
if [ $? -ne 0 ]; then
    echo "❌ CMake配置失败，检查cmake.log"
    exit 1
fi

make debug_comparison matlab_cpp_comparison -j4 > make.log 2>&1
if [ $? -ne 0 ]; then
    echo "❌ 编译失败，检查make.log"
    exit 1
fi

echo "✅ 编译完成"

# 创建结果目录
mkdir -p result

echo ""
echo "======================================================="
echo "    开始调试分析"
echo "======================================================="

# 运行详细调试对比
echo "🔍 运行详细调试对比分析..."
echo "   - 测试多种配置参数"
echo "   - 生成详细的模块日志"
echo "   - 分析各阶段处理结果"

./debug_comparison $IMAGE_PATH > result/detailed_debug_output.log 2>&1
if [ $? -eq 0 ]; then
    echo "✅ 详细调试分析完成"
else
    echo "⚠️  详细调试分析遇到问题，但继续执行"
fi

# 运行综合对比分析
echo ""
echo "📊 运行MATLAB vs C++综合对比分析..."
echo "   - 标准化性能测试"
echo "   - 生成对比表格和图表"
echo "   - 评估最佳配置"

./matlab_cpp_comparison $IMAGE_PATH
if [ $? -eq 0 ]; then
    echo "✅ 综合对比分析完成"
else
    echo "❌ 综合对比分析失败"
    exit 1
fi

echo ""
echo "======================================================="
echo "    分析完成 - 结果总结"
echo "======================================================="

# 统计生成的文件
echo "📁 生成的分析文件："
ls -la result/ | grep -E '\.(png|txt|log)$' | while read line; do
    filename=$(echo $line | awk '{print $9}')
    size=$(echo $line | awk '{print $5}')
    echo "   - $filename ($(($size / 1024))KB)"
done

echo ""
echo "🎯 主要发现："

# 从报告中提取关键信息
if [ -f "result/matlab_cpp_detailed_report.txt" ]; then
    echo "   - MATLAB目标：51个角点在目标区域"
    
    # 提取最佳结果
    best_method=$(grep "Best performing method:" result/matlab_cpp_detailed_report.txt | cut -d: -f2 | xargs)
    best_corners=$(grep "MATLAB region corners:" result/matlab_cpp_detailed_report.txt | head -1 | grep -o '[0-9]*\/51')
    best_accuracy=$(grep "Accuracy:" result/matlab_cpp_detailed_report.txt | head -1 | grep -o '[0-9]*\.[0-9]*%')
    
    if [ ! -z "$best_method" ]; then
        echo "   - 最佳配置：$best_method"
        echo "   - 检测结果：$best_corners 角点"
        echo "   - 准确率：$best_accuracy"
        
        # 判断性能等级
        accuracy_num=$(echo $best_accuracy | sed 's/%//')
        if (( $(echo "$accuracy_num >= 70" | bc -l) )); then
            echo "   - 性能评级：✅ GOOD - 接近MATLAB目标"
        elif (( $(echo "$accuracy_num >= 50" | bc -l) )); then
            echo "   - 性能评级：⚠️ MODERATE - 有改进空间"
        else
            echo "   - 性能评级：❌ POOR - 需要优化"
        fi
    fi
fi

echo ""
echo "📊 可视化分析："
if [ -f "result/matlab_cpp_comparison.png" ]; then
    echo "   - ✅ 多配置对比图表已生成"
fi

echo ""
echo "📋 详细报告："
if [ -f "result/matlab_cpp_detailed_report.txt" ]; then
    echo "   - ✅ 详细分析报告已生成"
    echo "   - 位置：result/matlab_cpp_detailed_report.txt"
fi

echo ""
echo "🔧 调试日志："
if [ -f "result/cpp_debug_log.txt" ]; then
    echo "   - ✅ C++ 调试日志已生成"
fi

echo ""
echo "======================================================="
echo "    使用建议"
echo "======================================================="
echo "📖 查看分析结果："
echo "   cat result/matlab_cpp_detailed_report.txt"
echo ""
echo "🖼️  查看可视化对比："
echo "   open result/matlab_cpp_comparison.png  # macOS"
echo "   xdg-open result/matlab_cpp_comparison.png  # Linux"
echo ""
echo "🔍 查看详细调试日志："
echo "   cat result/cpp_debug_log.txt"
echo ""
echo "📄 查看最终总结报告："
echo "   cat ../DEBUG_COMPARISON_FINAL_REPORT.md"

echo ""
echo "🎉 调试对比分析套件运行完成！"
echo "=======================================================" 