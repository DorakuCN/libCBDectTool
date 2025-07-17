#!/bin/bash

echo "================================="
echo "C++ vs MATLAB Comparison Script"
echo "================================="

# 确保在正确的目录
cd /Users/genesis/GitHub/libCBDectTool

echo
echo "=== RUNNING C++ VERSION ==="
echo "Command: ./build/demo data/04.png"
echo

# 运行C++版本并保存输出
./build/demo data/04.png > cpp_results.txt 2>&1

# 显示C++结果的关键信息
echo "C++ Results Summary:"
grep -E "(corners|chessboards|Energy check|SUCCESS:|Found.*corners|Detected.*chessboards)" cpp_results.txt || echo "No results found"

echo
echo "=== RUNNING MATLAB VERSION ==="
echo "Command: cd 3rdparty/libcbdetM && matlab -batch demo"
echo

# 运行MATLAB版本
cd 3rdparty/libcbdetM

# 使用matlab -batch来运行不带GUI的MATLAB
matlab -batch "demo" > ../../matlab_results.txt 2>&1

cd ../..

# 显示MATLAB结果的关键信息
echo "MATLAB Results Summary:"
grep -E "(corners|chessboards|Energy check|SUCCESS:|Found.*corners|Processing seed)" matlab_results.txt || echo "No results found"

echo
echo "=== COMPARISON SUMMARY ==="

# 提取关键数字进行比较
cpp_corners=$(grep "After merging:" cpp_results.txt | grep -o "[0-9]\+ corners" | head -1 | grep -o "[0-9]\+")
cpp_chessboards=$(grep "Detected.*chessboards" cpp_results.txt | grep -o "[0-9]\+" | head -1)

matlab_corners=$(grep "Found.*corners" matlab_results.txt | grep -o "[0-9]\+" | head -1)
matlab_chessboards=$(grep "Found.*chessboards" matlab_results.txt | grep -o "[0-9]\+" | tail -1)

echo "Corner Detection:"
echo "  C++:    ${cpp_corners:-"N/A"} corners"
echo "  MATLAB: ${matlab_corners:-"N/A"} corners"

echo "Chessboard Detection:"
echo "  C++:    ${cpp_chessboards:-"N/A"} chessboards" 
echo "  MATLAB: ${matlab_chessboards:-"N/A"} chessboards"

if [ "$cpp_corners" = "$matlab_corners" ] && [ "$cpp_chessboards" = "$matlab_chessboards" ]; then
    echo "✅ RESULTS MATCH!"
else
    echo "❌ RESULTS DIFFER"
fi

echo
echo "Full output files saved as:"
echo "  - cpp_results.txt"
echo "  - matlab_results.txt"
echo
echo "For detailed comparison, run:"
echo "  diff -u cpp_results.txt matlab_results.txt" 