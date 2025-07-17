# 脚本工具说明

本目录包含所有Shell脚本和构建工具，用于自动化构建、测试和调试流程。

## 构建脚本

### 1. 主要构建脚本
- **build_and_test.sh** - 主要的构建和测试脚本
  - 功能：自动构建C++项目并运行测试
  - 特性：
    - 自动创建build目录
    - 配置CMake项目
    - 编译所有目标
    - 运行示例程序
    - 生成测试报告
  - 使用：`./build_and_test.sh`
  - 输出：构建日志、测试结果、可执行文件

## 调试脚本

### 1. 调试对比脚本
- **run_debug_comparison.sh** - 详细的调试对比脚本
  - 功能：运行C++和MATLAB版本的详细调试对比
  - 特性：
    - 启用详细调试输出
    - 对比多个测试图像
    - 生成调试日志
    - 创建可视化结果
    - 性能分析
  - 使用：`./run_debug_comparison.sh`
  - 输出：详细调试日志、对比报告、可视化图表

### 2. 算法对比脚本
- **compare_cpp_matlab.sh** - C++/MATLAB算法对比
  - 功能：对比C++和MATLAB版本的检测结果
  - 特性：
    - 批量处理测试图像
    - 结果对比分析
    - 性能统计
    - 准确性评估
  - 使用：`./compare_cpp_matlab.sh`
  - 输出：对比结果、统计报告

## 脚本功能详解

### build_and_test.sh
```bash
#!/bin/bash
# 主要功能：
# 1. 检查依赖
# 2. 创建构建目录
# 3. 配置CMake项目
# 4. 编译源代码
# 5. 运行测试
# 6. 生成报告

# 使用示例：
./build_and_test.sh
```

### run_debug_comparison.sh
```bash
#!/bin/bash
# 主要功能：
# 1. 启用详细调试模式
# 2. 运行C++检测器
# 3. 运行MATLAB检测器
# 4. 对比检测结果
# 5. 生成调试报告
# 6. 创建可视化图表

# 使用示例：
./run_debug_comparison.sh
```

### compare_cpp_matlab.sh
```bash
#!/bin/bash
# 主要功能：
# 1. 批量处理测试图像
# 2. 运行C++检测
# 3. 运行MATLAB检测
# 4. 结果对比分析
# 5. 性能统计
# 6. 生成对比报告

# 使用示例：
./compare_cpp_matlab.sh
```

## 使用指南

### 快速开始
```bash
# 1. 构建项目
./build_and_test.sh

# 2. 运行调试对比
./run_debug_comparison.sh

# 3. 查看结果
ls -la ../logs/
ls -la ../result/
```

### 调试模式
```bash
# 启用详细调试
export DEBUG_LEVEL=2
./run_debug_comparison.sh

# 查看调试日志
tail -f ../logs/cpp_debug_detailed.txt
```

### 性能测试
```bash
# 运行性能对比
./compare_cpp_matlab.sh

# 查看性能报告
cat ../logs/performance_report.txt
```

## 环境要求

### 系统依赖
- bash shell
- cmake (>= 3.10)
- gcc/g++ (>= 7.0)
- OpenCV (>= 4.0)
- MATLAB (可选，用于对比)

### 目录结构
```
scripts/
├── build_and_test.sh
├── run_debug_comparison.sh
├── compare_cpp_matlab.sh
└── README.md

../build/          # 构建输出目录
../logs/           # 日志文件目录
../result/         # 结果文件目录
../data/           # 测试数据目录
```

## 输出文件

### 构建输出
- `../build/` - 编译生成的文件
- `../build/CMakeCache.txt` - CMake缓存
- `../build/Makefile` - 生成的Makefile
- `../build/demo` - 主程序可执行文件

### 调试输出
- `../logs/cpp_debug_detailed.txt` - C++详细调试日志
- `../logs/matlab_debug_detailed.txt` - MATLAB详细调试日志
- `../logs/cpp_results.txt` - C++检测结果
- `../logs/matlab_results.txt` - MATLAB检测结果

### 结果文件
- `../result/detection_visualization.png` - 检测可视化
- `../result/comparison_chart.png` - 对比图表
- `../result/performance_report.txt` - 性能报告

## 错误处理

### 常见问题
1. **CMake配置失败**
   - 检查OpenCV安装
   - 验证CMake版本
   - 确认依赖库路径

2. **编译错误**
   - 检查C++编译器版本
   - 验证头文件路径
   - 确认库文件链接

3. **运行时错误**
   - 检查可执行文件权限
   - 验证输入文件路径
   - 确认输出目录权限

### 调试技巧
```bash
# 启用详细输出
set -x

# 检查环境变量
env | grep -E "(OPENCV|CMAKE)"

# 验证文件存在
ls -la ../data/*.png
```

## 性能优化

### 编译优化
- 使用Release模式编译
- 启用编译器优化
- 使用多线程编译

### 运行优化
- 调整图像处理参数
- 优化内存使用
- 使用并行处理

## 维护说明

### 定期维护
1. 清理构建目录
2. 更新依赖版本
3. 检查脚本兼容性
4. 更新文档

### 版本控制
- 脚本版本管理
- 配置参数版本化
- 结果文件归档
- 日志文件轮转 