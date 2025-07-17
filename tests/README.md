# 测试脚本说明

本目录包含所有Python测试脚本和包装器，用于测试和比较不同的棋盘格检测算法。

## 核心测试脚本

### 1. 算法对比测试
- **compare_pycbd_libcbdetcpp.py** - 主要的算法对比测试脚本
  - 功能：对比PyCBD和libcbdetCpp的检测性能
  - 输出：JSON结果文件、可视化图表、统计报告
  - 使用：`python compare_pycbd_libcbdetcpp.py`

### 2. 结果分析
- **comparison_summary.py** - 对比结果分析脚本
  - 功能：分析对比测试结果，生成统计报告
  - 输入：comparison_results.json
  - 输出：详细分析报告和可视化图表

## C++库包装器

### 1. 基础包装器
- **libcbdet_wrapper.py** - 基础C++库Python包装器
  - 功能：直接调用C++共享库函数
  - 状态：基础版本，功能有限

- **libcbdet_wrapper_v2.py** - 改进版包装器
  - 功能：增强的错误处理和参数验证
  - 状态：改进版本，更稳定

### 2. 高级包装器
- **libcbdetCpp_wrapper.py** - 完整功能包装器
  - 功能：完整的C++库功能封装
  - 特性：支持所有检测参数和选项
  - 使用：`python test_cpp_wrapper.py`

- **cpp_detector_wrapper.py** - 检测器专用包装器
  - 功能：专门用于棋盘格检测的包装器
  - 特性：简化的API，专注于检测功能

## PyCBD集成

### 1. 配置脚本
- **pycbd_config.py** - PyCBD配置管理
  - 功能：管理PyCBD的配置参数
  - 特性：支持多种配置模式

- **pycbd_example_config.py** - PyCBD示例配置
  - 功能：提供PyCBD的示例配置
  - 特性：包含完整的配置示例

### 2. 兼容性检测器
- **pycbd_compatible_detector.py** - PyCBD兼容检测器
  - 功能：实现PyCBD兼容的检测器接口
  - 特性：可以替代PyCBD的libCBDetect模块
  - 使用：`python test_pycbd_compatible.py`

### 3. 调试脚本
- **debug_pycbd.py** - PyCBD基础调试
  - 功能：基础PyCBD功能测试
  - 状态：基础调试版本

- **debug_pycbd_simple.py** - 简化PyCBD调试
  - 功能：简化的PyCBD测试
  - 特性：快速验证基本功能

- **debug_pycbd_basic.py** - 基础功能调试
  - 功能：测试PyCBD基础功能
  - 特性：最小化依赖测试

- **debug_pycbd_conda.py** - Conda环境调试
  - 功能：在Conda环境中测试PyCBD
  - 特性：环境隔离测试

## 测试脚本

### 1. 包装器测试
- **test_libcbdet_wrapper.py** - 基础包装器测试
  - 功能：测试基础C++库包装器
  - 输入：测试图像
  - 输出：检测结果和性能指标

- **test_libcbdet_wrapper_v2.py** - 改进版包装器测试
  - 功能：测试改进版C++库包装器
  - 特性：更详细的错误报告

- **test_cpp_wrapper.py** - C++包装器综合测试
  - 功能：测试完整的C++库包装器
  - 特性：全面的功能验证

### 2. PyCBD测试
- **test_pycbd_compatible.py** - PyCBD兼容性测试
  - 功能：测试PyCBD兼容检测器
  - 特性：验证与PyCBD的兼容性

## 使用指南

### 快速测试
```bash
# 运行主要对比测试
python compare_pycbd_libcbdetcpp.py

# 测试C++包装器
python test_cpp_wrapper.py

# 测试PyCBD兼容性
python test_pycbd_compatible.py
```

### 调试模式
```bash
# 基础PyCBD调试
python debug_pycbd_simple.py

# 完整功能调试
python debug_pycbd_conda.py
```

### 结果分析
```bash
# 分析对比结果
python comparison_summary.py

# 查看结果文件
cat ../logs/comparison_results.json
```

## 依赖要求

### Python包
- numpy
- opencv-python
- matplotlib
- scipy
- pandas

### 系统依赖
- C++编译环境
- OpenCV库
- CMake

### 环境配置
```bash
# 创建虚拟环境
python -m venv test_env
source test_env/bin/activate

# 安装依赖
pip install -r requirements.txt
```

## 输出文件

### 结果文件
- `../logs/comparison_results.json` - 对比测试结果
- `../result/detection_visualization.png` - 检测可视化
- `../result/detailed_comparison_analysis.png` - 详细分析图表

### 日志文件
- `../logs/cpp_debug_detailed.txt` - C++调试日志
- `../logs/matlab_debug_detailed.txt` - MATLAB调试日志

## 注意事项

1. 确保C++库已正确编译
2. 检查图像文件路径是否正确
3. 验证Python环境配置
4. 注意内存使用情况
5. 定期清理临时文件 