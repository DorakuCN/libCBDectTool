# libCBDectTool - 棋盘格角点检测工具集

一个综合性的棋盘格角点检测工具集，集成了多种检测算法和工具链，包括MATLAB、C++和Python实现。

## 项目概述

libCBDectTool 是一个高性能的棋盘格角点检测工具集，提供了多种算法实现和完整的测试工具链。项目包含：

- **C++核心检测器** - 高性能的C++实现
- **MATLAB版本** - 用于算法验证和对比
- **PyCBD集成** - Python增强版检测工具箱
- **完整测试工具链** - 自动化测试和性能分析

## 目录结构

```
libCBDectTool/
├── src/                    # C++核心检测算法
├── include/                # C++头文件
├── data/                   # 测试图像数据
├── tests/                  # Python测试脚本和包装器
├── scripts/                # Shell脚本和构建工具
├── docs/                   # 项目文档和分析报告
├── logs/                   # 调试输出和结果日志
├── result/                 # 检测结果图像
├── build/                  # 构建输出目录
├── 3rdparty/              # 第三方组件
│   ├── libcbdetM/         # MATLAB版本
│   ├── libcdetSample/     # 原始C++示例
│   └── pyCBD/             # Python增强版
└── python_binding/        # Python绑定
```

## 快速开始

### 1. 构建C++项目
```bash
# 自动构建和测试
./scripts/build_and_test.sh
```

### 2. 运行Python测试
```bash
# 安装Python依赖
cd tests
pip install -r requirements.txt

# 运行算法对比测试
python compare_pycbd_libcbdetcpp.py
```

### 3. 查看结果
```bash
# 查看检测结果
open result/detection_visualization.png

# 查看对比分析
open result/detailed_comparison_analysis.png
```

## 功能特性

### 核心检测算法
- **角点检测** - 高精度角点定位
- **棋盘格识别** - 自动棋盘格结构分析
- **亚像素精化** - 提高检测精度
- **模板匹配** - 相关性评分算法
- **非极大值抑制** - 去除重复检测

### 算法对比
- **libcbdetCpp**: 成功率 83.3%, 平均检测时间 0.15s
- **PyCBD**: 成功率 50.0%, 平均检测时间 0.08s

### 测试工具链
- 自动化批量测试
- 性能指标统计
- 结果可视化
- 详细调试输出

## 使用指南

### C++检测器
```cpp
#include "cbdetect/chessboard_detector.h"

// 创建检测器
ChessboardDetector detector;

// 检测棋盘格
std::vector<Chessboard> boards = detector.detect(image);
```

### Python包装器
```python
from tests.libcbdetCpp_wrapper import LibCBDetCppWrapper

# 创建包装器
wrapper = LibCBDetCppWrapper()

# 检测棋盘格
corners, boards = wrapper.detect("image.png")
```

### PyCBD集成
```python
from tests.pycbd_compatible_detector import PyCBDCompatibleDetector

# 创建兼容检测器
detector = PyCBDCompatibleDetector()

# 检测棋盘格
result = detector.detect(image)
```

## 性能对比

### 检测成功率
- **libcbdetCpp**: 83.3% (5/6 图像)
- **PyCBD**: 50.0% (3/6 图像)

### 执行时间
- **libcbdetCpp**: 平均 0.15s
- **PyCBD**: 平均 0.08s

### 角点检测精度
- **libcbdetCpp**: 平均 84.2 个角点
- **PyCBD**: 平均 82.8 个角点

## 调试工具

### 详细调试模式
```bash
# 运行详细调试对比
./scripts/run_debug_comparison.sh

# 查看调试日志
cat logs/cpp_debug_detailed.txt
```

### 性能分析
```bash
# 运行性能对比
./scripts/compare_cpp_matlab.sh

# 查看性能报告
cat logs/performance_report.txt
```

## 环境要求

### 系统依赖
- **C++**: CMake >= 3.10, OpenCV >= 4.0, gcc >= 7.0
- **Python**: Python >= 3.8, 见 `tests/requirements.txt`
- **MATLAB**: R2018b+ (可选，用于对比)

### 安装步骤
```bash
# 1. 克隆项目
git clone <repository-url>
cd libCBDectTool

# 2. 构建C++项目
./scripts/build_and_test.sh

# 3. 安装Python依赖
cd tests
pip install -r requirements.txt

# 4. 运行测试
python compare_pycbd_libcbdetcpp.py
```

## 文档

- [项目状况总结](docs/PROJECT_STATUS_SUMMARY.md) - 详细的项目状态和功能说明
- [测试脚本说明](tests/README.md) - Python测试脚本使用指南
- [脚本工具说明](scripts/README.md) - Shell脚本使用指南
- [算法对比报告](docs/PYCBD_LIBCBDETCPP_COMPARISON_REPORT.md) - 详细的算法对比分析
- [核心算法分析](docs/PYCBD_CORE_ALGORITHM_ANALYSIS.md) - PyCBD核心算法分析

## 贡献

欢迎提交Issue和Pull Request来改进项目。

## 许可证

本项目基于MIT许可证开源，详见 [LICENSE](LICENSE) 文件。

## 引用

如果您觉得这个软件有用，请引用：

```bibtex
@INPROCEEDINGS{Geiger12,
 author = {Andreas Geiger and Frank Moosmann and Omer Car and Bernhard Schuster},
 title = {Automatic Camera and Range Sensor Calibration using a single Shot},
 booktitle = {International Conference on Robotics and Automation (ICRA)},
 year = {2012},
 month = {May},
 address = {St. Paul, USA}
}
```

## 联系方式

- 项目维护者: [您的姓名]
- 邮箱: [您的邮箱]
- 项目地址: [GitHub链接] 