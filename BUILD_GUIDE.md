# libcbdetect 编译和测试指南

## 前提条件

### 系统要求
- **操作系统**: macOS 10.15+, Ubuntu 18.04+, Windows 10+
- **编译器**: 支持C++14的编译器 (GCC 5.4+, Clang 3.4+, MSVC 2017+)
- **CMake**: 版本 3.10 或更高

### 依赖库
1. **OpenCV** >= 3.4 (推荐 4.x)
2. **Eigen3** >= 3.3 (可选，当前版本已禁用)

## 快速开始

### 1. 克隆代码
```bash
cd libcbdetect
ls -la  # 确认项目文件结构
```

### 2. 一键编译和测试
```bash
# 给脚本执行权限
chmod +x build_and_test.sh

# 运行自动化构建脚本
./build_and_test.sh
```

### 3. 手动编译（可选）
```bash
# 创建构建目录
mkdir build && cd build

# 配置项目
cmake .. -DCMAKE_BUILD_TYPE=Release

# 编译
make -j$(nproc)  # Linux
make -j$(sysctl -n hw.ncpu)  # macOS
```

## 依赖安装指南

### macOS (Homebrew)
```bash
# 安装基础工具
brew install cmake

# 安装OpenCV (当前正在安装中...)
brew install opencv

# 可选：安装Eigen3
brew install eigen
```

### Ubuntu/Debian
```bash
# 更新包列表
sudo apt update

# 安装依赖
sudo apt install build-essential cmake
sudo apt install libopencv-dev
sudo apt install libeigen3-dev  # 可选
```

### Windows (vcpkg)
```bash
# 安装依赖
vcpkg install opencv
vcpkg install eigen3  # 可选

# 配置CMake
cmake .. -DCMAKE_TOOLCHAIN_FILE=path/to/vcpkg.cmake
```

## 编译配置选项

### 构建类型
```bash
# Debug模式 (包含调试信息)
cmake .. -DCMAKE_BUILD_TYPE=Debug

# Release模式 (优化性能)
cmake .. -DCMAKE_BUILD_TYPE=Release

# RelWithDebInfo模式 (优化+调试信息)
cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo
```

### OpenCV路径指定
如果OpenCV安装在非标准位置：
```bash
# 指定OpenCV路径
cmake .. -DOpenCV_DIR=/path/to/opencv/lib/cmake/opencv4

# macOS Homebrew路径示例
cmake .. -DOpenCV_DIR=/opt/homebrew/lib/cmake/opencv4
```

### 编译选项
```bash
# 启用详细输出
make VERBOSE=1

# 指定线程数
make -j4  # 使用4个线程

# 只编译库，不编译demo
make cbdetect
```

## 运行测试

### 基础测试
```bash
# 编译完成后，运行demo
./demo data/04.png

# 或使用默认测试图像
./demo
```

### 功能测试

#### 1. 测试不同检测方法
修改 `src/demo.cpp` 中的参数：

```cpp
// 模板匹配 (默认，精度高)
params.detect_method = DetectMethod::TEMPLATE_MATCH_FAST;

// Hessian响应 (速度快)
params.detect_method = DetectMethod::HESSIAN_RESPONSE;

// Harris角点 (基础方法)
params.detect_method = DetectMethod::HARRIS_CORNER;
```

#### 2. 测试调试功能
```cpp
// 启用处理过程显示
params.show_processing = true;

// 启用调试图像显示
params.show_debug_images = true;
```

#### 3. 测试不同角点类型
```cpp
// 常规棋盘格
params.corner_type = CornerType::SADDLE_POINT;

// Deltille模式
params.corner_type = CornerType::MONKEY_SADDLE_POINT;
```

### 性能测试
```bash
# 计时运行
time ./demo data/04.png

# 内存使用监控 (Linux)
valgrind --tool=massif ./demo data/04.png

# 性能分析 (macOS)
xcrun xctrace record --template "Time Profiler" --launch ./demo data/04.png
```

## 故障排除

### 常见编译错误

#### 1. OpenCV未找到
```
CMake Error: find_package(OpenCV REQUIRED)
```
**解决方案**:
```bash
# 检查OpenCV安装
pkg-config --modversion opencv4

# 设置OpenCV路径
export OpenCV_DIR=/opt/homebrew/lib/cmake/opencv4
```

#### 2. C++14支持问题
```
error: C++14 features not supported
```
**解决方案**:
```bash
# 检查编译器版本
g++ --version
clang++ --version

# 更新编译器或指定新版本
export CXX=/usr/bin/g++-7
```

#### 3. 模板匹配头文件错误
```
fatal error: 'cbdetect/template_matching.h' file not found
```
**解决方案**:
```bash
# 确认文件存在
ls -la include/cbdetect/
ls -la src/

# 重新生成构建文件
rm -rf build && mkdir build && cd build && cmake ..
```

### 运行时错误

#### 1. 图像读取失败
```
Error: Could not load image data/04.png
```
**解决方案**:
```bash
# 检查图像文件
ls -la data/
file data/04.png

# 使用绝对路径
./demo /absolute/path/to/image.png
```

#### 2. 内存访问错误
```
Segmentation fault (core dumped)
```
**解决方案**:
```bash
# 使用调试版本
cmake .. -DCMAKE_BUILD_TYPE=Debug
make
gdb ./demo
```

## 性能优化建议

### 1. 编译优化
```bash
# 启用所有优化
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-O3 -march=native"

# 启用链接时优化
cmake .. -DCMAKE_CXX_FLAGS="-flto"
```

### 2. 运行时优化
```cpp
// 调整检测参数以平衡速度和精度
DetectionParams params;
params.detect_method = DetectMethod::HESSIAN_RESPONSE;  // 更快
params.template_radii = {8};  // 减少尺度数量
params.nms_radius = 2;  // 减小NMS半径
```

### 3. 内存优化
```cpp
// 预分配图像缓冲区
// 重用检测器实例
// 及时释放不需要的图像
```

## 集成到其他项目

### CMake集成
```cmake
# 在你的CMakeLists.txt中
find_package(OpenCV REQUIRED)
add_subdirectory(path/to/libcbdetect)

target_link_libraries(your_target cbdetect ${OpenCV_LIBS})
target_include_directories(your_target PRIVATE path/to/libcbdetect/include)
```

### 代码示例
```cpp
#include "cbdetect/cbdetect.h"

int main() {
    cv::Mat image = cv::imread("image.jpg");
    
    // 简单使用
    auto chessboards = cbdetect::detect(image);
    
    // 高级使用
    cbdetect::DetectionParams params;
    params.detect_method = cbdetect::DetectMethod::TEMPLATE_MATCH_FAST;
    params.show_processing = true;
    
    cbdetect::ChessboardDetector detector(params);
    auto corners = detector.findCorners(image);
    auto boards = detector.chessboardsFromCorners(corners);
    
    return 0;
}
```

## 下一步

1. **等待OpenCV安装完成**: 当前正在后台安装OpenCV
2. **运行自动化测试**: 使用 `./build_and_test.sh`
3. **查看检测结果**: 结果图像将保存在 `result/` 目录
4. **阅读改进文档**: 查看 `IMPROVEMENTS.md` 了解新功能
5. **参考架构分析**: 查看 `ANALYSIS.md` 了解技术细节

## 技术支持

如遇问题，请检查：
1. 编译输出中的具体错误信息
2. OpenCV和CMake版本兼容性
3. 文件路径和权限设置
4. 系统环境变量配置

更多技术细节请参考项目文档：
- `README.md` - 项目概述和基础使用
- `ANALYSIS.md` - 深入的架构分析
- `IMPROVEMENTS.md` - 功能改进总结 