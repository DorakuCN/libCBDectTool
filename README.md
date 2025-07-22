# libcbdetCpp - Enhanced Chessboard Detection Library

## Overview
Enhanced version of libcbdetCpp with improved chessboard detection algorithms, featuring robust corner detection, grid building, and missing corner completion.

## Recent Updates (Latest)

### Major Improvements
1. **极简1-D DBSCAN聚类算法**
   - 替换复杂的DBSCAN实现为高效的1-D版本
   - 自动间距估计和参数优化
   - 支持行列标签的精确分配

2. **单应性矩阵 + 局部搜索预测缺失角点**
   - 使用`cv::findHomography`建立(r,c)→(x,y)映射
   - `localCornerSearch`基于OpenCV棋盘格角点检测候选点
   - 完全移除像素模板匹配，提高鲁棒性
   - 支持三种点类型：0=预测-低置信, 1=原测得, 2=补全

3. **改进的角点补全系统**
   - `predictMissingCorners`: 基于单应性矩阵的预测
   - `completeMissingCorners`: 简化的补全流程
   - 智能权重处理，对补全点使用较低权重

4. **增强的异常值剔除**
   - `rejectOutliers2`: 支持点类型权重
   - 自适应k_sigma搜索
   - 局部一致性检查

### Key Features
- **Robust Corner Detection**: Enhanced libcbdetect integration
- **Dynamic Parameter Adjustment**: Automatic parameter optimization
- **Grid Quality Validation**: Fill rate and spacing analysis
- **Bundle Adjustment Support**: Optional Ceres Solver integration
- **Comprehensive Visualization**: Grid drawing and result analysis

### Build Instructions
```bash
mkdir build && cd build
cmake ..
make
```

### Usage
```bash
./bin/example <image_path>
```

### Dependencies
- OpenCV 4.x
- Optional: Ceres Solver (for bundle adjustment)

### Test Results
- **Color.bmp**: 81→119角点，填充率63.6%
- **IR.bmp**: 95→112角点，填充率59.1%

## Technical Details

### DBSCAN Clustering
```cpp
void dbscan1D(const std::vector<float>& proj, float eps, int min_samples, std::vector<int>& labels)
```
- 极简1-D实现，基于排序的区间扩张
- 自动标签归一化
- 参数：`eps = 0.3 * median_spacing`, `min_samples = 2`

### Corner Completion
```cpp
void predictMissingCorners(const cv::Mat& gray, ...)
void completeMissingCorners(const cv::Mat& gray, ...)
```
- 单应性矩阵预测
- 局部搜索精化
- 智能权重处理

### Outlier Rejection
```cpp
void rejectOutliers2(..., const std::vector<char>& point_types, float k_sigma = 3.0f)
```
- 支持点类型权重
- 自适应阈值搜索
- 局部一致性验证

## License
Original libcbdetect license applies. 