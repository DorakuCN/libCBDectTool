#pragma once

#include <vector>
#include <opencv2/opencv.hpp>
#include "corner.h"

namespace cbdetect {

/**
 * @brief 零交叉过滤器：基于Sample版本的高精度角点验证算法
 * 
 * 该算法通过分析角点周围的强度分布零交叉模式和角度直方图
 * 来验证角点是否真正对应棋盘格角点的几何特征
 */
class ZeroCrossingFilter {
public:
    struct FilterParams {
        int n_circle;        // 圆周采样点数
        int n_bin;           // 角度直方图bin数
        int crossing_threshold; // 零交叉阈值
        int need_crossings;   // 需要的零交叉次数 (SaddlePoint)
        int need_modes;       // 需要的模式数 (SaddlePoint)
        double sample_radius_factor; // 采样半径因子
        
                FilterParams() : n_circle(32), n_bin(32), crossing_threshold(1),  // Further relaxed to 1
                         need_crossings(2), need_modes(1), sample_radius_factor(0.75) {}  // Further relaxed to 2
    };

    /**
     * @brief 构造函数
     * @param params 过滤参数
     */
    explicit ZeroCrossingFilter(const FilterParams& params = FilterParams());

    /**
     * @brief 对角点集合进行零交叉过滤
     * @param image 输入图像 (灰度, CV_64F, 0-1范围)
     * @param angle_image 角度图像
     * @param weight_image 权重图像
     * @param corners 待过滤的角点集合 (输入输出)
     * @return 过滤后保留的角点数量
     */
    int filter(const cv::Mat& image, 
               const cv::Mat& angle_image,
               const cv::Mat& weight_image,
               Corners& corners);

private:
    FilterParams params_;
    std::vector<double> cos_table_;  // 预计算的cos值
    std::vector<double> sin_table_;  // 预计算的sin值

    /**
     * @brief 初始化三角函数查找表
     */
    void initializeTrigTables();

    /**
     * @brief 对单个角点进行零交叉检测
     * @param image 输入图像
     * @param angle_image 角度图像  
     * @param weight_image 权重图像
     * @param corner 角点信息
     * @param corner_idx 角点索引
     * @return 是否通过零交叉检测
     */
    bool checkZeroCrossing(const cv::Mat& image,
                          const cv::Mat& angle_image,
                          const cv::Mat& weight_image,
                          const Corner& corner,
                          int corner_idx);

    /**
     * @brief 圆周采样并计算零交叉次数
     * @param image 输入图像
     * @param center_u 中心u坐标
     * @param center_v 中心v坐标
     * @param radius 采样半径
     * @return 零交叉次数
     */
    int countZeroCrossings(const cv::Mat& image,
                          int center_u, int center_v,
                          int radius);

    /**
     * @brief 构建角度直方图并检测模式数
     * @param angle_image 角度图像
     * @param weight_image 权重图像
     * @param center_u 中心u坐标
     * @param center_v 中心v坐标
     * @param radius 区域半径
     * @return 检测到的主要模式数
     */
    int countAngleModes(const cv::Mat& angle_image,
                       const cv::Mat& weight_image,
                       int center_u, int center_v,
                       int radius);

public:
    /**
     * @brief 使用Mean Shift算法查找角度直方图模式
     * @param histogram 角度直方图
     * @param bandwidth Mean Shift带宽
     * @return 模式列表 (位置, 强度)
     */
    std::vector<std::pair<int, double>> findModesMeanShift(
        const std::vector<double>& histogram,
        double bandwidth = 1.5);

private:

    /**
     * @brief 创建权重掩码
     * @param radius 半径
     * @return 权重掩码矩阵
     */
    cv::Mat createWeightMask(int radius);
};

} // namespace cbdetect 