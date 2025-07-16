#pragma once

#include <vector>
#include <opencv2/opencv.hpp>
#include "corner.h"

namespace cbdetect {

/**
 * @brief 相关性评分器：基于Sample版本的高精度角点评分算法
 * 
 * 该算法使用方向向量投影、梯度滤波和模板匹配来计算准确的角点质量评分
 * 相比简单的统计评分，能显著提高角点质量判断的准确性
 */
class CorrelationScoring {
public:
    /**
     * @brief 构造函数
     */
    CorrelationScoring() = default;

    /**
     * @brief 对角点集合进行高精度评分
     * @param image 输入图像 (灰度, CV_64F, 0-1范围)
     * @param weight_image 权重图像 (梯度强度)
     * @param corners 待评分的角点集合 (输入输出)
     */
    void scoreCorners(const cv::Mat& image,
                     const cv::Mat& weight_image, 
                     Corners& corners);

private:
    /**
     * @brief 计算单个角点的相关性评分
     * @param image_patch 角点区域图像块
     * @param weight_patch 权重图像块
     * @param v1 第一方向向量
     * @param v2 第二方向向量
     * @return 相关性评分
     */
    double computeCorrelationScore(const cv::Mat& image_patch,
                                  const cv::Mat& weight_patch,
                                  const cv::Point2d& v1,
                                  const cv::Point2d& v2);

    /**
     * @brief 创建梯度滤波核
     * @param v1 第一方向向量
     * @param v2 第二方向向量 
     * @param patch_size 滤波核大小
     * @return 梯度滤波核
     */
    cv::Mat createGradientFilter(const cv::Point2d& v1,
                                const cv::Point2d& v2,
                                int patch_size);

    /**
     * @brief 创建相关模板核
     * @param angle1 第一角度
     * @param angle2 第二角度
     * @param half_size 半径大小
     * @return 4个相关模板 (a1, a2, b1, b2)
     */
    std::vector<cv::Mat> createCorrelationPatches(double angle1, 
                                                 double angle2,
                                                 int half_size);

    /**
     * @brief 计算梯度评分
     * @param weight_patch 权重图像块
     * @param gradient_filter 梯度滤波核
     * @return 梯度评分
     */
    double computeGradientScore(const cv::Mat& weight_patch,
                               const cv::Mat& gradient_filter);

    /**
     * @brief 计算强度评分 (双模式检测)
     * @param image_patch 图像块
     * @param templates 相关模板
     * @return 强度评分
     */
    double computeIntensityScore(const cv::Mat& image_patch,
                                const std::vector<cv::Mat>& templates);

    /**
     * @brief 标准化图像块
     * @param input 输入图像块
     * @return 标准化后的图像块
     */
    cv::Mat normalizeImagePatch(const cv::Mat& input);

    /**
     * @brief 提取角点周围的图像块
     * @param image 输入图像
     * @param center 角点中心
     * @param radius 提取半径
     * @return 图像块
     */
    cv::Mat extractImagePatch(const cv::Mat& image,
                             const cv::Point2d& center,
                             int radius);
};

} // namespace cbdetect 