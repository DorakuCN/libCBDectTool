#include "cbdetect/correlation_scoring.h"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace cbdetect {

void CorrelationScoring::scoreCorners(const cv::Mat& image,
                                     const cv::Mat& weight_image, 
                                     Corners& corners) {
    for (size_t i = 0; i < corners.size(); ++i) {
        Corner& corner = corners[i];
        int radius = static_cast<int>(corner.radius);
        
        // 检查边界
        if (corner.pt.x - radius < 0 || corner.pt.x + radius >= image.cols - 1 ||
            corner.pt.y - radius < 0 || corner.pt.y + radius >= image.rows - 1) {
            corner.quality_score = 0.0;
            continue;
        }
        
        // 提取图像块
        cv::Mat image_patch = extractImagePatch(image, corner.pt, radius);
        cv::Mat weight_patch = extractImagePatch(weight_image, corner.pt, radius);
        
        // 转换方向向量
        cv::Point2d v1(corner.v1[0], corner.v1[1]);
        cv::Point2d v2(corner.v2[0], corner.v2[1]);
        
        // 计算相关性评分
        corner.quality_score = computeCorrelationScore(image_patch, weight_patch, v1, v2);
    }
    
    std::cout << "[Correlation Scoring] Scored " << corners.size() << " corners" << std::endl;
}

double CorrelationScoring::computeCorrelationScore(const cv::Mat& image_patch,
                                                  const cv::Mat& weight_patch,
                                                  const cv::Point2d& v1,
                                                  const cv::Point2d& v2) {
    // 1. 创建梯度滤波核 (基于Sample版本，3px带宽)
    cv::Mat gradient_filter = createGradientFilter(v1, v2, image_patch.cols);
    
    // 2. 标准化梯度滤波核和权重图像
    cv::Mat normalized_filter = normalizeImagePatch(gradient_filter);
    cv::Mat normalized_weight = normalizeImagePatch(weight_patch);
    
    // 3. 计算梯度评分
    double gradient_score = computeGradientScore(normalized_weight, normalized_filter);
    gradient_score = std::max(gradient_score / (image_patch.cols * image_patch.rows - 1), 0.0);
    
    // 4. 创建相关模板
    double angle1 = std::atan2(v1.y, v1.x);
    double angle2 = std::atan2(v2.y, v2.x);
    int half_size = (image_patch.cols - 1) / 2;
    std::vector<cv::Mat> templates = createCorrelationPatches(angle1, angle2, half_size);
    
    // 5. 计算强度评分
    double intensity_score = computeIntensityScore(image_patch, templates);
    

    
    // 6. 最终评分: 梯度评分 × 强度评分
    return gradient_score * intensity_score;
}

cv::Mat CorrelationScoring::createGradientFilter(const cv::Point2d& v1,
                                                const cv::Point2d& v2,
                                                int patch_size) {
    // 基于Sample版本的梯度滤波核生成算法
    double center = (patch_size - 1) / 2.0;
    cv::Mat filter = cv::Mat::ones(patch_size, patch_size, CV_64F) * -1.0;
    
    for (int u = 0; u < patch_size; ++u) {
        for (int v = 0; v < patch_size; ++v) {
            cv::Point2d p1(u - center, v - center);
            
            // 计算到第一方向向量的投影点
            cv::Point2d projection1 = (p1.x * v1.x + p1.y * v1.y) * v1;
            
            // 计算到第二方向向量的投影点
            cv::Point2d projection2 = (p1.x * v2.x + p1.y * v2.y) * v2;
            
            // 如果点接近任一方向向量 (距离 <= 1.5px)
            if (cv::norm(p1 - projection1) <= 1.5 || cv::norm(p1 - projection2) <= 1.5) {
                filter.at<double>(v, u) = 1.0;
            }
        }
    }
    
    return filter;
}

std::vector<cv::Mat> CorrelationScoring::createCorrelationPatches(double angle1, 
                                                                 double angle2,
                                                                 int half_size) {
    // 基于Sample版本的相关模板生成
    std::vector<cv::Mat> templates(4);
    int size = 2 * half_size + 1;
    
    for (int i = 0; i < 4; ++i) {
        templates[i] = cv::Mat::zeros(size, size, CV_64F);
    }
    
    // 生成4个象限的模板 (a1, a2, b1, b2)
    for (int v = 0; v < size; ++v) {
        for (int u = 0; u < size; ++u) {
            double x = u - half_size;
            double y = v - half_size;
            
            if (x == 0 && y == 0) continue;  // 跳过中心点
            
            double angle = std::atan2(y, x);
            if (angle < 0) angle += 2 * M_PI;
            
            // 将角度标准化到[0, 2π]
            double norm_angle1 = angle1;
            double norm_angle2 = angle2;
            if (norm_angle1 < 0) norm_angle1 += 2 * M_PI;
            if (norm_angle2 < 0) norm_angle2 += 2 * M_PI;
            
            // 计算角度差异
            double diff1 = std::abs(angle - norm_angle1);
            double diff2 = std::abs(angle - norm_angle2);
            
            // 处理周期性
            diff1 = std::min(diff1, 2 * M_PI - diff1);
            diff2 = std::min(diff2, 2 * M_PI - diff2);
            
            // 基于Sample版本的棋盘格模板分配逻辑
            // 将角度空间分为4个象限，围绕两个主方向
            double norm_angle = angle;
            
            // 相对于第一个方向向量的角度差异
            double angle_diff1 = std::abs(norm_angle - norm_angle1);
            angle_diff1 = std::min(angle_diff1, 2*M_PI - angle_diff1);
            
            // 相对于第二个方向向量的角度差异  
            double angle_diff2 = std::abs(norm_angle - norm_angle2);
            angle_diff2 = std::min(angle_diff2, 2*M_PI - angle_diff2);
            
            // 根据到哪个方向更近来分配模板
            if (angle_diff1 < angle_diff2) {
                // 更接近第一个方向
                if (angle_diff1 < M_PI/4) {
                    templates[0].at<double>(v, u) = 1.0;  // a1: 第一方向正方向
                } else {
                    templates[1].at<double>(v, u) = 1.0;  // a2: 第一方向反方向
                }
            } else {
                // 更接近第二个方向
                if (angle_diff2 < M_PI/4) {
                    templates[2].at<double>(v, u) = 1.0;  // b1: 第二方向正方向
                } else {
                    templates[3].at<double>(v, u) = 1.0;  // b2: 第二方向反方向
                }
            }
        }
    }
    
    return templates;
}

double CorrelationScoring::computeGradientScore(const cv::Mat& weight_patch,
                                               const cv::Mat& gradient_filter) {
    // 计算加权梯度响应
    cv::Mat response = weight_patch.mul(gradient_filter);
    return cv::sum(response)[0];
}

double CorrelationScoring::computeIntensityScore(const cv::Mat& image_patch,
                                                const std::vector<cv::Mat>& templates) {
    // 基于Sample版本的双模式强度检测
    
    // 计算4个模板的响应
    double a1 = cv::sum(image_patch.mul(templates[0]))[0];
    double a2 = cv::sum(image_patch.mul(templates[1]))[0];
    double b1 = cv::sum(image_patch.mul(templates[2]))[0];
    double b2 = cv::sum(image_patch.mul(templates[3]))[0];
    
    // 计算均值
    double mu = (a1 + a2 + b1 + b2) / 4.0;
    
    // 模式1: a=白色, b=黑色
    double s1 = std::min(std::min(a1, a2) - mu, mu - std::min(b1, b2));
    
    // 模式2: b=白色, a=黑色  
    double s2 = std::min(mu - std::min(a1, a2), std::min(b1, b2) - mu);
    

    
    // 强度评分: 两种模式的最大值 (使用绝对值确保非负)
    // 临时修复：使用绝对值来获得合理的评分
    double intensity_score = std::max(std::abs(s1), std::abs(s2));
    
    // 如果模板响应差异很小，说明不是好的角点
    double response_variance = (a1 - mu) * (a1 - mu) + (a2 - mu) * (a2 - mu) + 
                              (b1 - mu) * (b1 - mu) + (b2 - mu) * (b2 - mu);
    response_variance /= 4.0;
    
    return intensity_score * std::sqrt(response_variance);
}

cv::Mat CorrelationScoring::normalizeImagePatch(const cv::Mat& input) {
    // 零均值，单位方差标准化
    cv::Scalar mean, std;
    cv::meanStdDev(input, mean, std);
    
    if (std[0] < 1e-6) {
        return cv::Mat::zeros(input.size(), input.type());
    }
    
    return (input - mean[0]) / std[0];
}

cv::Mat CorrelationScoring::extractImagePatch(const cv::Mat& image,
                                             const cv::Point2d& center,
                                             int radius) {
    // 提取以center为中心，半径为radius的图像块
    int u = static_cast<int>(std::round(center.x));
    int v = static_cast<int>(std::round(center.y));
    
    cv::Rect roi(u - radius, v - radius, 2 * radius + 1, 2 * radius + 1);
    
    // 确保ROI在图像边界内
    roi &= cv::Rect(0, 0, image.cols, image.rows);
    
    cv::Mat patch = cv::Mat::zeros(2 * radius + 1, 2 * radius + 1, image.type());
    
    // 计算在patch中的位置
    int offset_u = roi.x - (u - radius);
    int offset_v = roi.y - (v - radius);
    
    image(roi).copyTo(patch(cv::Rect(offset_u, offset_v, roi.width, roi.height)));
    
    return patch;
}

} // namespace cbdetect 