#include "cbdetect/zero_crossing_filter.h"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace cbdetect {

ZeroCrossingFilter::ZeroCrossingFilter(const FilterParams& params) 
    : params_(params) {
    initializeTrigTables();
}

void ZeroCrossingFilter::initializeTrigTables() {
    cos_table_.resize(params_.n_circle);
    sin_table_.resize(params_.n_circle);
    
    for (int i = 0; i < params_.n_circle; ++i) {
        double angle = i * 2.0 * M_PI / (params_.n_circle - 1);
        cos_table_[i] = std::cos(angle);
        sin_table_[i] = std::sin(angle);
    }
}

int ZeroCrossingFilter::filter(const cv::Mat& image, 
                              const cv::Mat& angle_image,
                              const cv::Mat& weight_image,
                              Corners& corners) {
    std::vector<Corner> filtered_corners;
    int kept_count = 0;
    int original_count = corners.size();  // 保存原始数量
    
    for (size_t i = 0; i < corners.size(); ++i) {
        if (checkZeroCrossing(image, angle_image, weight_image, corners[i], i)) {
            filtered_corners.push_back(corners[i]);
            kept_count++;
        }
    }
    
    // 替换现有角点
    corners.corners = filtered_corners;
    
    std::cout << "[Zero Crossing Filter] " << kept_count << "/" << original_count 
              << " corners passed filter (" << (kept_count * 100.0 / original_count) << "%)" << std::endl;
    
    return kept_count;
}

bool ZeroCrossingFilter::checkZeroCrossing(const cv::Mat& image,
                                          const cv::Mat& angle_image,
                                          const cv::Mat& weight_image,
                                          const Corner& corner,
                                          int corner_idx) {
    int center_u = static_cast<int>(std::round(corner.pt.x));
    int center_v = static_cast<int>(std::round(corner.pt.y));
    int radius = static_cast<int>(corner.radius);
    
    // 检查边界
    if (center_u - radius < 0 || center_u + radius >= image.cols - 1 ||
        center_v - radius < 0 || center_v + radius >= image.rows - 1) {
        return false;
    }
    
    // 1. 计算零交叉次数
    int num_crossings = countZeroCrossings(image, center_u, center_v, radius);
    
    // 2. 计算角度模式数
    int num_modes = countAngleModes(angle_image, weight_image, center_u, center_v, radius);
    
    // 3. 验证条件 (基于Sample版本的SaddlePoint条件)
    bool passes_filter = (num_crossings == params_.need_crossings && 
                         num_modes == params_.need_modes);
    

    
    return passes_filter;
}

int ZeroCrossingFilter::countZeroCrossings(const cv::Mat& image,
                                          int center_u, int center_v,
                                          int radius) {
    // 1. 圆周采样
    std::vector<double> circle_values(params_.n_circle);
    for (int j = 0; j < params_.n_circle; ++j) {
        int sample_u = static_cast<int>(std::round(center_u + params_.sample_radius_factor * radius * cos_table_[j]));
        int sample_v = static_cast<int>(std::round(center_v + params_.sample_radius_factor * radius * sin_table_[j]));
        
        // 边界处理
        sample_u = std::max(0, std::min(sample_u, image.cols - 1));
        sample_v = std::max(0, std::min(sample_v, image.rows - 1));
        
        circle_values[j] = image.at<double>(sample_v, sample_u);
    }
    
    // 2. 零中心化 (基于Sample版本的实现)
    auto minmax = std::minmax_element(circle_values.begin(), circle_values.end());
    double min_val = *minmax.first;
    double max_val = *minmax.second;
    
    for (int j = 0; j < params_.n_circle; ++j) {
        circle_values[j] = circle_values[j] - min_val - (max_val - min_val) / 2.0;
    }
    
    // 3. 计算零交叉次数 (复制Sample版本的逻辑)
    int num_crossings = 0;
    
    // 找到第一个零交叉点
    int first_cross_index = 0;
    for (int j = 0; j < params_.n_circle; ++j) {
        bool current_positive = circle_values[j] > 0;
        bool next_positive = circle_values[(j + 1) % params_.n_circle] > 0;
        if (current_positive != next_positive) {  // 发生零交叉
            first_cross_index = (j + 1) % params_.n_circle;
            break;
        }
    }
    
    // 从第一个零交叉点开始计数
    for (int j = first_cross_index, count = 1; j < params_.n_circle + first_cross_index; ++j, ++count) {
        int current_idx = j % params_.n_circle;
        int next_idx = (j + 1) % params_.n_circle;
        
        bool current_positive = circle_values[current_idx] > 0;
        bool next_positive = circle_values[next_idx] > 0;
        
        if (current_positive != next_positive) {  // 发生零交叉
            if (count >= params_.crossing_threshold) {
                num_crossings++;
            }
            count = 1;  // 重置计数
        }
    }
    
    return num_crossings;
}

int ZeroCrossingFilter::countAngleModes(const cv::Mat& angle_image,
                                       const cv::Mat& weight_image,
                                       int center_u, int center_v,
                                       int radius) {
    // 1. 创建权重掩码
    cv::Mat weight_mask = createWeightMask(radius);
    
    // 2. 提取图像块
    int top_left_u = std::max(center_u - radius, 0);
    int top_left_v = std::max(center_v - radius, 0);
    int bottom_right_u = std::min(center_u + radius, weight_image.cols - 1);
    int bottom_right_v = std::min(center_v + radius, weight_image.rows - 1);
    
    // 3. 创建权重子图像
    cv::Mat weight_sub = cv::Mat::zeros(2 * radius + 1, 2 * radius + 1, CV_64F);
    weight_image(cv::Range(top_left_v, bottom_right_v + 1), 
                cv::Range(top_left_u, bottom_right_u + 1)).copyTo(
        weight_sub(cv::Range(top_left_v - center_v + radius, bottom_right_v - center_v + radius + 1),
                  cv::Range(top_left_u - center_u + radius, bottom_right_u - center_u + radius + 1)));
    
    // 4. 应用权重掩码
    weight_sub = weight_sub.mul(weight_mask);
    
    // 5. 权重阈值处理 (基于Sample版本)
    double max_weight;
    cv::minMaxLoc(weight_sub, nullptr, &max_weight);
    double weight_threshold = 0.5 * max_weight;
    
    weight_sub.forEach<double>([weight_threshold](double& val, const int* pos) {
        val = val < weight_threshold ? 0 : val;
    });
    
    // 6. 构建角度直方图
    std::vector<double> angle_histogram(params_.n_bin, 0.0);
    
    for (int v = top_left_v; v <= bottom_right_v; ++v) {
        for (int u = top_left_u; u <= bottom_right_u; ++u) {
            double angle_val = angle_image.at<double>(v, u);
            double weight_val = weight_sub.at<double>(v - center_v + radius, u - center_u + radius);
            
            if (weight_val > 0) {
                int bin = static_cast<int>(std::floor(angle_val / (M_PI / params_.n_bin))) % params_.n_bin;
                angle_histogram[bin] += weight_val;
            }
        }
    }
    
    // 7. 使用Mean Shift查找模式
    auto modes = findModesMeanShift(angle_histogram);
    
    // 8. 计算主要模式数 (强度大于最强模式的50%)
    int num_major_modes = 0;
    if (!modes.empty()) {
        double max_mode_strength = modes[0].second;
        for (const auto& mode : modes) {
            if (2 * mode.second > max_mode_strength) {
                num_major_modes++;
            }
        }
    }
    
    return num_major_modes;
}

std::vector<std::pair<int, double>> ZeroCrossingFilter::findModesMeanShift(
    const std::vector<double>& histogram,
    double bandwidth) {
    
    std::vector<std::pair<int, double>> modes;
    int n = histogram.size();
    
    if (n == 0) return modes;
    
    // 简化的Mean Shift实现
    std::vector<bool> visited(n, false);
    
    for (int i = 0; i < n; ++i) {
        if (visited[i] || histogram[i] == 0) continue;
        
        double center = i;
        double prev_center = center;
        
        // Mean Shift迭代
        for (int iter = 0; iter < 20; ++iter) {
            double numerator = 0.0;
            double denominator = 0.0;
            
            for (int j = 0; j < n; ++j) {
                double distance = std::min(std::abs(j - center), n - std::abs(j - center));
                if (distance <= bandwidth) {
                    numerator += j * histogram[j];
                    denominator += histogram[j];
                }
            }
            
            if (denominator > 0) {
                center = numerator / denominator;
            }
            
            if (std::abs(center - prev_center) < 0.1) break;
            prev_center = center;
        }
        
        // 标记邻近的bin为已访问
        int mode_center = static_cast<int>(std::round(center)) % n;
        for (int j = mode_center - static_cast<int>(bandwidth); 
             j <= mode_center + static_cast<int>(bandwidth); ++j) {
            int idx = (j + n) % n;
            visited[idx] = true;
        }
        
        // 计算该模式的强度
        double mode_strength = 0.0;
        for (int j = mode_center - static_cast<int>(bandwidth); 
             j <= mode_center + static_cast<int>(bandwidth); ++j) {
            int idx = (j + n) % n;
            mode_strength += histogram[idx];
        }
        
        if (mode_strength > 0) {
            modes.emplace_back(mode_center, mode_strength);
        }
    }
    
    // 按强度排序
    std::sort(modes.begin(), modes.end(), 
              [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                  return a.second > b.second;
              });
    
    return modes;
}

cv::Mat ZeroCrossingFilter::createWeightMask(int radius) {
    cv::Mat mask = cv::Mat::zeros(2 * radius + 1, 2 * radius + 1, CV_64F);
    
    cv::Point2f center(radius, radius);
    for (int v = 0; v < mask.rows; ++v) {
        for (int u = 0; u < mask.cols; ++u) {
            double distance = cv::norm(cv::Point2f(u, v) - center);
            if (distance <= radius) {
                // 使用简单的圆形掩码
                mask.at<double>(v, u) = 1.0;
            }
        }
    }
    
    return mask;
}

} // namespace cbdetect 