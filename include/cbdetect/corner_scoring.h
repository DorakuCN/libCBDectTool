#pragma once

#include "cbdetect/corner.h"
#include <opencv2/opencv.hpp>

namespace cbdetect {

// Forward declarations
struct DetectionParams;

// Corner statistics for adaptive filtering
struct CornerStatistics {
    double mean = 0.0;
    double std = 0.0;
    double median = 0.0;
    double percentile_95 = 0.0;
    double percentile_99 = 0.0;
};

// Compute statistical metrics for corner scores
CornerStatistics computeScoreStatistics(const Corners& corners);

// Multi-stage corner filtering functions
void spatialNonMaximumSuppression(Corners& corners, double min_distance = 12.0);
Corners filterByQuality(const Corners& corners, double quality_threshold);
Corners filterByStatistics(const Corners& corners, const CornerStatistics& stats);
Corners filterByTopPercentage(const Corners& corners, double percentage);

// Main filtering function implementing multi-stage strategy
void filterCorners(Corners& corners, const DetectionParams& params);

// Enhanced corner scoring function
void scoreCorners(Corners& corners, const cv::Mat& image, const cv::Mat& gradient_magnitude);

// Legacy compatibility
float computeCornerScore(const cv::Mat& image_patch, 
                        const cv::Mat& weight_patch,
                        const cv::Vec2f& v1, 
                        const cv::Vec2f& v2);

} // namespace cbdetect 