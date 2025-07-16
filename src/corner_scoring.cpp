#include "cbdetect/corner_scoring.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>

namespace cbdetect {

CornerStatistics computeScoreStatistics(const Corners& corners) {
    if (corners.empty()) {
        return CornerStatistics{};
    }
    
    std::vector<double> scores;
    scores.reserve(corners.size());
    for (const auto& corner : corners) {
        scores.push_back(corner.quality_score);
    }
    
    // Sort scores for percentile calculation
    std::sort(scores.begin(), scores.end());
    
    CornerStatistics stats;
    
    // Mean
    stats.mean = std::accumulate(scores.begin(), scores.end(), 0.0) / scores.size();
    
    // Standard deviation
    double variance = 0.0;
    for (double score : scores) {
        variance += (score - stats.mean) * (score - stats.mean);
    }
    stats.std = std::sqrt(variance / scores.size());
    
    // Median
    size_t mid = scores.size() / 2;
    if (scores.size() % 2 == 0) {
        stats.median = (scores[mid - 1] + scores[mid]) / 2.0;
    } else {
        stats.median = scores[mid];
    }
    
    // Percentiles
    stats.percentile_95 = scores[static_cast<size_t>(scores.size() * 0.95)];
    stats.percentile_99 = scores[static_cast<size_t>(scores.size() * 0.99)];
    
    return stats;
}

void spatialNonMaximumSuppression(Corners& corners, double min_distance) {
    if (corners.empty()) return;
    
    // Sort by score (highest first)
    std::sort(corners.begin(), corners.end(), 
              [](const Corner& a, const Corner& b) { return a.quality_score > b.quality_score; });
    
    Corners filtered_corners;
    
    for (const auto& candidate : corners) {
        bool too_close = false;
        
        // Check distance to all accepted corners
        for (const auto& accepted : filtered_corners) {
            double dx = candidate.pt.x - accepted.pt.x;
            double dy = candidate.pt.y - accepted.pt.y;
            double distance = std::sqrt(dx * dx + dy * dy);
            
            if (distance < min_distance) {
                too_close = true;
                break;
            }
        }
        
        if (!too_close) {
            filtered_corners.push_back(candidate);
        }
    }
    
    corners = std::move(filtered_corners);
}

Corners filterByQuality(const Corners& corners, double quality_threshold) {
    Corners filtered_corners;
    
    for (const auto& corner : corners) {
        if (corner.quality_score >= quality_threshold) {
            filtered_corners.push_back(corner);
        }
    }
    
    return filtered_corners;
}

Corners filterByStatistics(const Corners& corners, const CornerStatistics& stats) {
    // Adaptive threshold based on statistics
    // Use mean + 1.5*std to filter out low-quality corners
    double adaptive_threshold = stats.mean + 1.5 * stats.std;
    
    // But ensure we don't filter too strictly - use at least median
    adaptive_threshold = std::max(adaptive_threshold, stats.median);
    
    return filterByQuality(corners, adaptive_threshold);
}

Corners filterByTopPercentage(const Corners& corners, double percentage) {
    if (corners.empty()) return corners;
    
    Corners sorted_corners = corners;
    
    // Sort by score (highest first)
    std::sort(sorted_corners.begin(), sorted_corners.end(),
              [](const Corner& a, const Corner& b) { return a.quality_score > b.quality_score; });
    
    // Keep only top percentage
    size_t keep_count = static_cast<size_t>(corners.size() * percentage);
    keep_count = std::max(keep_count, size_t(1));  // Keep at least 1 corner
    
    // Create result with only top corners
    Corners result_corners;
    for (size_t i = 0; i < keep_count && i < sorted_corners.size(); ++i) {
        result_corners.push_back(sorted_corners[i]);
    }
    
    return result_corners;
}

void filterCorners(Corners& corners, const DetectionParams& params) {
    if (corners.empty()) {
        std::cout << "No corners to filter" << std::endl;
        return;
    }
    
    size_t initial_count = corners.size();
    std::cout << "Multi-stage corner filtering:" << std::endl;
    std::cout << "  Initial corners: " << initial_count << std::endl;
    
    // Stage 1: Compute statistics
    auto stats = computeScoreStatistics(corners);
    std::cout << "  Score statistics:" << std::endl;
    std::cout << "    Mean: " << stats.mean << ", Std: " << stats.std << std::endl;
    std::cout << "    Median: " << stats.median << std::endl;
    std::cout << "    95th percentile: " << stats.percentile_95 << std::endl;
    
    // Stage 2: Statistical filtering (remove obvious outliers)
    corners = filterByStatistics(corners, stats);
    std::cout << "  After statistical filter: " << corners.size() 
              << " (" << (100.0 * corners.size() / initial_count) << "%)" << std::endl;
    
    // Stage 3: Spatial distribution filtering (minimum 6 pixels apart)
    spatialNonMaximumSuppression(corners, 6.0);
    std::cout << "  After spatial filter (6px): " << corners.size() 
              << " (" << (100.0 * corners.size() / initial_count) << "%)" << std::endl;
    
    // Stage 4: Keep top corners for structure recovery (target: 30-50 corners)
    size_t target_count = std::max(30, static_cast<int>(initial_count * 0.05));  // At least 30 corners or 5%
    target_count = std::min(target_count, static_cast<size_t>(initial_count * 0.15));  // But not more than 15%
    target_count = std::min(target_count, corners.size());  // But not more than available
    
    // Sort by score and keep top N corners
    std::sort(corners.corners.begin(), corners.corners.end(),
              [](const Corner& a, const Corner& b) { return a.quality_score > b.quality_score; });
    
    Corners filtered_final;
    for (size_t i = 0; i < target_count; ++i) {
        filtered_final.push_back(corners[i]);
    }
    corners = filtered_final;
    
    double final_filter_rate = 100.0 * corners.size() / initial_count;
    std::cout << "  After top " << target_count << " filter: " << corners.size() 
              << " (" << final_filter_rate << "%)" << std::endl;
    
    // Target: 85-95% filter rate (keep 5-15% for structure recovery)
    if (final_filter_rate > 15.0) {
        std::cout << "  WARNING: Filter rate " << final_filter_rate 
                  << "% is higher than target 5-15%" << std::endl;
        std::cout << "           Consider more strict filtering parameters" << std::endl;
    } else if (final_filter_rate < 2.0) {
        std::cout << "  WARNING: Filter rate " << final_filter_rate 
                  << "% is too strict, may lose good corners" << std::endl;
    } else {
        std::cout << "  SUCCESS: Achieved target filter rate (" << (100 - final_filter_rate) 
                  << "% filtered out, suitable for structure recovery)" << std::endl;
    }
}

void scoreCorners(Corners& corners, const cv::Mat& image, const cv::Mat& gradient_magnitude) {
    // Enhanced corner scoring using gradient × intensity correlation
    for (auto& corner : corners) {
        // Get local patch around corner
        int radius = 8;
        int x = static_cast<int>(std::round(corner.pt.x));
        int y = static_cast<int>(std::round(corner.pt.y));
        
        if (x - radius < 0 || x + radius >= image.cols || 
            y - radius < 0 || y + radius >= image.rows) {
            corner.quality_score = 0.0;
            continue;
        }
        
        cv::Rect patch_rect(x - radius, y - radius, 2 * radius + 1, 2 * radius + 1);
        cv::Mat image_patch = image(patch_rect);
        cv::Mat gradient_patch = gradient_magnitude(patch_rect);
        
        // Compute contrast-based score
        cv::Scalar mean_intensity, std_intensity;
        cv::meanStdDev(image_patch, mean_intensity, std_intensity);
        double contrast_score = std_intensity[0];  // Standard deviation as contrast measure
        
        // Compute gradient strength score
        cv::Scalar mean_gradient = cv::mean(gradient_patch);
        double gradient_score = mean_gradient[0];
        
        // Combine scores (gradient strength × contrast)
        corner.quality_score = gradient_score * contrast_score;
        
        // Normalize by corner direction consistency (if available)
        if (corner.v1[0] != 0 || corner.v1[1] != 0) {
            // Add direction consistency bonus
            double v1_length = std::sqrt(corner.v1[0] * corner.v1[0] + corner.v1[1] * corner.v1[1]);
            double v2_length = std::sqrt(corner.v2[0] * corner.v2[0] + corner.v2[1] * corner.v2[1]);
            if (v1_length > 0.1 && v2_length > 0.1) {
                // Reward strong directional components
                corner.quality_score *= (1.0 + 0.5 * std::min(v1_length, v2_length));
            }
        }
    }
    
    std::cout << "Corner scoring completed" << std::endl;
}

} // namespace cbdetect 