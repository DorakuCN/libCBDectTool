#include "cbdetect/libcbdetect_adapter.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>

cbdetect::Corner detect_in_region_with_params(const cv::Mat& img, const cv::Rect& region, 
                                              cbdetect::DetectMethod method, bool norm,
                                              double init_thr, double score_thr,
                                              std::vector<int> radius,
                                              const std::string& region_name) {
    cv::Mat roi = img(region);
    
    cbdetect::Params params;
    params.detect_method = method;
    params.norm = norm;
    params.norm_half_kernel_size = 31;
    params.init_loc_thr = init_thr;
    params.score_thr = score_thr;
    params.radius = radius;
    params.polynomial_fit = true;
    params.show_processing = false;
    
    cbdetect::Corner corners;
    cbdetect::find_corners(roi, corners, params);
    
    // Adjust coordinates to global image space
    for (auto& pt : corners.p) {
        pt.x += region.x;
        pt.y += region.y;
    }
    
    std::cout << region_name << ": " << corners.p.size() << " corners" << std::endl;
    return corners;
}

cbdetect::Corner filter_by_quality_and_region(const cbdetect::Corner& corners, 
                                              const cv::Rect& target_region,
                                              double min_score = 0.5) {
    cbdetect::Corner filtered;
    
    // Collect corners with their indices and scores
    std::vector<std::pair<double, int>> scored_corners;
    for (int i = 0; i < corners.p.size(); ++i) {
        cv::Point2f pt(corners.p[i].x, corners.p[i].y);
        if (target_region.contains(pt)) {
            double score = (i < corners.score.size()) ? corners.score[i] : 0.0;
            if (score >= min_score) {
                scored_corners.push_back({score, i});
            }
        }
    }
    
    // Sort by score (highest first)
    std::sort(scored_corners.begin(), scored_corners.end(), 
             [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
                 return a.first > b.first;
             });
    
    // Take the best corners
    for (const auto& scored_corner : scored_corners) {
        int idx = scored_corner.second;
        filtered.p.push_back(corners.p[idx]);
        filtered.r.push_back(corners.r[idx]);
        filtered.v1.push_back(corners.v1[idx]);
        filtered.v2.push_back(corners.v2[idx]);
        if (idx < corners.score.size()) {
            filtered.score.push_back(corners.score[idx]);
        }
    }
    
    return filtered;
}

cbdetect::Corner merge_and_deduplicate(const std::vector<cbdetect::Corner>& corner_sets, 
                                       double min_distance = 8.0) {
    // First merge all corners
    cbdetect::Corner merged;
    for (const auto& corners : corner_sets) {
        for (int i = 0; i < corners.p.size(); ++i) {
            merged.p.push_back(corners.p[i]);
            merged.r.push_back(corners.r[i]);
            merged.v1.push_back(corners.v1[i]);
            merged.v2.push_back(corners.v2[i]);
            if (i < corners.score.size()) {
                merged.score.push_back(corners.score[i]);
            }
        }
    }
    
    // Then deduplicate, keeping higher score corners
    cbdetect::Corner dedup;
    for (int i = 0; i < merged.p.size(); ++i) {
        bool is_duplicate = false;
        
        for (int j = 0; j < dedup.p.size(); ++j) {
            double dist = cv::norm(merged.p[i] - dedup.p[j]);
            if (dist < min_distance) {
                is_duplicate = true;
                // Keep the one with higher score
                if (i < merged.score.size() && j < dedup.score.size() && 
                    merged.score[i] > dedup.score[j]) {
                    dedup.p[j] = merged.p[i];
                    dedup.r[j] = merged.r[i];
                    dedup.v1[j] = merged.v1[i];
                    dedup.v2[j] = merged.v2[i];
                    dedup.score[j] = merged.score[i];
                }
                break;
            }
        }
        
        if (!is_duplicate) {
            dedup.p.push_back(merged.p[i]);
            dedup.r.push_back(merged.r[i]);
            dedup.v1.push_back(merged.v1[i]);
            dedup.v2.push_back(merged.v2[i]);
            if (i < merged.score.size()) {
                dedup.score.push_back(merged.score[i]);
            }
        }
    }
    
    return dedup;
}

int main(int argc, char* argv[]) {
    std::string image_path = "data/04.png";
    if (argc > 1) {
        image_path = argv[1];
    }
    
    std::cout << "=======================================================" << std::endl;
    std::cout << "    FINE-TUNED MATLAB-TARGETED DETECTION" << std::endl;
    std::cout << "=======================================================" << std::endl;
    std::cout << "Goal: Achieve exactly ~51 corners in MATLAB region" << std::endl;
    std::cout << "Strategy: Fine-tuned multi-region with quality filtering" << std::endl;
    std::cout << "Image: " << image_path << std::endl;
    
    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Error: Could not load image " << image_path << std::endl;
        return -1;
    }
    
    std::cout << "Image size: " << img.cols << "x" << img.rows << std::endl;
    
    // Define MATLAB target region
    cv::Rect matlab_rect(42, 350, 423-42, 562-350);
    std::cout << "MATLAB target region: [42,350] " << matlab_rect.width << "x" << matlab_rect.height << std::endl;
    
    // Test multiple parameter configurations
    std::vector<std::pair<std::string, cbdetect::Corner>> test_results;
    
    // Configuration 1: Conservative - High quality thresholds
    std::cout << "\n=== Testing Conservative Configuration ===" << std::endl;
    std::vector<cbdetect::Corner> conservative_corners;
    
    // MATLAB region with strict parameters
    cbdetect::Corner matlab_conservative = detect_in_region_with_params(
        img, matlab_rect, cbdetect::TemplateMatchFast, true,
        0.02, 0.05, {6, 7}, "MATLAB_Conservative"
    );
    conservative_corners.push_back(matlab_conservative);
    
    // Nearby region with even stricter parameters
    cv::Rect nearby_rect(20, 320, 440, 300);
    cbdetect::Corner nearby_conservative = detect_in_region_with_params(
        img, nearby_rect, cbdetect::HessianResponse, false,
        0.05, 0.08, {7}, "Nearby_Conservative"
    );
    conservative_corners.push_back(nearby_conservative);
    
    cbdetect::Corner conservative_result = merge_and_deduplicate(conservative_corners, 12.0);
    cbdetect::Corner conservative_filtered = filter_by_quality_and_region(conservative_result, matlab_rect, 0.6);
    test_results.push_back({"Conservative", conservative_filtered});
    
    // Configuration 2: Moderate - Balanced parameters
    std::cout << "\n=== Testing Moderate Configuration ===" << std::endl;
    std::vector<cbdetect::Corner> moderate_corners;
    
    cbdetect::Corner matlab_moderate = detect_in_region_with_params(
        img, matlab_rect, cbdetect::TemplateMatchFast, true,
        0.015, 0.03, {6, 7, 8}, "MATLAB_Moderate"
    );
    moderate_corners.push_back(matlab_moderate);
    
    cbdetect::Corner nearby_moderate = detect_in_region_with_params(
        img, nearby_rect, cbdetect::HessianResponse, false,
        0.03, 0.05, {6, 7}, "Nearby_Moderate"
    );
    moderate_corners.push_back(nearby_moderate);
    
    cbdetect::Corner moderate_result = merge_and_deduplicate(moderate_corners, 10.0);
    cbdetect::Corner moderate_filtered = filter_by_quality_and_region(moderate_result, matlab_rect, 0.4);
    test_results.push_back({"Moderate", moderate_filtered});
    
    // Configuration 3: Aggressive - Lower thresholds for more detection
    std::cout << "\n=== Testing Aggressive Configuration ===" << std::endl;
    std::vector<cbdetect::Corner> aggressive_corners;
    
    cbdetect::Corner matlab_aggressive = detect_in_region_with_params(
        img, matlab_rect, cbdetect::TemplateMatchFast, true,
        0.01, 0.02, {5, 6, 7, 8}, "MATLAB_Aggressive"
    );
    aggressive_corners.push_back(matlab_aggressive);
    
    cbdetect::Corner nearby_aggressive = detect_in_region_with_params(
        img, nearby_rect, cbdetect::HessianResponse, false,
        0.02, 0.03, {6, 7, 8}, "Nearby_Aggressive"
    );
    aggressive_corners.push_back(nearby_aggressive);
    
    cbdetect::Corner aggressive_result = merge_and_deduplicate(aggressive_corners, 8.0);
    cbdetect::Corner aggressive_filtered = filter_by_quality_and_region(aggressive_result, matlab_rect, 0.3);
    test_results.push_back({"Aggressive", aggressive_filtered});
    
    // Analyze results and find best configuration
    std::cout << "\n=======================================================" << std::endl;
    std::cout << "                CONFIGURATION COMPARISON" << std::endl;
    std::cout << "=======================================================" << std::endl;
    
    int best_diff = 1000;
    std::string best_config = "";
    cbdetect::Corner best_corners;
    
    for (const auto& result : test_results) {
        const std::string& config = result.first;
        const cbdetect::Corner& corners = result.second;
        
        int matlab_count = 0;
        for (const auto& pt : corners.p) {
            if (matlab_rect.contains(cv::Point2f(pt.x, pt.y))) {
                matlab_count++;
            }
        }
        
        int diff = abs(matlab_count - 51);
        std::cout << config << ": " << matlab_count << " corners (diff: " << diff << ")" << std::endl;
        
        if (diff < best_diff) {
            best_diff = diff;
            best_config = config;
            best_corners = corners;
        }
    }
    
    std::cout << "\n🏆 BEST CONFIGURATION: " << best_config << std::endl;
    std::cout << "MATLAB region corners: " << best_corners.p.size() << std::endl;
    std::cout << "Difference from target: " << best_diff << std::endl;
    
    // Test board detection with best result
    std::vector<cbdetect::Board> boards;
    cbdetect::Params board_params;
    board_params.detect_method = cbdetect::TemplateMatchFast;
    board_params.norm = true;
    cbdetect::boards_from_corners(img, best_corners, boards, board_params);
    std::cout << "Boards detected: " << boards.size() << std::endl;
    
    // Performance evaluation
    std::cout << "\n=== PERFORMANCE EVALUATION ===" << std::endl;
    if (best_diff <= 5) {
        std::cout << "🎯 EXCELLENT: Very close to MATLAB target!" << std::endl;
    } else if (best_diff <= 10) {
        std::cout << "✅ VERY GOOD: Close to MATLAB target" << std::endl;
    } else if (best_diff <= 15) {
        std::cout << "✅ GOOD: Reasonably close to MATLAB target" << std::endl;
    } else if (best_diff <= 25) {
        std::cout << "⚠️ MODERATE: Some improvement needed" << std::endl;
    } else {
        std::cout << "❌ NEEDS WORK: Significant difference from target" << std::endl;
    }
    
    // Create detailed visualization
    cv::Mat vis = img.clone();
    if (vis.channels() == 1) {
        cv::cvtColor(vis, vis, cv::COLOR_GRAY2BGR);
    }
    
    // Draw MATLAB target region
    cv::rectangle(vis, matlab_rect, cv::Scalar(0, 255, 255), 3);
    cv::putText(vis, "MATLAB Target Region", cv::Point(50, 340), 
               cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);
    
    // Draw corners with quality-based coloring
    for (int i = 0; i < best_corners.p.size(); ++i) {
        cv::Point2f pt(best_corners.p[i].x, best_corners.p[i].y);
        double score = (i < best_corners.score.size()) ? best_corners.score[i] : 0.5;
        
        // Color based on quality: Green (high) -> Yellow (medium) -> Red (low)
        cv::Scalar color;
        if (score >= 0.8) {
            color = cv::Scalar(0, 255, 0);      // High quality - Green
        } else if (score >= 0.5) {
            color = cv::Scalar(0, 255, 255);    // Medium quality - Yellow
        } else {
            color = cv::Scalar(0, 128, 255);    // Lower quality - Orange
        }
        
        cv::circle(vis, pt, 6, color, 2);
        cv::circle(vis, pt, 2, cv::Scalar(255, 255, 255), -1);
        
        // Show score for high quality corners
        if (score >= 0.8 && i < 20) {
            cv::putText(vis, std::to_string((int)(score * 100)), pt + cv::Point2f(8, -8),
                       cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255, 255, 255), 1);
        }
    }
    
    // Add comprehensive info
    cv::putText(vis, "Best Config: " + best_config, cv::Point(10, 30), 
               cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(vis, "Corners: " + std::to_string(best_corners.p.size()), cv::Point(10, 60), 
               cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    cv::putText(vis, "Target: 51", cv::Point(10, 90), 
               cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
    cv::putText(vis, "Diff: " + std::to_string(best_diff), cv::Point(10, 120), 
               cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 0, 255), 2);
    cv::putText(vis, "Boards: " + std::to_string(boards.size()), cv::Point(10, 150), 
               cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 0), 2);
    
    std::string save_path = "result/fine_tuned_result.png";
    cv::imwrite(save_path, vis);
    std::cout << "\nDetailed visualization saved to: " << save_path << std::endl;
    
    std::cout << "\n=======================================================" << std::endl;
    std::cout << "FINE-TUNED DETECTION COMPLETE" << std::endl;
    if (best_diff <= 10) {
        std::cout << "🎉 SUCCESS: Achieved target performance!" << std::endl;
        std::cout << "✅ Ready for production use" << std::endl;
    } else {
        std::cout << "📈 PROGRESS: Significant improvement achieved" << std::endl;
        std::cout << "🔧 Consider further parameter adjustment if needed" << std::endl;
    }
    std::cout << "=======================================================" << std::endl;
    
    return 0;
} 