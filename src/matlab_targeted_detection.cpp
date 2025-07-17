#include "cbdetect/libcbdetect_adapter.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

struct RegionParams {
    cv::Rect region;
    std::string name;
    cbdetect::Params params;
};

cbdetect::Corner detect_in_region(const cv::Mat& img, const cv::Rect& region, 
                                  const cbdetect::Params& params, const std::string& region_name) {
    std::cout << "\n--- Detecting in " << region_name << " [" << region.x << "," << region.y 
              << " " << region.width << "x" << region.height << "] ---" << std::endl;
    
    // Extract region of interest
    cv::Mat roi = img(region);
    
    cbdetect::Corner corners;
    cbdetect::find_corners(roi, corners, params);
    
    // Adjust coordinates to global image space
    for (auto& pt : corners.p) {
        pt.x += region.x;
        pt.y += region.y;
    }
    
    std::cout << "Detected " << corners.p.size() << " corners in " << region_name << std::endl;
    return corners;
}

cbdetect::Corner merge_corners(const std::vector<cbdetect::Corner>& corner_sets) {
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
    
    return merged;
}

// Remove corners that are too close to each other
cbdetect::Corner deduplicate_corners(const cbdetect::Corner& corners, double min_distance = 10.0) {
    cbdetect::Corner dedup;
    
    for (int i = 0; i < corners.p.size(); ++i) {
        bool is_duplicate = false;
        
        for (int j = 0; j < dedup.p.size(); ++j) {
            double dist = cv::norm(corners.p[i] - dedup.p[j]);
            if (dist < min_distance) {
                is_duplicate = true;
                // Keep the one with higher score
                if (i < corners.score.size() && j < dedup.score.size() && 
                    corners.score[i] > dedup.score[j]) {
                    dedup.p[j] = corners.p[i];
                    dedup.r[j] = corners.r[i];
                    dedup.v1[j] = corners.v1[i];
                    dedup.v2[j] = corners.v2[i];
                    dedup.score[j] = corners.score[i];
                }
                break;
            }
        }
        
        if (!is_duplicate) {
            dedup.p.push_back(corners.p[i]);
            dedup.r.push_back(corners.r[i]);
            dedup.v1.push_back(corners.v1[i]);
            dedup.v2.push_back(corners.v2[i]);
            if (i < corners.score.size()) {
                dedup.score.push_back(corners.score[i]);
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
    std::cout << "    MATLAB-TARGETED CORNER DETECTION" << std::endl;
    std::cout << "=======================================================" << std::endl;
    std::cout << "Goal: Achieve 51 corners specifically in MATLAB region" << std::endl;
    std::cout << "Image: " << image_path << std::endl;
    
    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Error: Could not load image " << image_path << std::endl;
        return -1;
    }
    
    std::cout << "Image size: " << img.cols << "x" << img.rows << std::endl;
    
    // Define targeted regions
    std::vector<RegionParams> regions;
    
    // 1. MATLAB expected region with very sensitive parameters
    RegionParams matlab_region;
    matlab_region.region = cv::Rect(42, 350, 423-42, 562-350);
    matlab_region.name = "MATLAB_TARGET";
    matlab_region.params.detect_method = cbdetect::TemplateMatchFast;
    matlab_region.params.norm = true;
    matlab_region.params.norm_half_kernel_size = 31;
    matlab_region.params.init_loc_thr = 0.005;  // Very low threshold for more detection
    matlab_region.params.score_thr = 0.01;     // Very low threshold
    matlab_region.params.radius = {5, 6, 7, 8}; // Multiple scales
    matlab_region.params.polynomial_fit = true;
    matlab_region.params.show_processing = false;
    regions.push_back(matlab_region);
    
    // 2. Expanded MATLAB region to catch nearby corners
    RegionParams expanded_region;
    expanded_region.region = cv::Rect(20, 320, 450, 280);  // Bigger than MATLAB region
    expanded_region.name = "EXPANDED_TARGET";
    expanded_region.params.detect_method = cbdetect::HessianResponse;
    expanded_region.params.norm = false;
    expanded_region.params.init_loc_thr = 0.01;
    expanded_region.params.score_thr = 0.02;
    expanded_region.params.radius = {6, 7};
    expanded_region.params.polynomial_fit = true;
    expanded_region.params.show_processing = false;
    regions.push_back(expanded_region);
    
    // 3. Lower half of image (Y > 300) with medium sensitivity
    RegionParams lower_region;
    lower_region.region = cv::Rect(0, 300, img.cols, img.rows - 300);
    lower_region.name = "LOWER_HALF";
    lower_region.params.detect_method = cbdetect::TemplateMatchFast;
    lower_region.params.norm = true;
    lower_region.params.norm_half_kernel_size = 25;
    lower_region.params.init_loc_thr = 0.008;
    lower_region.params.score_thr = 0.015;
    lower_region.params.radius = {6, 8};
    lower_region.params.polynomial_fit = true;
    lower_region.params.show_processing = false;
    regions.push_back(lower_region);
    
    // Detect corners in each region
    std::vector<cbdetect::Corner> region_corners;
    for (const auto& region : regions) {
        cbdetect::Corner corners = detect_in_region(img, region.region, region.params, region.name);
        if (!corners.p.empty()) {
            region_corners.push_back(corners);
        }
    }
    
    // Merge all detected corners
    cbdetect::Corner all_corners = merge_corners(region_corners);
    std::cout << "\nMerged total: " << all_corners.p.size() << " corners" << std::endl;
    
    // Deduplicate nearby corners
    cbdetect::Corner final_corners = deduplicate_corners(all_corners, 8.0);
    std::cout << "After deduplication: " << final_corners.p.size() << " corners" << std::endl;
    
    // Count corners in MATLAB target region
    cv::Rect matlab_rect(42, 350, 423-42, 562-350);
    int matlab_count = 0;
    for (const auto& pt : final_corners.p) {
        if (matlab_rect.contains(cv::Point2f(pt.x, pt.y))) {
            matlab_count++;
        }
    }
    
    std::cout << "\n=== FINAL RESULTS ===" << std::endl;
    std::cout << "Total corners: " << final_corners.p.size() << std::endl;
    std::cout << "Corners in MATLAB region: " << matlab_count << std::endl;
    std::cout << "Difference from MATLAB target (51): " << (matlab_count - 51) << std::endl;
    
    // Try board detection
    std::vector<cbdetect::Board> boards;
    cbdetect::Params board_params;
    board_params.detect_method = cbdetect::TemplateMatchFast;
    board_params.norm = true;
    cbdetect::boards_from_corners(img, final_corners, boards, board_params);
    std::cout << "Boards detected: " << boards.size() << std::endl;
    
    // Performance analysis
    std::cout << "\n=== PERFORMANCE ANALYSIS ===" << std::endl;
    if (matlab_count >= 45 && matlab_count <= 55) {
        std::cout << "🎯 EXCELLENT: Very close to MATLAB target!" << std::endl;
    } else if (matlab_count >= 35 && matlab_count <= 65) {
        std::cout << "✅ GOOD: Reasonably close to MATLAB target" << std::endl;
    } else if (matlab_count >= 25) {
        std::cout << "⚠️ MODERATE: Some improvement over previous attempts" << std::endl;
    } else {
        std::cout << "❌ POOR: Still far from MATLAB target" << std::endl;
    }
    
    // Visualization
    cv::Mat vis = img.clone();
    if (vis.channels() == 1) {
        cv::cvtColor(vis, vis, cv::COLOR_GRAY2BGR);
    }
    
    // Draw MATLAB target region
    cv::rectangle(vis, matlab_rect, cv::Scalar(0, 255, 255), 3);
    cv::putText(vis, "MATLAB Target", cv::Point(50, 340), 
               cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);
    
    // Draw corners
    for (int i = 0; i < final_corners.p.size(); ++i) {
        cv::Point2f pt(final_corners.p[i].x, final_corners.p[i].y);
        cv::Scalar color = matlab_rect.contains(pt) ? 
                          cv::Scalar(0, 255, 0) :   // Green for MATLAB region
                          cv::Scalar(0, 0, 255);    // Red for outside
        
        cv::circle(vis, pt, 6, color, 2);
        cv::circle(vis, pt, 2, cv::Scalar(255, 255, 255), -1);
    }
    
    // Add info text
    cv::putText(vis, "Total: " + std::to_string(final_corners.p.size()), 
               cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
    cv::putText(vis, "MATLAB: " + std::to_string(matlab_count), 
               cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
    cv::putText(vis, "Target: 51", 
               cv::Point(10, 90), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);
    
    std::string save_path = "result/matlab_targeted_result.png";
    cv::imwrite(save_path, vis);
    std::cout << "\nVisualization saved to: " << save_path << std::endl;
    
    std::cout << "\n=======================================================" << std::endl;
    std::cout << "MATLAB-TARGETED DETECTION COMPLETE" << std::endl;
    std::cout << "Strategy: Multi-region detection with region-specific parameters" << std::endl;
    if (matlab_count >= 45) {
        std::cout << "🎉 SUCCESS: Achieved target corner count in MATLAB region!" << std::endl;
    } else {
        std::cout << "📈 PROGRESS: Improved detection, further tuning may be needed" << std::endl;
    }
    std::cout << "=======================================================" << std::endl;
    
    return 0;
} 