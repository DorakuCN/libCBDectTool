#include "cbdetect/libcbdetect_adapter.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>

void visualize_regions_and_corners(const cv::Mat& img, const cbdetect::Corner& corners, 
                                  const std::string& title) {
    cv::Mat vis = img.clone();
    if (vis.channels() == 1) {
        cv::cvtColor(vis, vis, cv::COLOR_GRAY2BGR);
    }
    
    // Draw MATLAB expected region rectangle
    cv::Rect matlab_region(42, 350, 423-42, 562-350);
    cv::rectangle(vis, matlab_region, cv::Scalar(0, 255, 255), 3);  // Yellow rectangle
    cv::putText(vis, "MATLAB Expected Region", cv::Point(50, 340), 
               cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);
    
    // Draw region dividers
    cv::line(vis, cv::Point(0, 200), cv::Point(img.cols, 200), cv::Scalar(255, 0, 0), 2);  // Blue line
    cv::line(vis, cv::Point(0, 350), cv::Point(img.cols, 350), cv::Scalar(255, 0, 0), 2);  // Blue line
    cv::putText(vis, "Y=200", cv::Point(10, 195), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 0, 0), 2);
    cv::putText(vis, "Y=350", cv::Point(10, 345), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 0, 0), 2);
    
    // Draw corners with different colors based on region
    for (int i = 0; i < corners.p.size(); ++i) {
        cv::Point2f pt(corners.p[i].x, corners.p[i].y);
        cv::Scalar color;
        std::string region_label;
        
        if (pt.x >= 42 && pt.x <= 423 && pt.y >= 350 && pt.y <= 562) {
            color = cv::Scalar(0, 255, 0);  // Green for MATLAB region
            region_label = "M";
        } else if (pt.y < 200) {
            color = cv::Scalar(0, 0, 255);  // Red for upper region (likely false positives)
            region_label = "U";
        } else if (pt.y >= 200 && pt.y < 350) {
            color = cv::Scalar(255, 255, 0);  // Cyan for middle region
            region_label = "Mid";
        } else {
            color = cv::Scalar(255, 0, 255);  // Magenta for lower region outside MATLAB
            region_label = "L";
        }
        
        cv::circle(vis, pt, 6, color, 2);
        cv::circle(vis, pt, 2, cv::Scalar(255, 255, 255), -1);
        
        // Show region label for first 20 corners
        if (i < 20) {
            cv::putText(vis, region_label + std::to_string(i), pt + cv::Point2f(8, -8),
                       cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
        }
    }
    
    // Add legend
    cv::putText(vis, "Green: MATLAB Region", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
    cv::putText(vis, "Red: Upper (False Pos?)", cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
    cv::putText(vis, "Cyan: Middle", cv::Point(10, 90), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 0), 2);
    cv::putText(vis, "Magenta: Lower Other", cv::Point(10, 120), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 0, 255), 2);
    cv::putText(vis, "Total: " + std::to_string(corners.p.size()), cv::Point(10, 150), 
               cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
    
    cv::imshow(title, vis);
    cv::waitKey(2000);  // Show for 2 seconds
    
    std::string save_path = "result/" + title + ".png";
    cv::imwrite(save_path, vis);
    std::cout << "Saved region analysis to: " << save_path << std::endl;
}

cbdetect::Corner filter_corners_by_region(const cbdetect::Corner& input_corners, 
                                          const std::string& filter_type = "matlab_only") {
    cbdetect::Corner filtered;
    
    for (int i = 0; i < input_corners.p.size(); ++i) {
        const auto& pt = input_corners.p[i];
        bool keep = false;
        
        if (filter_type == "matlab_only") {
            // Keep only corners in MATLAB expected region
            keep = (pt.x >= 42 && pt.x <= 423 && pt.y >= 350 && pt.y <= 562);
        } else if (filter_type == "remove_upper") {
            // Remove upper region false positives
            keep = (pt.y >= 200);  // Keep middle and lower regions
        } else if (filter_type == "high_quality_only") {
            // Keep only high quality corners regardless of region
            double score = (i < input_corners.score.size()) ? input_corners.score[i] : 0.0;
            keep = (score >= 1.3);  // High threshold
        }
        
        if (keep) {
            filtered.p.push_back(input_corners.p[i]);
            filtered.r.push_back(input_corners.r[i]);
            filtered.v1.push_back(input_corners.v1[i]);
            filtered.v2.push_back(input_corners.v2[i]);
            if (i < input_corners.score.size()) {
                filtered.score.push_back(input_corners.score[i]);
            }
        }
    }
    
    return filtered;
}

int main(int argc, char* argv[]) {
    std::string image_path = "data/04.png";
    if (argc > 1) {
        image_path = argv[1];
    }
    
    std::cout << "=======================================================" << std::endl;
    std::cout << "    REGION-FOCUSED CORNER DETECTION ANALYSIS" << std::endl;
    std::cout << "=======================================================" << std::endl;
    std::cout << "Goal: Focus detection on MATLAB expected region" << std::endl;
    std::cout << "Image: " << image_path << std::endl;
    
    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Error: Could not load image " << image_path << std::endl;
        return -1;
    }
    
    std::cout << "Image size: " << img.cols << "x" << img.rows << std::endl;
    std::cout << "MATLAB expected region: X[42-423] Y[350-562]" << std::endl;
    
    // Test our best configuration
    cbdetect::Params best_config;
    best_config.detect_method = cbdetect::TemplateMatchFast;
    best_config.show_processing = false;
    best_config.norm = true;
    best_config.norm_half_kernel_size = 31;
    best_config.init_loc_thr = 0.012;
    best_config.score_thr = 0.025;
    best_config.radius = {6, 8};
    best_config.polynomial_fit = true;
    
    cbdetect::Corner all_corners;
    std::vector<cbdetect::Board> boards;
    cbdetect::find_corners(img, all_corners, best_config);
    cbdetect::boards_from_corners(img, all_corners, boards, best_config);
    
    std::cout << "\n=== ORIGINAL DETECTION RESULTS ===" << std::endl;
    std::cout << "Total corners detected: " << all_corners.p.size() << std::endl;
    
    // Count corners in each region
    int matlab_count = 0, upper_count = 0, middle_count = 0, lower_count = 0;
    for (const auto& pt : all_corners.p) {
        if (pt.x >= 42 && pt.x <= 423 && pt.y >= 350 && pt.y <= 562) {
            matlab_count++;
        }
        if (pt.y < 200) upper_count++;
        else if (pt.y >= 200 && pt.y < 350) middle_count++;
        else lower_count++;
    }
    
    std::cout << "Region breakdown:" << std::endl;
    std::cout << "  MATLAB region (target): " << matlab_count << " corners" << std::endl;
    std::cout << "  Upper region (Y<200):   " << upper_count << " corners" << std::endl;
    std::cout << "  Middle region:          " << middle_count << " corners" << std::endl;  
    std::cout << "  Lower region (Y>350):   " << lower_count << " corners" << std::endl;
    
    // Visualize all corners with region highlighting
    visualize_regions_and_corners(img, all_corners, "all_corners_analysis");
    
    // Test filtering strategies
    std::cout << "\n=== FILTERING STRATEGIES ===" << std::endl;
    
    // Strategy 1: Keep only MATLAB region corners
    cbdetect::Corner matlab_only = filter_corners_by_region(all_corners, "matlab_only");
    std::cout << "1. MATLAB region only: " << matlab_only.p.size() << " corners" << std::endl;
    visualize_regions_and_corners(img, matlab_only, "matlab_region_only");
    
    // Strategy 2: Remove upper region (likely false positives)
    cbdetect::Corner no_upper = filter_corners_by_region(all_corners, "remove_upper");
    std::cout << "2. Remove upper region: " << no_upper.p.size() << " corners" << std::endl;
    visualize_regions_and_corners(img, no_upper, "remove_upper_region");
    
    // Strategy 3: High quality corners only
    cbdetect::Corner high_quality = filter_corners_by_region(all_corners, "high_quality_only");
    std::cout << "3. High quality only: " << high_quality.p.size() << " corners" << std::endl;
    visualize_regions_and_corners(img, high_quality, "high_quality_only");
    
    // Analysis and recommendations
    std::cout << "\n=======================================================" << std::endl;
    std::cout << "                    ANALYSIS RESULTS" << std::endl;
    std::cout << "=======================================================" << std::endl;
    
    std::cout << "📊 DETECTION BREAKDOWN:" << std::endl;
    std::cout << "   Total detected: " << all_corners.p.size() << " corners" << std::endl;
    std::cout << "   MATLAB target:  " << matlab_count << " corners (" 
              << (100.0 * matlab_count / all_corners.p.size()) << "%)" << std::endl;
    std::cout << "   False positives: " << (all_corners.p.size() - matlab_count) << " corners (" 
              << (100.0 * (all_corners.p.size() - matlab_count) / all_corners.p.size()) << "%)" << std::endl;
    
    std::cout << "\n🎯 COMPARISON WITH MATLAB TARGET (51 corners):" << std::endl;
    std::cout << "   Strategy 1 (MATLAB only): " << matlab_only.p.size() << " corners (diff: " 
              << (matlab_only.p.size() - 51) << ")" << std::endl;
    std::cout << "   Strategy 2 (No upper):    " << no_upper.p.size() << " corners (diff: " 
              << (no_upper.p.size() - 51) << ")" << std::endl;
    std::cout << "   Strategy 3 (High quality): " << high_quality.p.size() << " corners (diff: " 
              << (high_quality.p.size() - 51) << ")" << std::endl;
    
    // Find best strategy
    int best_diff = abs((int)matlab_only.p.size() - 51);
    std::string best_strategy = "MATLAB region only";
    int best_count = matlab_only.p.size();
    
    if (abs((int)no_upper.p.size() - 51) < best_diff) {
        best_diff = abs((int)no_upper.p.size() - 51);
        best_strategy = "Remove upper region";
        best_count = no_upper.p.size();
    }
    
    if (abs((int)high_quality.p.size() - 51) < best_diff) {
        best_diff = abs((int)high_quality.p.size() - 51);
        best_strategy = "High quality only";
        best_count = high_quality.p.size();
    }
    
    std::cout << "\n🏆 BEST STRATEGY: " << best_strategy << std::endl;
    std::cout << "   Result: " << best_count << " corners (diff: " << (best_count - 51) << " from MATLAB)" << std::endl;
    
    if (best_diff <= 10) {
        std::cout << "✅ EXCELLENT: Very close to MATLAB target!" << std::endl;
        std::cout << "🎉 SOLUTION FOUND: Use filtering strategy to match MATLAB performance" << std::endl;
    } else {
        std::cout << "⚠️ NEEDS IMPROVEMENT: Still differs significantly from MATLAB" << std::endl;
    }
    
    std::cout << "\n💡 RECOMMENDATIONS:" << std::endl;
    std::cout << "1. Use region-based filtering to remove false positives" << std::endl;
    std::cout << "2. Focus algorithm parameters on Y[350-562] region" << std::endl;
    std::cout << "3. Upper region (Y<200) likely contains text/noise artifacts" << std::endl;
    std::cout << "4. Consider quality-based filtering for better precision" << std::endl;
    
    return 0;
} 