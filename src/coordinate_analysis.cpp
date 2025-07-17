#include "cbdetect/libcbdetect_adapter.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>

void analyze_coordinate_distribution(const cbdetect::Corner& corners, const std::string& method_name) {
    std::cout << "\n=== " << method_name << " Coordinate Analysis ===" << std::endl;
    
    if (corners.p.empty()) {
        std::cout << "No corners detected!" << std::endl;
        return;
    }
    
    // Calculate bounds
    double min_x = corners.p[0].x, max_x = corners.p[0].x;
    double min_y = corners.p[0].y, max_y = corners.p[0].y;
    
    for (const auto& pt : corners.p) {
        min_x = std::min(min_x, pt.x);
        max_x = std::max(max_x, pt.x);
        min_y = std::min(min_y, pt.y);
        max_y = std::max(max_y, pt.y);
    }
    
    std::cout << "Total corners: " << corners.p.size() << std::endl;
    std::cout << "Detection bounds: X[" << std::fixed << std::setprecision(1) 
              << min_x << "-" << max_x << "] Y[" << min_y << "-" << max_y << "]" << std::endl;
    
    // Compare with MATLAB expected region
    std::cout << "MATLAB expected:  X[42-423] Y[350-562]" << std::endl;
    
    // Count corners in different regions
    int matlab_region = 0;  // X[42-423] Y[350-562]
    int upper_region = 0;   // Y < 200 (where we're detecting)
    int middle_region = 0;  // Y[200-350]
    int lower_region = 0;   // Y > 350 (MATLAB region)
    
    for (const auto& pt : corners.p) {
        if (pt.x >= 42 && pt.x <= 423 && pt.y >= 350 && pt.y <= 562) {
            matlab_region++;
        }
        
        if (pt.y < 200) {
            upper_region++;
        } else if (pt.y >= 200 && pt.y < 350) {
            middle_region++;
        } else {
            lower_region++;
        }
    }
    
    std::cout << "Region distribution:" << std::endl;
    std::cout << "  Upper region (Y<200):     " << upper_region << " corners (" 
              << (100.0 * upper_region / corners.p.size()) << "%)" << std::endl;
    std::cout << "  Middle region (Y200-350): " << middle_region << " corners (" 
              << (100.0 * middle_region / corners.p.size()) << "%)" << std::endl;
    std::cout << "  Lower region (Y>350):     " << lower_region << " corners (" 
              << (100.0 * lower_region / corners.p.size()) << "%)" << std::endl;
    std::cout << "  MATLAB expected region:   " << matlab_region << " corners (" 
              << (100.0 * matlab_region / corners.p.size()) << "%)" << std::endl;
    
    // Show all corners for detailed analysis
    std::cout << "\nAll detected corners:" << std::endl;
    for (int i = 0; i < corners.p.size(); ++i) {
        std::cout << std::fixed << std::setprecision(1);
        std::cout << "  [" << std::setw(2) << i << "] (" << std::setw(6) << corners.p[i].x 
                  << "," << std::setw(6) << corners.p[i].y << ")";
        
        if (i < corners.score.size()) {
            std::cout << " score=" << std::setprecision(3) << corners.score[i];
        }
        
        // Mark region
        if (corners.p[i].x >= 42 && corners.p[i].x <= 423 && corners.p[i].y >= 350 && corners.p[i].y <= 562) {
            std::cout << " [MATLAB]";
        } else if (corners.p[i].y < 200) {
            std::cout << " [UPPER]";
        } else if (corners.p[i].y >= 200 && corners.p[i].y < 350) {
            std::cout << " [MIDDLE]";
        } else {
            std::cout << " [LOWER]";
        }
        
        std::cout << std::endl;
    }
}

void save_coordinate_comparison(const std::vector<std::pair<std::string, cbdetect::Corner>>& results) {
    std::ofstream report("result/coordinate_analysis.txt");
    
    report << "=======================================================" << std::endl;
    report << "         COORDINATE DISTRIBUTION ANALYSIS" << std::endl;
    report << "=======================================================" << std::endl;
    report << "Problem: Our detection region doesn't match MATLAB" << std::endl;
    report << "MATLAB Expected: X[42-423] Y[350-562]" << std::endl;
    report << "Our Detection:   Mainly in Y[80-140] region" << std::endl;
    report << std::endl;
    
    for (const auto& result : results) {
        const std::string& method = result.first;
        const cbdetect::Corner& corners = result.second;
        
        if (corners.p.empty()) continue;
        
        report << "=== " << method << " ===" << std::endl;
        
        // Calculate bounds
        double min_x = corners.p[0].x, max_x = corners.p[0].x;
        double min_y = corners.p[0].y, max_y = corners.p[0].y;
        
        for (const auto& pt : corners.p) {
            min_x = std::min(min_x, pt.x);
            max_x = std::max(max_x, pt.x);
            min_y = std::min(min_y, pt.y);
            max_y = std::max(max_y, pt.y);
        }
        
        report << "Corners: " << corners.p.size() << std::endl;
        report << "Bounds: X[" << std::fixed << std::setprecision(1) 
               << min_x << "-" << max_x << "] Y[" << min_y << "-" << max_y << "]" << std::endl;
        
        // Region analysis
        int matlab_region = 0, upper_region = 0, middle_region = 0, lower_region = 0;
        for (const auto& pt : corners.p) {
            if (pt.x >= 42 && pt.x <= 423 && pt.y >= 350 && pt.y <= 562) matlab_region++;
            if (pt.y < 200) upper_region++;
            else if (pt.y >= 200 && pt.y < 350) middle_region++;
            else lower_region++;
        }
        
        report << "MATLAB region: " << matlab_region << "/" << corners.p.size() 
               << " (" << (100.0 * matlab_region / corners.p.size()) << "%)" << std::endl;
        report << "Upper region (Y<200): " << upper_region << "/" << corners.p.size() 
               << " (" << (100.0 * upper_region / corners.p.size()) << "%)" << std::endl;
        
        // First 10 corners
        report << "Sample coordinates: ";
        for (int i = 0; i < std::min(10, (int)corners.p.size()); ++i) {
            report << "(" << std::fixed << std::setprecision(0) 
                   << corners.p[i].x << "," << corners.p[i].y << ")";
            if (i < 9 && i < corners.p.size() - 1) report << ", ";
        }
        report << std::endl << std::endl;
    }
    
    // Conclusion
    report << "=== PROBLEM ANALYSIS ===" << std::endl;
    report << "CRITICAL ISSUE: We are detecting corners in the WRONG region!" << std::endl;
    report << "- Our detection focuses on Y[80-140] (image top)" << std::endl;
    report << "- MATLAB expects Y[350-562] (image bottom)" << std::endl;
    report << "- This suggests our algorithm is finding false positives" << std::endl;
    report << "  or the wrong type of features" << std::endl;
    report << std::endl;
    report << "POSSIBLE CAUSES:" << std::endl;
    report << "1. Template matching parameters are incorrect" << std::endl;
    report << "2. Image preprocessing (normalization) is wrong" << std::endl;
    report << "3. We're detecting text/noise instead of chessboard" << std::endl;
    report << "4. Scale or orientation issues" << std::endl;
    report << std::endl;
    report << "NEXT STEPS:" << std::endl;
    report << "1. Visualize what our algorithm is actually detecting" << std::endl;
    report << "2. Compare with actual MATLAB findCorners output" << std::endl;
    report << "3. Check if chessboard is really in Y[350-562] region" << std::endl;
    report << "4. Adjust detection to focus on correct image region" << std::endl;
    
    report.close();
    std::cout << "\nDetailed coordinate analysis saved to: result/coordinate_analysis.txt" << std::endl;
}

int main(int argc, char* argv[]) {
    std::string image_path = "data/04.png";
    if (argc > 1) {
        image_path = argv[1];
    }
    
    std::cout << "=======================================================" << std::endl;
    std::cout << "    COORDINATE DISTRIBUTION ANALYSIS" << std::endl;
    std::cout << "=======================================================" << std::endl;
    std::cout << "Investigating: Why our corners don't match MATLAB region" << std::endl;
    std::cout << "Image: " << image_path << std::endl;
    
    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Error: Could not load image " << image_path << std::endl;
        return -1;
    }
    
    std::cout << "Image size: " << img.cols << "x" << img.rows << std::endl;
    std::cout << "MATLAB expected chessboard region: X[42-423] Y[350-562]" << std::endl;
    
    std::vector<std::pair<std::string, cbdetect::Corner>> results;
    
    // Test 1: Our current best method
    cbdetect::Params best_config;
    best_config.detect_method = cbdetect::TemplateMatchFast;
    best_config.show_processing = false;
    best_config.norm = true;
    best_config.norm_half_kernel_size = 31;
    best_config.init_loc_thr = 0.012;
    best_config.score_thr = 0.025;
    best_config.radius = {6, 8};
    best_config.polynomial_fit = true;
    
    cbdetect::Corner corners_best;
    std::vector<cbdetect::Board> boards;
    cbdetect::find_corners(img, corners_best, best_config);
    cbdetect::boards_from_corners(img, corners_best, boards, best_config);
    
    results.push_back({"Final_Optimized", corners_best});
    analyze_coordinate_distribution(corners_best, "Final_Optimized");
    
    // Test 2: HessianResponse with very high threshold to see real corners
    cbdetect::Params hessian_strict;
    hessian_strict.detect_method = cbdetect::HessianResponse;
    hessian_strict.show_processing = false;
    hessian_strict.norm = false;
    hessian_strict.init_loc_thr = 0.1;   // Very high threshold
    hessian_strict.score_thr = 0.1;
    hessian_strict.radius = {7};
    
    cbdetect::Corner corners_hessian;
    cbdetect::find_corners(img, corners_hessian, hessian_strict);
    
    results.push_back({"HessianResponse_Strict", corners_hessian});
    analyze_coordinate_distribution(corners_hessian, "HessianResponse_Strict");
    
    // Test 3: TemplateMatch without normalization 
    cbdetect::Params template_raw;
    template_raw.detect_method = cbdetect::TemplateMatchFast;
    template_raw.show_processing = false;
    template_raw.norm = false;  // No normalization
    template_raw.init_loc_thr = 0.05;
    template_raw.score_thr = 0.05;
    template_raw.radius = {7};
    
    cbdetect::Corner corners_raw;
    cbdetect::find_corners(img, corners_raw, template_raw);
    
    results.push_back({"TemplateMatch_Raw", corners_raw});
    analyze_coordinate_distribution(corners_raw, "TemplateMatch_Raw");
    
    // Save detailed analysis
    save_coordinate_comparison(results);
    
    std::cout << "\n=======================================================" << std::endl;
    std::cout << "              CRITICAL FINDINGS" << std::endl;
    std::cout << "=======================================================" << std::endl;
    std::cout << "🚨 MAJOR ISSUE IDENTIFIED:" << std::endl;
    std::cout << "   Our algorithm detects corners in Y[80-140] region" << std::endl;
    std::cout << "   MATLAB expects corners in Y[350-562] region" << std::endl;
    std::cout << "   This is a 250+ pixel difference!" << std::endl;
    std::cout << std::endl;
    std::cout << "🔍 ANALYSIS:" << std::endl;
    std::cout << "   - We may be detecting false positives (text, noise)" << std::endl;
    std::cout << "   - Real chessboard might be in the lower part of image" << std::endl;
    std::cout << "   - Our template matching may have wrong parameters" << std::endl;
    std::cout << std::endl;
    std::cout << "📋 NEXT ACTIONS NEEDED:" << std::endl;
    std::cout << "   1. Visually inspect what we're actually detecting" << std::endl;
    std::cout << "   2. Verify chessboard location in the image" << std::endl;
    std::cout << "   3. Adjust algorithm to focus on correct region" << std::endl;
    std::cout << "   4. Compare with actual MATLAB output if available" << std::endl;
    std::cout << "=======================================================" << std::endl;
    
    return 0;
} 