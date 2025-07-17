#include "cbdetect/libcbdetect_adapter.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <vector>

void test_parameter_configuration(const cv::Mat& img, const std::string& config_name, 
                                 const cbdetect::Params& params) {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Testing Configuration: " << config_name << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    cbdetect::Corner corners;
    std::vector<cbdetect::Board> boards;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    cbdetect::find_corners(img, corners, params);
    auto corner_time = std::chrono::high_resolution_clock::now();
    cbdetect::boards_from_corners(img, corners, boards, params);
    auto end_time = std::chrono::high_resolution_clock::now();
    
    auto corner_duration = std::chrono::duration_cast<std::chrono::milliseconds>(corner_time - start_time);
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "Results: " << corners.p.size() << " corners, " << boards.size() << " boards" << std::endl;
    std::cout << "Time: " << corner_duration.count() << " ms" << std::endl;
    
    // Print some corner coordinates for comparison with MATLAB
    if (!corners.p.empty()) {
        std::cout << "First 10 corners:" << std::endl;
        for (int i = 0; i < std::min(10, (int)corners.p.size()); ++i) {
            std::cout << std::fixed << std::setprecision(1);
            std::cout << "  [" << i << "] (" << corners.p[i].x << "," << corners.p[i].y << ")";
            if (i < corners.score.size()) {
                std::cout << " score=" << std::setprecision(3) << corners.score[i];
            }
            std::cout << std::endl;
        }
    }
    
    // MATLAB comparison
    std::cout << "\nMATLAB baseline: 51 corners → 1 chessboard (7x6)" << std::endl;
    std::cout << "Current result: " << corners.p.size() << " corners → " << boards.size() << " boards" << std::endl;
    
    if (corners.p.size() >= 30 && corners.p.size() <= 70) {
        std::cout << "✅ Corner count within expected range!" << std::endl;
    } else if (corners.p.size() > 70) {
        std::cout << "⚠️  Too many corners detected (threshold too low)" << std::endl;
    } else {
        std::cout << "⚠️  Too few corners detected (threshold too high)" << std::endl;
    }
}

int main(int argc, char* argv[]) {
    std::string image_path = "data/04.png";
    if (argc > 1) {
        image_path = argv[1];
    }
    
    std::cout << "=== Parameter Optimization Test ===" << std::endl;
    std::cout << "Target: Match MATLAB performance (51 corners → 1 chessboard)" << std::endl;
    std::cout << "Testing with image: " << image_path << std::endl;
    
    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Error: Could not load image " << image_path << std::endl;
        return -1;
    }
    
    std::cout << "Image size: " << img.cols << "x" << img.rows << std::endl;
    
    // Configuration 1: TemplateMatchFast with higher thresholds
    cbdetect::Params config1;
    config1.detect_method = cbdetect::TemplateMatchFast;
    config1.show_processing = false;
    config1.polynomial_fit = true;
    config1.norm = false;
    config1.init_loc_thr = 0.02;  // Increased threshold
    config1.score_thr = 0.05;     // Increased threshold
    config1.radius = {6, 8};      // Reduced scale variation
    test_parameter_configuration(img, "TemplateMatchFast - Higher Thresholds", config1);
    
    // Configuration 2: HessianResponse with much higher threshold
    cbdetect::Params config2;
    config2.detect_method = cbdetect::HessianResponse;
    config2.show_processing = false;
    config2.polynomial_fit = true;
    config2.norm = false;
    config2.init_loc_thr = 0.1;   // Much higher threshold
    config2.score_thr = 0.1;      // Much higher threshold
    config2.radius = {6, 8};
    test_parameter_configuration(img, "HessianResponse - High Thresholds", config2);
    
    // Configuration 3: TemplateMatchFast with image normalization
    cbdetect::Params config3;
    config3.detect_method = cbdetect::TemplateMatchFast;
    config3.show_processing = false;
    config3.polynomial_fit = true;
    config3.norm = true;          // Enable normalization like libcdetSample
    config3.norm_half_kernel_size = 31;
    config3.init_loc_thr = 0.01;
    config3.score_thr = 0.02;
    config3.radius = {6, 8};
    test_parameter_configuration(img, "TemplateMatchFast - With Normalization", config3);
    
    // Configuration 4: TemplateMatchSlow with optimization
    cbdetect::Params config4;
    config4.detect_method = cbdetect::TemplateMatchSlow;
    config4.show_processing = false;
    config4.polynomial_fit = true;
    config4.norm = false;
    config4.init_loc_thr = 0.03;
    config4.score_thr = 0.08;
    config4.radius = {6, 8};
    test_parameter_configuration(img, "TemplateMatchSlow - Optimized", config4);
    
    // Configuration 5: Conservative HessianResponse
    cbdetect::Params config5;
    config5.detect_method = cbdetect::HessianResponse;
    config5.show_processing = false;
    config5.polynomial_fit = true;
    config5.norm = false;
    config5.init_loc_thr = 0.05;
    config5.score_thr = 0.05;
    config5.radius = {6};         // Single scale
    test_parameter_configuration(img, "HessianResponse - Conservative", config5);
    
    // Configuration 6: MATLAB-like parameters (best guess)
    cbdetect::Params config6;
    config6.detect_method = cbdetect::TemplateMatchFast;
    config6.show_processing = false;
    config6.polynomial_fit = true;
    config6.norm = true;
    config6.norm_half_kernel_size = 31;
    config6.init_loc_thr = 0.015;
    config6.score_thr = 0.03;
    config6.radius = {7};         // MATLAB likely uses single scale
    test_parameter_configuration(img, "MATLAB-like Configuration", config6);
    
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Parameter optimization test complete!" << std::endl;
    std::cout << "Recommendation: Use the configuration with corner count closest to 51" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    return 0;
} 