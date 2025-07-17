#include "cbdetect/libcbdetect_adapter.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <vector>
#include <numeric>
#include <algorithm>

void visualize_corners_and_save(const cv::Mat& img, const cbdetect::Corner& corners, 
                                const std::string& output_name) {
    cv::Mat vis = img.clone();
    if (vis.channels() == 1) {
        cv::cvtColor(vis, vis, cv::COLOR_GRAY2BGR);
    }
    
    for (int i = 0; i < corners.p.size(); ++i) {
        cv::Point2f pt(corners.p[i].x, corners.p[i].y);
        
        // Color coding based on score
        cv::Scalar color;
        if (i < corners.score.size()) {
            double score = corners.score[i];
            if (score > 1.2) {
                color = cv::Scalar(0, 0, 255);  // Red for high quality
            } else if (score > 0.8) {
                color = cv::Scalar(0, 255, 255); // Yellow for medium quality
            } else {
                color = cv::Scalar(0, 255, 0);   // Green for low quality
            }
        } else {
            color = cv::Scalar(255, 0, 0);  // Blue for no score
        }
        
        cv::circle(vis, pt, corners.r[i], color, 2);
        cv::circle(vis, pt, 2, cv::Scalar(255, 255, 255), -1);
        
        // Draw corner index for first 20 corners
        if (i < 20) {
            cv::putText(vis, std::to_string(i), pt + cv::Point2f(8, -8),
                       cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
        }
    }
    
    // Add legend
    cv::putText(vis, "Red: High quality (>1.2)", cv::Point(10, 30),
               cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
    cv::putText(vis, "Yellow: Medium quality (0.8-1.2)", cv::Point(10, 60),
               cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
    cv::putText(vis, "Green: Lower quality (<0.8)", cv::Point(10, 90),
               cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
    
    cv::putText(vis, "Total corners: " + std::to_string(corners.p.size()), cv::Point(10, 130),
               cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
    
    cv::imshow(output_name, vis);
    cv::waitKey(1000);  // Show for 1 second
    
    // Save result
    std::string save_path = "result/" + output_name + ".png";
    cv::imwrite(save_path, vis);
    std::cout << "Saved visualization to: " << save_path << std::endl;
}

void detailed_corner_analysis(const cbdetect::Corner& corners) {
    if (corners.p.empty()) {
        std::cout << "No corners detected for analysis!" << std::endl;
        return;
    }
    
    std::cout << "\n=== Detailed Corner Analysis ===" << std::endl;
    
    // Score statistics
    if (!corners.score.empty()) {
        double min_score = *std::min_element(corners.score.begin(), corners.score.end());
        double max_score = *std::max_element(corners.score.begin(), corners.score.end());
        double avg_score = std::accumulate(corners.score.begin(), corners.score.end(), 0.0) / corners.score.size();
        
        std::cout << "Score range: " << std::fixed << std::setprecision(3) 
                  << min_score << " - " << max_score << " (avg: " << avg_score << ")" << std::endl;
        
        // High quality corners (top quartile)
        auto sorted_scores = corners.score;
        std::sort(sorted_scores.begin(), sorted_scores.end(), std::greater<double>());
        double high_quality_threshold = sorted_scores[sorted_scores.size() / 4];
        
        int high_quality_count = 0;
        for (double score : corners.score) {
            if (score >= high_quality_threshold) high_quality_count++;
        }
        
        std::cout << "High quality corners (>=" << high_quality_threshold << "): " 
                  << high_quality_count << std::endl;
    }
    
    // Spatial distribution analysis
    double min_x = corners.p[0].x, max_x = corners.p[0].x;
    double min_y = corners.p[0].y, max_y = corners.p[0].y;
    
    for (const auto& pt : corners.p) {
        min_x = std::min(min_x, pt.x);
        max_x = std::max(max_x, pt.x);
        min_y = std::min(min_y, pt.y);
        max_y = std::max(max_y, pt.y);
    }
    
    std::cout << "Spatial extent: X[" << std::fixed << std::setprecision(1) 
              << min_x << "-" << max_x << "] Y[" << min_y << "-" << max_y << "]" << std::endl;
    
    // Compare with expected MATLAB region: (42,350)-(423,562)
    std::cout << "Expected MATLAB region: X[42-423] Y[350-562]" << std::endl;
    
    // Count corners in expected region
    int corners_in_expected_region = 0;
    for (const auto& pt : corners.p) {
        if (pt.x >= 42 && pt.x <= 423 && pt.y >= 350 && pt.y <= 562) {
            corners_in_expected_region++;
        }
    }
    std::cout << "Corners in expected chessboard region: " << corners_in_expected_region << std::endl;
}

int main(int argc, char* argv[]) {
    std::string image_path = "data/04.png";
    if (argc > 1) {
        image_path = argv[1];
    }
    
    std::cout << "=== Final Optimized libcdetSample Implementation Test ===" << std::endl;
    std::cout << "Goal: Achieve ~51 corners matching MATLAB performance" << std::endl;
    std::cout << "Testing with image: " << image_path << std::endl;
    
    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Error: Could not load image " << image_path << std::endl;
        return -1;
    }
    
    std::cout << "Image size: " << img.cols << "x" << img.rows << std::endl;
    
    // Fine-tuned configuration based on best result (67 corners from normalization config)
    // Goal: reduce from 67 to ~51 by slightly increasing thresholds
    
    cbdetect::Params optimal_config;
    optimal_config.detect_method = cbdetect::TemplateMatchFast;
    optimal_config.show_processing = true;
    optimal_config.polynomial_fit = true;
    optimal_config.norm = true;                    // Critical: normalization enabled
    optimal_config.norm_half_kernel_size = 31;
    optimal_config.init_loc_thr = 0.012;          // Slightly higher than 0.01
    optimal_config.score_thr = 0.025;             // Slightly higher than 0.02 
    optimal_config.radius = {6, 8};               // Multi-scale but reduced
    
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Testing Optimal Configuration" << std::endl;
    std::cout << "detect_method: TemplateMatchFast" << std::endl;
    std::cout << "norm: true (libcdetSample style)" << std::endl;
    std::cout << "init_loc_thr: " << optimal_config.init_loc_thr << std::endl;
    std::cout << "score_thr: " << optimal_config.score_thr << std::endl;
    std::cout << "radius: {6, 8}" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    cbdetect::Corner corners;
    std::vector<cbdetect::Board> boards;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    cbdetect::find_corners(img, corners, optimal_config);
    auto corner_time = std::chrono::high_resolution_clock::now();
    cbdetect::boards_from_corners(img, corners, boards, optimal_config);
    auto end_time = std::chrono::high_resolution_clock::now();
    
    auto corner_duration = std::chrono::duration_cast<std::chrono::milliseconds>(corner_time - start_time);
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "\n=== FINAL RESULTS ===" << std::endl;
    std::cout << "Corners detected: " << corners.p.size() << std::endl;
    std::cout << "Boards detected: " << boards.size() << std::endl;
    std::cout << "Corner detection time: " << corner_duration.count() << " ms" << std::endl;
    std::cout << "Total time: " << total_duration.count() << " ms" << std::endl;
    
    // Comparison with targets
    std::cout << "\n=== PERFORMANCE COMPARISON ===" << std::endl;
    std::cout << "MATLAB target:      51 corners → 1 chessboard (7x6)" << std::endl;
    std::cout << "libcdetSample ref:  ~39 corners → working" << std::endl;
    std::cout << "Our result:         " << corners.p.size() << " corners → " << boards.size() << " boards" << std::endl;
    
    int difference_from_matlab = static_cast<int>(corners.p.size()) - 51;
    std::cout << "Difference from MATLAB: " << (difference_from_matlab >= 0 ? "+" : "") 
              << difference_from_matlab << " corners" << std::endl;
    
    if (corners.p.size() >= 45 && corners.p.size() <= 60) {
        std::cout << "🎯 EXCELLENT: Corner count very close to MATLAB!" << std::endl;
    } else if (corners.p.size() >= 35 && corners.p.size() <= 70) {
        std::cout << "✅ GOOD: Corner count within reasonable range of MATLAB!" << std::endl;
    } else {
        std::cout << "⚠️  NEEDS TUNING: Corner count differs significantly from MATLAB" << std::endl;
    }
    
    // Detailed analysis
    detailed_corner_analysis(corners);
    
    // Visualization
    if (!corners.p.empty()) {
        std::cout << "\nFirst 15 detected corners:" << std::endl;
        for (int i = 0; i < std::min(15, (int)corners.p.size()); ++i) {
            std::cout << std::fixed << std::setprecision(1);
            std::cout << "  [" << i << "] (" << corners.p[i].x << "," << corners.p[i].y << ")";
            if (i < corners.score.size()) {
                std::cout << " score=" << std::setprecision(3) << corners.score[i];
            }
            std::cout << std::endl;
        }
        
        visualize_corners_and_save(img, corners, "final_optimized_result");
    }
    
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "libcdetSample Compatible Implementation Complete!" << std::endl;
    std::cout << "This represents our best effort to match MATLAB performance" << std::endl;
    std::cout << "using the proven libcdetSample algorithm pipeline." << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    cv::destroyAllWindows();
    return 0;
} 