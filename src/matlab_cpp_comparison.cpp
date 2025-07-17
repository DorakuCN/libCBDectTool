#include "cbdetect/libcbdetect_adapter.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <chrono>

struct ComparisonResults {
    std::string method_name;
    int total_corners;
    int matlab_region_corners;
    int upper_region_corners;
    int middle_region_corners;
    int lower_region_corners;
    int detected_boards;
    double processing_time_ms;
    double avg_score;
    std::vector<cv::Point2f> corner_positions;
    std::vector<double> corner_scores;
};

void analyze_and_compare_method(const cv::Mat& img, const cbdetect::Params& params, 
                               const std::string& method_name, ComparisonResults& results) {
    
    cv::Rect matlab_rect(42, 350, 423-42, 562-350);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    cbdetect::Corner corners;
    cbdetect::find_corners(img, corners, params);
    
    std::vector<cbdetect::Board> boards;
    cbdetect::boards_from_corners(img, corners, boards, params);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    
    // Fill results
    results.method_name = method_name;
    results.total_corners = corners.p.size();
    results.processing_time_ms = duration;
    results.detected_boards = boards.size();
    
    // Reset counters
    results.matlab_region_corners = 0;
    results.upper_region_corners = 0;
    results.middle_region_corners = 0;
    results.lower_region_corners = 0;
    
    double score_sum = 0.0;
    for (int i = 0; i < corners.p.size(); ++i) {
        cv::Point2f pt = corners.p[i];
        double score = (i < corners.score.size()) ? corners.score[i] : 0.0;
        
        results.corner_positions.push_back(pt);
        results.corner_scores.push_back(score);
        score_sum += score;
        
        // Classify by region
        if (pt.y < 200) {
            results.upper_region_corners++;
        } else if (pt.y >= 200 && pt.y < 350) {
            results.middle_region_corners++;
        } else if (matlab_rect.contains(pt)) {
            results.matlab_region_corners++;
        } else {
            results.lower_region_corners++;
        }
    }
    
    results.avg_score = (corners.p.size() > 0) ? score_sum / corners.p.size() : 0.0;
}

void print_comparison_table(const std::vector<ComparisonResults>& results) {
    std::cout << "\n" << std::string(100, '=') << std::endl;
    std::cout << "                    DETAILED COMPARISON TABLE" << std::endl;
    std::cout << std::string(100, '=') << std::endl;
    
    // Header
    std::cout << std::left;
    std::cout << std::setw(20) << "Method";
    std::cout << std::setw(8) << "Total";
    std::cout << std::setw(8) << "MATLAB";
    std::cout << std::setw(8) << "Upper";
    std::cout << std::setw(8) << "Middle";
    std::cout << std::setw(8) << "Lower";
    std::cout << std::setw(8) << "Boards";
    std::cout << std::setw(8) << "Time(ms)";
    std::cout << std::setw(8) << "Score";
    std::cout << std::setw(12) << "MATLAB%";
    std::cout << "Status" << std::endl;
    
    std::cout << std::string(100, '-') << std::endl;
    
    // MATLAB target reference
    std::cout << std::setw(20) << "MATLAB_TARGET";
    std::cout << std::setw(8) << "51";
    std::cout << std::setw(8) << "51";
    std::cout << std::setw(8) << "-";
    std::cout << std::setw(8) << "-";
    std::cout << std::setw(8) << "-";
    std::cout << std::setw(8) << "1";
    std::cout << std::setw(8) << "-";
    std::cout << std::setw(8) << "-";
    std::cout << std::setw(12) << "100.0%";
    std::cout << "🎯 TARGET" << std::endl;
    
    std::cout << std::string(100, '-') << std::endl;
    
    // Results for each method
    for (const auto& result : results) {
        std::cout << std::setw(20) << result.method_name;
        std::cout << std::setw(8) << result.total_corners;
        std::cout << std::setw(8) << result.matlab_region_corners;
        std::cout << std::setw(8) << result.upper_region_corners;
        std::cout << std::setw(8) << result.middle_region_corners;
        std::cout << std::setw(8) << result.lower_region_corners;
        std::cout << std::setw(8) << result.detected_boards;
        std::cout << std::setw(8) << std::fixed << std::setprecision(0) << result.processing_time_ms;
        std::cout << std::setw(8) << std::fixed << std::setprecision(3) << result.avg_score;
        
        double matlab_percentage = (result.matlab_region_corners * 100.0) / 51.0;
        std::cout << std::setw(12) << std::fixed << std::setprecision(1) << matlab_percentage << "%";
        
        // Status evaluation
        int diff = abs(result.matlab_region_corners - 51);
        if (diff <= 5) {
            std::cout << "🎯 EXCELLENT";
        } else if (diff <= 10) {
            std::cout << "✅ VERY_GOOD";
        } else if (diff <= 15) {
            std::cout << "✅ GOOD";
        } else if (result.matlab_region_corners >= 25) {
            std::cout << "⚠️ MODERATE";
        } else {
            std::cout << "❌ POOR";
        }
        
        std::cout << std::endl;
    }
    
    std::cout << std::string(100, '=') << std::endl;
}

void create_comparison_visualization(const cv::Mat& img, const std::vector<ComparisonResults>& results) {
    cv::Rect matlab_rect(42, 350, 423-42, 562-350);
    
    // Create a large canvas for multiple visualizations
    int img_width = img.cols;
    int img_height = img.rows;
    int canvas_width = img_width * 2;
    int canvas_height = img_height * ((results.size() + 1) / 2);
    
    cv::Mat canvas = cv::Mat::zeros(canvas_height, canvas_width, CV_8UC3);
    
    for (int i = 0; i < results.size(); ++i) {
        const auto& result = results[i];
        
        // Calculate position on canvas
        int row = i / 2;
        int col = i % 2;
        int x_offset = col * img_width;
        int y_offset = row * img_height;
        
        cv::Rect roi(x_offset, y_offset, img_width, img_height);
        cv::Mat sub_canvas = canvas(roi);
        
        // Convert image to color if needed
        cv::Mat vis_img;
        if (img.channels() == 1) {
            cv::cvtColor(img, vis_img, cv::COLOR_GRAY2BGR);
        } else {
            vis_img = img.clone();
        }
        
        // Draw MATLAB region
        cv::rectangle(vis_img, matlab_rect, cv::Scalar(0, 255, 255), 2);
        
        // Draw corners with region-based coloring
        for (int j = 0; j < result.corner_positions.size(); ++j) {
            cv::Point2f pt = result.corner_positions[j];
            cv::Scalar color;
            
            if (matlab_rect.contains(pt)) {
                color = cv::Scalar(0, 255, 0);      // Green for MATLAB region
            } else if (pt.y < 200) {
                color = cv::Scalar(0, 0, 255);      // Red for upper region
            } else if (pt.y >= 200 && pt.y < 350) {
                color = cv::Scalar(255, 255, 0);    // Cyan for middle region
            } else {
                color = cv::Scalar(255, 0, 255);    // Magenta for lower region
            }
            
            cv::circle(vis_img, pt, 6, color, 2);
            cv::circle(vis_img, pt, 2, cv::Scalar(255, 255, 255), -1);
        }
        
        // Add method info
        cv::putText(vis_img, result.method_name, cv::Point(10, 30), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
        cv::putText(vis_img, "Total: " + std::to_string(result.total_corners), 
                   cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
        cv::putText(vis_img, "MATLAB: " + std::to_string(result.matlab_region_corners), 
                   cv::Point(10, 90), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
        cv::putText(vis_img, "Boards: " + std::to_string(result.detected_boards), 
                   cv::Point(10, 120), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 0), 2);
        
        // Copy to canvas
        vis_img.copyTo(sub_canvas);
    }
    
    std::string save_path = "result/matlab_cpp_comparison.png";
    cv::imwrite(save_path, canvas);
    std::cout << "\nComparison visualization saved to: " << save_path << std::endl;
}

void save_detailed_report(const std::vector<ComparisonResults>& results) {
    std::ofstream report("result/matlab_cpp_detailed_report.txt");
    
    report << "=======================================================" << std::endl;
    report << "    MATLAB vs C++ DETAILED COMPARISON REPORT" << std::endl;
    report << "=======================================================" << std::endl;
    report << "Analysis Date: " << __DATE__ << " " << __TIME__ << std::endl;
    report << "Image: data/04.png (480x752)" << std::endl;
    report << "MATLAB Target: 51 corners in region X[42-423] Y[350-562]" << std::endl;
    report << std::endl;
    
    report << "=== SUMMARY TABLE ===" << std::endl;
    report << std::left;
    report << std::setw(20) << "Method" << std::setw(8) << "Total" << std::setw(8) << "MATLAB" 
           << std::setw(12) << "Accuracy" << std::setw(10) << "Time(ms)" << "Performance" << std::endl;
    report << std::string(70, '-') << std::endl;
    
    for (const auto& result : results) {
        report << std::setw(20) << result.method_name;
        report << std::setw(8) << result.total_corners;
        report << std::setw(8) << result.matlab_region_corners;
        report << std::setw(12) << std::fixed << std::setprecision(1) 
               << (result.matlab_region_corners * 100.0 / 51.0) << "%";
        report << std::setw(10) << std::fixed << std::setprecision(0) << result.processing_time_ms;
        
        int diff = abs(result.matlab_region_corners - 51);
        if (diff <= 5) report << "EXCELLENT";
        else if (diff <= 10) report << "VERY_GOOD";
        else if (diff <= 15) report << "GOOD";
        else if (result.matlab_region_corners >= 25) report << "MODERATE";
        else report << "POOR";
        
        report << std::endl;
    }
    
    report << std::endl << "=== DETAILED ANALYSIS ===" << std::endl;
    
    for (const auto& result : results) {
        report << std::endl << "--- " << result.method_name << " ---" << std::endl;
        report << "Total corners: " << result.total_corners << std::endl;
        report << "MATLAB region: " << result.matlab_region_corners << " (" 
               << std::fixed << std::setprecision(1) << (result.matlab_region_corners * 100.0 / result.total_corners) << "%)" << std::endl;
        report << "Upper region: " << result.upper_region_corners << " (" 
               << std::fixed << std::setprecision(1) << (result.upper_region_corners * 100.0 / result.total_corners) << "%)" << std::endl;
        report << "Middle region: " << result.middle_region_corners << std::endl;
        report << "Lower region: " << result.lower_region_corners << std::endl;
        report << "Detected boards: " << result.detected_boards << std::endl;
        report << "Processing time: " << std::fixed << std::setprecision(1) << result.processing_time_ms << " ms" << std::endl;
        report << "Average score: " << std::fixed << std::setprecision(3) << result.avg_score << std::endl;
        report << "MATLAB target diff: " << (result.matlab_region_corners - 51) << " corners" << std::endl;
    }
    
    report << std::endl << "=== CONCLUSIONS ===" << std::endl;
    
    // Find best performer
    auto best_it = std::min_element(results.begin(), results.end(), 
        [](const ComparisonResults& a, const ComparisonResults& b) {
            return abs(a.matlab_region_corners - 51) < abs(b.matlab_region_corners - 51);
        });
    
    if (best_it != results.end()) {
        report << "Best performing method: " << best_it->method_name << std::endl;
        report << "  MATLAB region corners: " << best_it->matlab_region_corners << "/51" << std::endl;
        report << "  Accuracy: " << std::fixed << std::setprecision(1) 
               << (best_it->matlab_region_corners * 100.0 / 51.0) << "%" << std::endl;
        report << "  Difference: " << (best_it->matlab_region_corners - 51) << " corners" << std::endl;
    }
    
    // Count how many methods are "good enough"
    int good_methods = 0;
    for (const auto& result : results) {
        if (abs(result.matlab_region_corners - 51) <= 15) {
            good_methods++;
        }
    }
    
    report << std::endl << "Methods within 15 corners of MATLAB target: " << good_methods 
           << "/" << results.size() << std::endl;
    
    if (good_methods > 0) {
        report << "✅ SUCCESS: Found working configurations" << std::endl;
    } else {
        report << "⚠️ IMPROVEMENT NEEDED: All methods significantly differ from MATLAB" << std::endl;
    }
    
    report.close();
    std::cout << "Detailed report saved to: result/matlab_cpp_detailed_report.txt" << std::endl;
}

int main(int argc, char* argv[]) {
    std::string image_path = "data/04.png";
    if (argc > 1) {
        image_path = argv[1];
    }
    
    std::cout << "=======================================================" << std::endl;
    std::cout << "    MATLAB vs C++ COMPREHENSIVE COMPARISON" << std::endl;
    std::cout << "=======================================================" << std::endl;
    std::cout << "Objective: Compare C++ implementations with MATLAB target" << std::endl;
    std::cout << "MATLAB Target: 51 corners in region X[42-423] Y[350-562]" << std::endl;
    std::cout << "Image: " << image_path << std::endl;
    
    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Error: Could not load image " << image_path << std::endl;
        return -1;
    }
    
    std::cout << "Image size: " << img.cols << "x" << img.rows << std::endl;
    
    // Define test configurations
    std::vector<std::pair<std::string, cbdetect::Params>> test_configs = {
        {"Original_Config", []() {
            cbdetect::Params p;
            p.detect_method = cbdetect::TemplateMatchFast;
            p.norm = true;
            p.init_loc_thr = 0.012;
            p.score_thr = 0.025;
            p.radius = {6, 8};
            p.polynomial_fit = true;
            p.show_processing = false;
            return p;
        }()},
        
        {"Conservative_Fine", []() {
            cbdetect::Params p;
            p.detect_method = cbdetect::TemplateMatchFast;
            p.norm = true;
            p.init_loc_thr = 0.02;
            p.score_thr = 0.05;
            p.radius = {6, 7};
            p.polynomial_fit = true;
            p.show_processing = false;
            return p;
        }()},
        
        {"HessianResponse_Opt", []() {
            cbdetect::Params p;
            p.detect_method = cbdetect::HessianResponse;
            p.norm = false;
            p.init_loc_thr = 0.1;
            p.score_thr = 0.1;
            p.radius = {7};
            p.polynomial_fit = true;
            p.show_processing = false;
            return p;
        }()},
        
        {"MATLAB_Like", []() {
            cbdetect::Params p;
            p.detect_method = cbdetect::TemplateMatchFast;
            p.norm = true;
            p.norm_half_kernel_size = 31;
            p.init_loc_thr = 0.015;
            p.score_thr = 0.03;
            p.radius = {6, 7, 8};
            p.polynomial_fit = true;
            p.show_processing = false;
            return p;
        }()}
    };
    
    std::vector<ComparisonResults> all_results;
    
    std::cout << "\n=== TESTING C++ CONFIGURATIONS ===" << std::endl;
    
    for (const auto& config : test_configs) {
        std::cout << "\nTesting: " << config.first << "..." << std::flush;
        
        ComparisonResults result;
        analyze_and_compare_method(img, config.second, config.first, result);
        all_results.push_back(result);
        
        std::cout << " Done. (" << result.total_corners << " corners, " 
                  << result.matlab_region_corners << " in MATLAB region)" << std::endl;
    }
    
    // Print comprehensive comparison
    print_comparison_table(all_results);
    
    // Create visualizations
    create_comparison_visualization(img, all_results);
    
    // Save detailed report
    save_detailed_report(all_results);
    
    std::cout << "\n=======================================================" << std::endl;
    std::cout << "    COMPARISON ANALYSIS COMPLETE" << std::endl;
    std::cout << "=======================================================" << std::endl;
    
    // Find and highlight best result
    auto best_result = std::min_element(all_results.begin(), all_results.end(),
        [](const ComparisonResults& a, const ComparisonResults& b) {
            return abs(a.matlab_region_corners - 51) < abs(b.matlab_region_corners - 51);
        });
    
    if (best_result != all_results.end()) {
        std::cout << "🏆 BEST RESULT: " << best_result->method_name << std::endl;
        std::cout << "   MATLAB region corners: " << best_result->matlab_region_corners << "/51" << std::endl;
        std::cout << "   Accuracy: " << std::fixed << std::setprecision(1) 
                  << (best_result->matlab_region_corners * 100.0 / 51.0) << "%" << std::endl;
        std::cout << "   Difference: " << (best_result->matlab_region_corners - 51) << " corners" << std::endl;
        
        int diff = abs(best_result->matlab_region_corners - 51);
        if (diff <= 5) {
            std::cout << "🎯 EXCELLENT: Very close to MATLAB target!" << std::endl;
        } else if (diff <= 15) {
            std::cout << "✅ GOOD: Reasonably close to MATLAB target" << std::endl;
        } else {
            std::cout << "⚠️ MODERATE: Room for improvement" << std::endl;
        }
    }
    
    std::cout << "\n📁 Generated files:" << std::endl;
    std::cout << "   - result/matlab_cpp_comparison.png (visual comparison)" << std::endl;
    std::cout << "   - result/matlab_cpp_detailed_report.txt (detailed analysis)" << std::endl;
    std::cout << "   - result/cpp_debug_log.txt (debug logs)" << std::endl;
    
    return 0;
} 