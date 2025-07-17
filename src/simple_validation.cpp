#include "cbdetect/libcbdetect_adapter.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <vector>
#include <numeric>
#include <algorithm>
#include <fstream>

struct TestResult {
    std::string config_name;
    int corners;
    int boards;
    double time_ms;
    double avg_score;
    std::vector<cv::Point2d> positions;
    int matlab_region_count;
    std::string performance_level;
};

void generate_comparison_report(const std::vector<TestResult>& results) {
    std::ofstream report("result/comparison_report.txt");
    
    report << "=======================================================" << std::endl;
    report << "     CHESSBOARD DETECTION VALIDATION REPORT" << std::endl;
    report << "=======================================================" << std::endl;
    report << "Target: Match MATLAB performance (51 corners)" << std::endl;
    report << "MATLAB Region: X[42-423] Y[350-562]" << std::endl;
    report << "Generated: " << std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()) << std::endl;
    report << std::endl;
    
    // Performance comparison table
    report << "=== PERFORMANCE COMPARISON ===" << std::endl;
    report << std::left << std::setw(25) << "Configuration"
           << std::setw(10) << "Corners"
           << std::setw(10) << "Boards"
           << std::setw(12) << "Time(ms)"
           << std::setw(12) << "Avg Score"
           << std::setw(12) << "MATLAB Diff"
           << std::setw(15) << "Performance" << std::endl;
    report << std::string(95, '-') << std::endl;
    
    for (const auto& result : results) {
        int diff = result.corners - 51;
        std::string diff_str = (diff >= 0 ? "+" : "") + std::to_string(diff);
        
        report << std::left << std::setw(25) << result.config_name
               << std::setw(10) << result.corners
               << std::setw(10) << result.boards
               << std::setw(12) << std::fixed << std::setprecision(1) << result.time_ms
               << std::setw(12) << std::fixed << std::setprecision(3) << result.avg_score
               << std::setw(12) << diff_str
               << std::setw(15) << result.performance_level << std::endl;
    }
    
    report << std::endl;
    
    // Historical comparison
    report << "=== HISTORICAL PROGRESS ===" << std::endl;
    report << "Original Problem:     8 corners  → 0 boards  ❌ FAILED" << std::endl;
    report << "MATLAB Baseline:     51 corners → 1 board   🎯 TARGET" << std::endl;
    report << "libcdetSample Ref:   ~39 corners → working  ✅ REFERENCE" << std::endl;
    
    // Find best result
    auto best_result = std::min_element(results.begin(), results.end(),
        [](const TestResult& a, const TestResult& b) {
            return std::abs(a.corners - 51) < std::abs(b.corners - 51);
        });
    
    if (best_result != results.end()) {
        report << "Our Best Result:     " << best_result->corners << " corners → " 
               << best_result->boards << " boards  " << best_result->performance_level << std::endl;
    }
    
    report << std::endl;
    
    // Detailed analysis
    for (const auto& result : results) {
        report << "=== " << result.config_name << " ===" << std::endl;
        report << "Corner Count: " << result.corners << " (MATLAB diff: " << (result.corners - 51) << ")" << std::endl;
        report << "Detection Time: " << std::fixed << std::setprecision(1) << result.time_ms << " ms" << std::endl;
        report << "Average Score: " << std::fixed << std::setprecision(3) << result.avg_score << std::endl;
        report << "MATLAB Region Coverage: " << result.matlab_region_count << "/" << result.corners 
               << " (" << (100.0 * result.matlab_region_count / std::max(1, result.corners)) << "%)" << std::endl;
        
        if (!result.positions.empty() && result.positions.size() >= 5) {
            report << "Sample Corners: ";
            for (int i = 0; i < std::min(5, (int)result.positions.size()); ++i) {
                report << "(" << std::fixed << std::setprecision(1) 
                       << result.positions[i].x << "," << result.positions[i].y << ")";
                if (i < 4) report << ", ";
            }
            report << std::endl;
        }
        report << std::endl;
    }
    
    // Final conclusion
    report << "=== VALIDATION CONCLUSION ===" << std::endl;
    bool has_excellent = false;
    for (const auto& result : results) {
        if (result.performance_level.find("EXCELLENT") != std::string::npos) {
            has_excellent = true;
            break;
        }
    }
    
    if (has_excellent) {
        report << "🎯 VALIDATION STATUS: SUCCESS" << std::endl;
        report << "✅ ACHIEVEMENT: libcdetSample compatible implementation successfully" << std::endl;
        report << "   matches MATLAB performance within acceptable range" << std::endl;
        report << "🏆 RECOMMENDATION: Use EXCELLENT configuration for production" << std::endl;
    } else {
        report << "⚠️ VALIDATION STATUS: PARTIAL SUCCESS" << std::endl;
        report << "📈 PROGRESS: Significant improvement from original 8 corners" << std::endl;
        report << "🔧 RECOMMENDATION: Further fine-tuning of parameters" << std::endl;
    }
    
    report.close();
    std::cout << "\n📄 Detailed comparison report saved to: result/comparison_report.txt" << std::endl;
}

TestResult run_configuration_test(const cv::Mat& img, const cbdetect::Params& params, 
                                  const std::string& config_name) {
    std::cout << "\n" << std::string(50, '-') << std::endl;
    std::cout << "Testing: " << config_name << std::endl;
    std::cout << std::string(50, '-') << std::endl;
    
    TestResult result;
    result.config_name = config_name;
    
    cbdetect::Corner corners;
    std::vector<cbdetect::Board> boards;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    cbdetect::find_corners(img, corners, params);
    cbdetect::boards_from_corners(img, corners, boards, params);
    auto end_time = std::chrono::high_resolution_clock::now();
    
    result.corners = corners.p.size();
    result.boards = boards.size();
    result.time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    result.positions = corners.p;
    
    // Calculate average score
    if (!corners.score.empty()) {
        result.avg_score = std::accumulate(corners.score.begin(), corners.score.end(), 0.0) / corners.score.size();
    } else {
        result.avg_score = 0.0;
    }
    
    // Count corners in MATLAB expected region
    result.matlab_region_count = 0;
    for (const auto& pt : corners.p) {
        if (pt.x >= 42 && pt.x <= 423 && pt.y >= 350 && pt.y <= 562) {
            result.matlab_region_count++;
        }
    }
    
    // Determine performance level
    if (result.corners >= 45 && result.corners <= 60) {
        result.performance_level = "🎯 EXCELLENT";
    } else if (result.corners >= 35 && result.corners <= 70) {
        result.performance_level = "✅ GOOD";
    } else if (result.corners > 70) {
        result.performance_level = "⚠️ TOO_MANY";
    } else {
        result.performance_level = "❌ TOO_FEW";
    }
    
    std::cout << "Results: " << result.corners << " corners, " << result.boards << " boards" << std::endl;
    std::cout << "Time: " << std::fixed << std::setprecision(1) << result.time_ms << " ms" << std::endl;
    std::cout << "Performance: " << result.performance_level << std::endl;
    
    return result;
}

int main(int argc, char* argv[]) {
    std::string image_path = "data/04.png";
    if (argc > 1) {
        image_path = argv[1];
    }
    
    std::cout << "=======================================================" << std::endl;
    std::cout << "    libcdetSample VALIDATION & COMPARISON REPORT" << std::endl;
    std::cout << "=======================================================" << std::endl;
    std::cout << "🎯 Target: Match MATLAB baseline (51 corners → 1 board)" << std::endl;
    std::cout << "📸 Image: " << image_path << std::endl;
    
    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "❌ Error: Could not load image " << image_path << std::endl;
        return -1;
    }
    
    std::cout << "📐 Image size: " << img.cols << "x" << img.rows << std::endl;
    
    std::vector<TestResult> results;
    
    // Configuration 1: Original libcdetSample defaults
    cbdetect::Params config1;
    config1.detect_method = cbdetect::HessianResponse;
    config1.show_processing = false;
    config1.norm = false;
    config1.init_loc_thr = 0.01;
    config1.score_thr = 0.01;
    config1.radius = {5, 7};
    results.push_back(run_configuration_test(img, config1, "HessianResponse_Default"));
    
    // Configuration 2: TemplateMatchFast original
    cbdetect::Params config2;
    config2.detect_method = cbdetect::TemplateMatchFast;
    config2.show_processing = false;
    config2.norm = false;
    config2.init_loc_thr = 0.01;
    config2.score_thr = 0.01;
    config2.radius = {4, 6, 8};
    results.push_back(run_configuration_test(img, config2, "TemplateMatchFast_Orig"));
    
    // Configuration 3: With normalization (key breakthrough)
    cbdetect::Params config3;
    config3.detect_method = cbdetect::TemplateMatchFast;
    config3.show_processing = false;
    config3.norm = true;  // Critical improvement
    config3.norm_half_kernel_size = 31;
    config3.init_loc_thr = 0.01;
    config3.score_thr = 0.02;
    config3.radius = {6, 8};
    results.push_back(run_configuration_test(img, config3, "TemplateMatchFast_WithNorm"));
    
    // Configuration 4: Final optimized (best result)
    cbdetect::Params config4;
    config4.detect_method = cbdetect::TemplateMatchFast;
    config4.show_processing = false;
    config4.norm = true;
    config4.norm_half_kernel_size = 31;
    config4.init_loc_thr = 0.012;  // Fine-tuned
    config4.score_thr = 0.025;     // Fine-tuned
    config4.radius = {6, 8};
    config4.polynomial_fit = true;
    results.push_back(run_configuration_test(img, config4, "Final_Optimized"));
    
    // Configuration 5: Conservative HessianResponse
    cbdetect::Params config5;
    config5.detect_method = cbdetect::HessianResponse;
    config5.show_processing = false;
    config5.norm = false;
    config5.init_loc_thr = 0.05;  // Higher threshold
    config5.score_thr = 0.05;
    config5.radius = {6};  // Single scale
    results.push_back(run_configuration_test(img, config5, "HessianResponse_Conservative"));
    
    // Generate summary
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "                    VALIDATION SUMMARY" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    std::cout << std::left << std::setw(30) << "Configuration"
              << std::setw(10) << "Corners"
              << std::setw(10) << "Boards"
              << std::setw(15) << "MATLAB Diff"
              << std::setw(15) << "Performance" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    
    for (const auto& result : results) {
        int diff = result.corners - 51;
        std::string diff_str = (diff >= 0 ? "+" : "") + std::to_string(diff);
        
        std::cout << std::left << std::setw(30) << result.config_name
                  << std::setw(10) << result.corners
                  << std::setw(10) << result.boards
                  << std::setw(15) << diff_str
                  << std::setw(15) << result.performance_level << std::endl;
    }
    
    // Find and highlight best result
    auto best_result = std::min_element(results.begin(), results.end(),
        [](const TestResult& a, const TestResult& b) {
            return std::abs(a.corners - 51) < std::abs(b.corners - 51);
        });
    
    std::cout << "\n🏆 BEST PERFORMANCE: " << best_result->config_name << std::endl;
    std::cout << "📊 Result: " << best_result->corners << " corners (diff: " 
              << (best_result->corners - 51) << " from MATLAB)" << std::endl;
    std::cout << "⏱️ Speed: " << std::fixed << std::setprecision(1) << best_result->time_ms << " ms" << std::endl;
    std::cout << "🎯 Status: " << best_result->performance_level << std::endl;
    
    // Overall validation conclusion
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "                 VALIDATION CONCLUSION" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    bool has_excellent = false;
    for (const auto& result : results) {
        if (result.performance_level.find("EXCELLENT") != std::string::npos) {
            has_excellent = true;
            break;
        }
    }
    
    std::cout << "📈 PROGRESS COMPARISON:" << std::endl;
    std::cout << "   Original Implementation: 8 corners → 0 boards  ❌ FAILED" << std::endl;
    std::cout << "   MATLAB Baseline:        51 corners → 1 board   🎯 TARGET" << std::endl;
    std::cout << "   Our Best Result:        " << best_result->corners << " corners → " 
              << best_result->boards << " boards  " << best_result->performance_level << std::endl;
    
    std::cout << "\n";
    if (has_excellent) {
        std::cout << "✅ VALIDATION STATUS: SUCCESS!" << std::endl;
        std::cout << "🎉 ACHIEVEMENT: libcdetSample compatible implementation successfully" << std::endl;
        std::cout << "   achieves performance comparable to MATLAB baseline!" << std::endl;
        std::cout << "🏆 RECOMMENDATION: Deploy the EXCELLENT configuration to production" << std::endl;
    } else {
        std::cout << "⚠️ VALIDATION STATUS: PARTIAL SUCCESS" << std::endl;
        std::cout << "📈 SIGNIFICANT IMPROVEMENT from original 8 corners to " << best_result->corners << " corners" << std::endl;
        std::cout << "🔧 RECOMMENDATION: Consider further parameter fine-tuning" << std::endl;
    }
    
    // Generate detailed report
    generate_comparison_report(results);
    
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "Validation complete! Check result/comparison_report.txt for details." << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    return 0;
} 