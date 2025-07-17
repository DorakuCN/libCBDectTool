#include "cbdetect/libcbdetect_adapter.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <numeric>

class DebugLogger {
private:
    std::ofstream log_file;
    
public:
    DebugLogger(const std::string& filename) {
        log_file.open(filename);
        log_file << std::fixed << std::setprecision(3);
    }
    
    ~DebugLogger() {
        if (log_file.is_open()) {
            log_file.close();
        }
    }
    
    template<typename T>
    void log(const T& message) {
        std::cout << message;
        if (log_file.is_open()) {
            log_file << message;
        }
    }
    
    void log_endl() {
        std::cout << std::endl;
        if (log_file.is_open()) {
            log_file << std::endl;
        }
    }
};

struct RegionStats {
    int upper_region = 0;    // Y < 200
    int middle_region = 0;   // Y 200-350
    int matlab_region = 0;   // X[42-423] Y[350-562]
    int lower_region = 0;    // Y > 350 but outside matlab region
    
    std::vector<cv::Point2f> matlab_corners;
    std::vector<double> matlab_scores;
};

RegionStats analyze_corner_regions(const cbdetect::Corner& corners, DebugLogger& logger) {
    RegionStats stats;
    cv::Rect matlab_rect(42, 350, 423-42, 562-350);
    
    logger.log("\nDetailed corner information:\n");
    logger.log("Corners in different regions:\n");
    
    for (int i = 0; i < corners.p.size(); ++i) {
        cv::Point2f pt = corners.p[i];
        double score = (i < corners.score.size()) ? corners.score[i] : 0.0;
        
        std::string region_label;
        if (pt.y < 200) {
            stats.upper_region++;
            region_label = "UPPER";
        } else if (pt.y >= 200 && pt.y < 350) {
            stats.middle_region++;
            region_label = "MIDDLE";
        } else if (matlab_rect.contains(pt)) {
            stats.matlab_region++;
            region_label = "MATLAB";
            stats.matlab_corners.push_back(pt);
            stats.matlab_scores.push_back(score);
        } else {
            stats.lower_region++;
            region_label = "LOWER";
        }
        
        // Log first 20 corners or all MATLAB region corners
        if (i < 20 || region_label == "MATLAB") {
            logger.log("  [" + std::to_string(i+1) + "] (" + 
                      std::to_string(pt.x) + ", " + std::to_string(pt.y) + 
                      ") score=" + std::to_string(score) + " [" + region_label + "]\n");
        }
    }
    
    return stats;
}

void log_corner_structure(const cbdetect::Corner& corners, DebugLogger& logger) {
    logger.log("\n=== STEP 2: CORNER PROCESSING ===\n");
    logger.log("Corner structure fields:\n");
    
    logger.log("  p: " + std::to_string(corners.p.size()) + " corner coordinates\n");
    
    if (!corners.r.empty()) {
        logger.log("  r: " + std::to_string(corners.r.size()) + " radius values\n");
    } else {
        logger.log("  r: empty\n");
    }
    
    if (!corners.v1.empty()) {
        logger.log("  v1: " + std::to_string(corners.v1.size()) + " direction vectors\n");
        // Show first few direction vectors
        for (int i = 0; i < std::min(5, (int)corners.v1.size()); ++i) {
            const auto& v = corners.v1[i];
            logger.log("    [" + std::to_string(i+1) + "]: (" + 
                      std::to_string(v.x) + ", " + std::to_string(v.y) + ")\n");
        }
    } else {
        logger.log("  v1: empty\n");
    }
    
    if (!corners.v2.empty()) {
        logger.log("  v2: " + std::to_string(corners.v2.size()) + " direction vectors\n");
        // Show first few direction vectors
        for (int i = 0; i < std::min(5, (int)corners.v2.size()); ++i) {
            const auto& v = corners.v2[i];
            logger.log("    [" + std::to_string(i+1) + "]: (" + 
                      std::to_string(v.x) + ", " + std::to_string(v.y) + ")\n");
        }
    } else {
        logger.log("  v2: empty\n");
    }
    
    if (!corners.score.empty()) {
        logger.log("  score: " + std::to_string(corners.score.size()) + " elements\n");
    } else {
        logger.log("  score: empty\n");
    }
}

void log_board_details(const std::vector<cbdetect::Board>& boards, 
                      const cbdetect::Corner& corners, DebugLogger& logger) {
    logger.log("\n=== STEP 3: CHESSBOARD DETECTION ===\n");
    logger.log("Chessboards detected: " + std::to_string(boards.size()) + "\n");
    
    for (int i = 0; i < boards.size(); ++i) {
        const auto& board = boards[i];
        logger.log("\nBoard " + std::to_string(i+1) + " details:\n");
        
        if (!board.energy.empty() && !board.energy[0].empty() && !board.energy[0][0].empty()) {
            logger.log("  Energy: " + std::to_string(board.energy[0][0][0]) + "\n");
        }
        
        if (!board.idx.empty()) {
            logger.log("  Corners: " + std::to_string(board.idx.size()) + "\n");
            
            // Calculate spatial extent
            if (!board.idx.empty()) {
                double min_x = std::numeric_limits<double>::max();
                double max_x = std::numeric_limits<double>::min();
                double min_y = std::numeric_limits<double>::max();
                double max_y = std::numeric_limits<double>::min();
                
                for (const auto& idx_vec : board.idx) {
                    for (int idx : idx_vec) {
                        if (idx < corners.p.size()) {
                            cv::Point2f pt = corners.p[idx];
                            min_x = std::min(min_x, (double)pt.x);
                            max_x = std::max(max_x, (double)pt.x);
                            min_y = std::min(min_y, (double)pt.y);
                            max_y = std::max(max_y, (double)pt.y);
                        }
                    }
                }
                
                logger.log("  Spatial extent: X[" + std::to_string(min_x) + "-" + 
                          std::to_string(max_x) + "] Y[" + std::to_string(min_y) + 
                          "-" + std::to_string(max_y) + "]\n");
            }
        }
        
        logger.log("  Corner indices size: " + std::to_string(board.idx.size()) + "\n");
    }
}

void create_debug_visualization(const cv::Mat& img, const cbdetect::Corner& corners, 
                               const std::vector<cbdetect::Board>& boards,
                               const RegionStats& stats, const std::string& title) {
    cv::Mat vis_all, vis_matlab, vis_board, vis_combined;
    cv::Rect matlab_rect(42, 350, 423-42, 562-350);
    
    // Convert to color if needed
    if (img.channels() == 1) {
        cv::cvtColor(img, vis_all, cv::COLOR_GRAY2BGR);
        cv::cvtColor(img, vis_matlab, cv::COLOR_GRAY2BGR);
        cv::cvtColor(img, vis_board, cv::COLOR_GRAY2BGR);
    } else {
        vis_all = img.clone();
        vis_matlab = img.clone();
        vis_board = img.clone();
    }
    
    // Visualization 1: All corners with region coloring
    cv::rectangle(vis_all, matlab_rect, cv::Scalar(0, 255, 255), 3);
    cv::putText(vis_all, "MATLAB Region", cv::Point(50, 340), 
               cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);
    
    for (int i = 0; i < corners.p.size(); ++i) {
        cv::Point2f pt = corners.p[i];
        cv::Scalar color;
        
        if (pt.y < 200) {
            color = cv::Scalar(0, 0, 255);      // Red - Upper region
        } else if (pt.y >= 200 && pt.y < 350) {
            color = cv::Scalar(255, 255, 0);    // Cyan - Middle region
        } else if (matlab_rect.contains(pt)) {
            color = cv::Scalar(0, 255, 0);      // Green - MATLAB region
        } else {
            color = cv::Scalar(255, 0, 255);    // Magenta - Lower region
        }
        
        cv::circle(vis_all, pt, 8, color, 2);
        cv::circle(vis_all, pt, 3, cv::Scalar(255, 255, 255), -1);
    }
    
    // Add statistics text
    cv::putText(vis_all, "Total: " + std::to_string(corners.p.size()), 
               cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    cv::putText(vis_all, "MATLAB: " + std::to_string(stats.matlab_region), 
               cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    cv::putText(vis_all, "Upper: " + std::to_string(stats.upper_region), 
               cv::Point(10, 90), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
    
    // Visualization 2: MATLAB region only
    cv::rectangle(vis_matlab, matlab_rect, cv::Scalar(0, 255, 255), 3);
    
    int matlab_corner_count = 0;
    for (int i = 0; i < corners.p.size(); ++i) {
        cv::Point2f pt = corners.p[i];
        if (matlab_rect.contains(pt)) {
            cv::circle(vis_matlab, pt, 8, cv::Scalar(0, 255, 0), 2);
            cv::circle(vis_matlab, pt, 3, cv::Scalar(255, 255, 255), -1);
            cv::putText(vis_matlab, std::to_string(++matlab_corner_count), 
                       pt + cv::Point2f(10, -10), cv::FONT_HERSHEY_SIMPLEX, 0.4, 
                       cv::Scalar(255, 255, 255), 1);
        }
    }
    
    cv::putText(vis_matlab, "MATLAB Region: " + std::to_string(stats.matlab_region), 
               cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
    
    // Visualization 3: Detected chessboards
    vis_board = vis_all.clone();
    if (!boards.empty()) {
        for (const auto& board : boards) {
            for (const auto& idx_vec : board.idx) {
                for (int idx : idx_vec) {
                    if (idx < corners.p.size()) {
                        cv::Point2f pt = corners.p[idx];
                        cv::circle(vis_board, pt, 12, cv::Scalar(255, 255, 0), 3);
                    }
                }
            }
        }
    }
    
    cv::putText(vis_board, "Boards: " + std::to_string(boards.size()), 
               cv::Point(10, 120), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 0), 2);
    
    // Combine visualizations
    int width = vis_all.cols;
    int height = vis_all.rows;
    vis_combined = cv::Mat::zeros(height * 2, width * 2, CV_8UC3);
    
    vis_all.copyTo(vis_combined(cv::Rect(0, 0, width, height)));
    vis_matlab.copyTo(vis_combined(cv::Rect(width, 0, width, height)));
    vis_board.copyTo(vis_combined(cv::Rect(0, height, width, height)));
    
    // Add titles
    cv::putText(vis_combined, "All Corners (" + std::to_string(corners.p.size()) + ")", 
               cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
    cv::putText(vis_combined, "MATLAB Region (" + std::to_string(stats.matlab_region) + ")", 
               cv::Point(width + 10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
    cv::putText(vis_combined, "Chessboards (" + std::to_string(boards.size()) + ")", 
               cv::Point(10, height + 25), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
    
    std::string save_path = "result/cpp_debug_" + title + ".png";
    cv::imwrite(save_path, vis_combined);
    std::cout << "Debug visualization saved to: " << save_path << std::endl;
}

int main(int argc, char* argv[]) {
    std::string image_path = "data/04.png";
    if (argc > 1) {
        image_path = argv[1];
    }
    
    DebugLogger logger("result/cpp_debug_log.txt");
    
    logger.log("\n=== C++ DEBUG VERSION - DETAILED MODULE ANALYSIS ===\n");
    logger.log("Image: " + image_path + "\n");
    
    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    if (img.empty()) {
        logger.log("Error: Could not load image " + image_path + "\n");
        return -1;
    }
    
    logger.log("Image size: " + std::to_string(img.cols) + " x " + std::to_string(img.rows) + "\n");
    logger.log("Image channels: " + std::to_string(img.channels()) + "\n");
    
    // Test multiple configurations for comparison
    std::vector<std::pair<std::string, cbdetect::Params>> configs = {
        {"Original", []() {
            cbdetect::Params p;
            p.detect_method = cbdetect::TemplateMatchFast;
            p.norm = true;
            p.norm_half_kernel_size = 31;
            p.init_loc_thr = 0.012;
            p.score_thr = 0.025;
            p.radius = {6, 8};
            p.polynomial_fit = true;
            p.show_processing = false;
            return p;
        }()},
        
        {"Conservative", []() {
            cbdetect::Params p;
            p.detect_method = cbdetect::TemplateMatchFast;
            p.norm = true;
            p.norm_half_kernel_size = 31;
            p.init_loc_thr = 0.02;
            p.score_thr = 0.05;
            p.radius = {6, 7};
            p.polynomial_fit = true;
            p.show_processing = false;
            return p;
        }()},
        
        {"HessianResponse", []() {
            cbdetect::Params p;
            p.detect_method = cbdetect::HessianResponse;
            p.norm = false;
            p.init_loc_thr = 0.1;
            p.score_thr = 0.1;
            p.radius = {7};
            p.polynomial_fit = true;
            p.show_processing = false;
            return p;
        }()}
    };
    
    for (const auto& config : configs) {
        const std::string& config_name = config.first;
        const cbdetect::Params& params = config.second;
        
        logger.log("\n" + std::string(60, '=') + "\n");
        logger.log("TESTING CONFIGURATION: " + config_name + "\n");
        logger.log(std::string(60, '=') + "\n");
        
        // Step 1: Corner Detection
        logger.log("\n=== STEP 1: CORNER DETECTION ===\n");
        logger.log("Configuration: " + config_name + "\n");
        logger.log("Parameters:\n");
        logger.log("  detect_method: " + std::to_string(params.detect_method) + "\n");
        logger.log("  norm: " + std::string(params.norm ? "true" : "false") + "\n");
        logger.log("  init_loc_thr: " + std::to_string(params.init_loc_thr) + "\n");
        logger.log("  score_thr: " + std::to_string(params.score_thr) + "\n");
        logger.log("  radius: [");
        for (int i = 0; i < params.radius.size(); ++i) {
            logger.log(std::to_string(params.radius[i]));
            if (i < params.radius.size() - 1) logger.log(", ");
        }
        logger.log("]\n");
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        cbdetect::Corner corners;
        cbdetect::find_corners(img, corners, params);
        
        auto corner_end_time = std::chrono::high_resolution_clock::now();
        auto corner_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            corner_end_time - start_time).count();
        
        logger.log("Corner detection completed in " + std::to_string(corner_duration) + " ms\n");
        logger.log("Raw corners detected: " + std::to_string(corners.p.size()) + "\n");
        
        // Analyze regions
        RegionStats stats = analyze_corner_regions(corners, logger);
        
        logger.log("\nRegion distribution:\n");
        logger.log("  Upper region (Y<200):     " + std::to_string(stats.upper_region) + 
                  " corners (" + std::to_string(100.0 * stats.upper_region / corners.p.size()) + "%)\n");
        logger.log("  Middle region (Y200-350): " + std::to_string(stats.middle_region) + 
                  " corners (" + std::to_string(100.0 * stats.middle_region / corners.p.size()) + "%)\n");
        logger.log("  MATLAB region:            " + std::to_string(stats.matlab_region) + 
                  " corners (" + std::to_string(100.0 * stats.matlab_region / corners.p.size()) + "%)\n");
        logger.log("  Lower region (Y>350):     " + std::to_string(stats.lower_region) + 
                  " corners (" + std::to_string(100.0 * stats.lower_region / corners.p.size()) + "%)\n");
        
        // Coordinate bounds
        if (!corners.p.empty()) {
            double min_x = corners.p[0].x, max_x = corners.p[0].x;
            double min_y = corners.p[0].y, max_y = corners.p[0].y;
            
            for (const auto& pt : corners.p) {
                min_x = std::min(min_x, (double)pt.x);
                max_x = std::max(max_x, (double)pt.x);
                min_y = std::min(min_y, (double)pt.y);
                max_y = std::max(max_y, (double)pt.y);
            }
            
            logger.log("\nCoordinate bounds:\n");
            logger.log("  X range: [" + std::to_string(min_x) + ", " + std::to_string(max_x) + "]\n");
            logger.log("  Y range: [" + std::to_string(min_y) + ", " + std::to_string(max_y) + "]\n");
        }
        
        // Score statistics
        if (!corners.score.empty()) {
            double min_score = *std::min_element(corners.score.begin(), corners.score.end());
            double max_score = *std::max_element(corners.score.begin(), corners.score.end());
            double avg_score = std::accumulate(corners.score.begin(), corners.score.end(), 0.0) / corners.score.size();
            int high_quality = std::count_if(corners.score.begin(), corners.score.end(), 
                                            [](double s) { return s > 1.0; });
            
            logger.log("\nScore statistics:\n");
            logger.log("  Score range: [" + std::to_string(min_score) + ", " + std::to_string(max_score) + "]\n");
            logger.log("  Average score: " + std::to_string(avg_score) + "\n");
            logger.log("  High quality corners (>1.0): " + std::to_string(high_quality) + "\n");
        }
        
        // Log corner structure
        log_corner_structure(corners, logger);
        
        // Step 3: Board detection
        auto board_start_time = std::chrono::high_resolution_clock::now();
        
        std::vector<cbdetect::Board> boards;
        cbdetect::boards_from_corners(img, corners, boards, params);
        
        auto board_end_time = std::chrono::high_resolution_clock::now();
        auto board_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            board_end_time - board_start_time).count();
        
        logger.log("Chessboard detection completed in " + std::to_string(board_duration) + " ms\n");
        
        log_board_details(boards, corners, logger);
        
        // Step 4: Final Results
        logger.log("\n=== STEP 4: FINAL RESULTS ===\n");
        
        auto total_duration = corner_duration + board_duration;
        logger.log("Total processing time: " + std::to_string(total_duration) + " ms\n");
        
        logger.log("\nFinal Summary for " + config_name + ":\n");
        logger.log("  Total corners: " + std::to_string(corners.p.size()) + "\n");
        logger.log("  MATLAB region corners: " + std::to_string(stats.matlab_region) + "\n");
        logger.log("  Detected boards: " + std::to_string(boards.size()) + "\n");
        
        if (!boards.empty() && !boards[0].energy.empty() && !boards[0].energy[0].empty() && !boards[0].energy[0][0].empty()) {
            logger.log("  Best board energy: " + std::to_string(boards[0].energy[0][0][0]) + "\n");
        }
        
        logger.log("  Corner detection time: " + std::to_string(corner_duration) + " ms\n");
        logger.log("  Board detection time: " + std::to_string(board_duration) + " ms\n");
        logger.log("  Total time: " + std::to_string(total_duration) + " ms\n");
        
        // Create visualization
        create_debug_visualization(img, corners, boards, stats, config_name);
        
        // Comparison with MATLAB target
        logger.log("\n=== COMPARISON WITH MATLAB TARGET ===\n");
        logger.log("MATLAB target: 51 corners in target region\n");
        logger.log("Our result: " + std::to_string(stats.matlab_region) + " corners in target region\n");
        logger.log("Difference: " + std::to_string(stats.matlab_region - 51) + " corners\n");
        
        double accuracy = 100.0 * stats.matlab_region / 51.0;
        logger.log("Accuracy: " + std::to_string(accuracy) + "% of MATLAB target\n");
        
        if (std::abs(stats.matlab_region - 51) <= 5) {
            logger.log("🎯 EXCELLENT: Very close to MATLAB target!\n");
        } else if (std::abs(stats.matlab_region - 51) <= 15) {
            logger.log("✅ GOOD: Reasonably close to MATLAB target\n");
        } else if (stats.matlab_region >= 25) {
            logger.log("⚠️ MODERATE: Some improvement over baseline\n");
        } else {
            logger.log("❌ POOR: Significant difference from MATLAB target\n");
        }
    }
    
    logger.log("\n" + std::string(60, '=') + "\n");
    logger.log("C++ DETAILED DEBUG ANALYSIS COMPLETE\n");
    logger.log(std::string(60, '=') + "\n");
    
    return 0;
} 