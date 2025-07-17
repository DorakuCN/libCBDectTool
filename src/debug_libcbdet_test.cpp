#include "cbdetect/libcbdetect_adapter.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <iomanip>

void print_corner_info(const cbdetect::Corner& corners, const std::string& stage, int max_corners = 10) {
    std::cout << "\n=== " << stage << " ===" << std::endl;
    std::cout << "Total corners: " << corners.p.size() << std::endl;
    
    if (corners.p.empty()) {
        std::cout << "No corners detected!" << std::endl;
        return;
    }
    
    std::cout << "Showing first " << std::min(max_corners, (int)corners.p.size()) << " corners:" << std::endl;
    for (int i = 0; i < std::min(max_corners, (int)corners.p.size()); ++i) {
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "  [" << i << "] pt=(" << corners.p[i].x << "," << corners.p[i].y << ")";
        std::cout << " r=" << corners.r[i];
        if (i < corners.v1.size()) {
            std::cout << " v1=(" << corners.v1[i].x << "," << corners.v1[i].y << ")";
        }
        if (i < corners.v2.size()) {
            std::cout << " v2=(" << corners.v2[i].x << "," << corners.v2[i].y << ")";
        }
        if (i < corners.score.size()) {
            std::cout << " score=" << corners.score[i];
        }
        std::cout << std::endl;
    }
}

void visualize_corners(const cv::Mat& img, const cbdetect::Corner& corners, const std::string& title) {
    cv::Mat vis = img.clone();
    if (vis.channels() == 1) {
        cv::cvtColor(vis, vis, cv::COLOR_GRAY2BGR);
    }
    
    for (int i = 0; i < corners.p.size(); ++i) {
        cv::Point2f pt(corners.p[i].x, corners.p[i].y);
        cv::circle(vis, pt, corners.r[i], cv::Scalar(0, 255, 0), 2);
        cv::circle(vis, pt, 2, cv::Scalar(0, 0, 255), -1);
        
        // Draw corner index
        cv::putText(vis, std::to_string(i), pt + cv::Point2f(5, -5),
                   cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 0), 1);
        
        // Draw direction vectors if available
        if (i < corners.v1.size() && (corners.v1[i].x != 0 || corners.v1[i].y != 0)) {
            cv::Point2f end1 = pt + cv::Point2f(corners.v1[i].x * 20, corners.v1[i].y * 20);
            cv::arrowedLine(vis, pt, end1, cv::Scalar(255, 0, 0), 1);
        }
        if (i < corners.v2.size() && (corners.v2[i].x != 0 || corners.v2[i].y != 0)) {
            cv::Point2f end2 = pt + cv::Point2f(corners.v2[i].x * 20, corners.v2[i].y * 20);
            cv::arrowedLine(vis, pt, end2, cv::Scalar(0, 255, 255), 1);
        }
    }
    
    cv::imshow(title, vis);
    cv::waitKey(0);
}

int main(int argc, char* argv[]) {
    std::string image_path = "data/04.png";
    if (argc > 1) {
        image_path = argv[1];
    }
    
    std::cout << "=== libcdetSample Compatible Algorithm Test ===" << std::endl;
    std::cout << "Testing with image: " << image_path << std::endl;
    
    // Load image
    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Error: Could not load image " << image_path << std::endl;
        return -1;
    }
    
    std::cout << "Image size: " << img.cols << "x" << img.rows << std::endl;
    
    // Test different detection methods
    std::vector<std::pair<cbdetect::DetectMethod, std::string>> methods = {
        {cbdetect::HessianResponse, "HessianResponse"},
        {cbdetect::TemplateMatchFast, "TemplateMatchFast"},
        {cbdetect::TemplateMatchSlow, "TemplateMatchSlow"}
    };
    
    for (const auto& method_pair : methods) {
        std::cout << "\n" << std::string(50, '=') << std::endl;
        std::cout << "Testing Method: " << method_pair.second << std::endl;
        std::cout << std::string(50, '=') << std::endl;
        
        cbdetect::Params params;
        params.detect_method = method_pair.first;
        params.show_processing = true;
        params.show_debug_image = false;
        params.polynomial_fit = true;
        params.norm = false;  // Start without normalization
        params.init_loc_thr = 0.01;
        params.score_thr = 0.01;
        params.radius = {4, 6, 8};  // Multi-scale radii
        
        cbdetect::Corner corners;
        std::vector<cbdetect::Board> boards;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Run corner detection
        cbdetect::find_corners(img, corners, params);
        
        auto corner_time = std::chrono::high_resolution_clock::now();
        
        // Run board detection
        cbdetect::boards_from_corners(img, corners, boards, params);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        
        // Timing results
        auto corner_duration = std::chrono::duration_cast<std::chrono::milliseconds>(corner_time - start_time);
        auto board_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - corner_time);
        auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        std::cout << "\n=== Results Summary ===" << std::endl;
        std::cout << "Corner detection time: " << corner_duration.count() << " ms" << std::endl;
        std::cout << "Board detection time: " << board_duration.count() << " ms" << std::endl;
        std::cout << "Total time: " << total_duration.count() << " ms" << std::endl;
        std::cout << "Final corners: " << corners.p.size() << std::endl;
        std::cout << "Boards detected: " << boards.size() << std::endl;
        
        // Print detailed corner information
        print_corner_info(corners, "Final Results", 15);
        
        // Visualize results
        visualize_corners(img, corners, method_pair.second + " - Corner Detection");
        
        if (!boards.empty()) {
            std::cout << "\nBoard details:" << std::endl;
            for (int i = 0; i < boards.size(); ++i) {
                std::cout << "  Board " << i << ": " << boards[i].idx.size() << "x";
                if (!boards[i].idx.empty()) {
                    std::cout << boards[i].idx[0].size();
                }
                std::cout << " grid" << std::endl;
            }
        }
        
        // Compare with expected MATLAB results
        std::cout << "\n=== MATLAB Comparison ===" << std::endl;
        std::cout << "MATLAB baseline: 51 corners → 1 chessboard (7x6)" << std::endl;
        std::cout << "libcdetSample baseline: ~39 corners → working" << std::endl;
        std::cout << "Current result: " << corners.p.size() << " corners → " << boards.size() << " boards" << std::endl;
        
        if (corners.p.size() >= 30 && corners.p.size() <= 60) {
            std::cout << "✅ Corner count within expected range!" << std::endl;
        } else {
            std::cout << "⚠️  Corner count outside expected range" << std::endl;
        }
        
        std::cout << "\nPress any key to continue to next method..." << std::endl;
        cv::waitKey(0);
    }
    
    std::cout << "\n=== Test Complete ===" << std::endl;
    cv::destroyAllWindows();
    return 0;
} 