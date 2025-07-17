#include <cstdio>
#include <iostream>
#include "libcbdetect/config.h"
#include "libcbdetect/find_corners.h"
#include "libcbdetect/boards_from_corners.h"
#include "libcbdetect/plot_boards.h"
#include <opencv2/opencv.hpp>

void detect(const std::string& image_path, cbdetect::CornerType corner_type) {
    std::cout << "Loading image: " << image_path << std::endl;
    
    // Load image
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "Error: Could not load image " << image_path << std::endl;
        return;
    }
    
    std::cout << "Image loaded successfully (" << img.rows << "x" << img.cols << ")" << std::endl;
    
    // Set parameters
    cbdetect::Params params;
    params.corner_type = corner_type;
    params.show_processing = true;
    
    // Find corners
    cbdetect::Corner corners;
    cbdetect::find_corners(img, corners, params);
    
    std::cout << "Found " << corners.p.size() << " corners" << std::endl;
    
    // Output first 10 corners for comparison
    std::cout << "First 10 corners (if available):" << std::endl;
    for (size_t i = 0; i < std::min(corners.p.size(), size_t(10)); ++i) {
        std::cout << "  Corner " << (i+1) << ": pos=(" 
                  << corners.p[i].x << "," << corners.p[i].y << ")";
        if (i < corners.v1.size()) {
            std::cout << ", v1=(" << corners.v1[i].x << "," << corners.v1[i].y << ")";
        }
        if (i < corners.v2.size()) {
            std::cout << ", v2=(" << corners.v2[i].x << "," << corners.v2[i].y << ")";
        }
        if (i < corners.score.size()) {
            std::cout << ", score=" << corners.score[i];
        }
        std::cout << std::endl;
    }
    
    // Find chessboards
    std::vector<cbdetect::Board> boards;
    cbdetect::boards_from_corners(img, corners, boards, params);
    
    std::cout << "Found " << boards.size() << " chessboards" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Testing default image..." << std::endl;
        detect("example_data/04.png", cbdetect::SaddlePoint);
    } else {
        std::cout << "Testing provided image: " << argv[1] << std::endl;
        detect(argv[1], cbdetect::SaddlePoint);
    }
    return 0;
} 