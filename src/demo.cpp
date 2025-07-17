#include <iostream>
#include <opencv2/opencv.hpp>
#include "cbdetect/cbdetect.h"

using namespace cv;
using namespace cbdetect;

void drawResults(Mat& image, const Chessboards& chessboards, const Corners& corners) {
    // Draw detected corners
    ChessboardDetector::drawCorners(image, corners, Scalar(0, 255, 0), 3);
    
    // Draw detected chessboards
    ChessboardDetector::drawChessboards(image, chessboards, corners, Scalar(0, 0, 255));
    
    // Print detection statistics
    std::cout << "Detected " << corners.size() << " corners" << std::endl;
    std::cout << "Detected " << chessboards.size() << " chessboards" << std::endl;
    
    for (size_t i = 0; i < chessboards.size(); ++i) {
        const auto& cb = chessboards[i];
        std::cout << "Chessboard " << i + 1 << ": " 
                  << cb->rows() << "x" << cb->cols() 
                  << " corners, energy: " << cb->energy << std::endl;
    }
}

int main(int argc, char** argv) {
    std::cout << "==================================" << std::endl;
    std::cout << "libcbdetect C++ Demo" << std::endl;
    std::cout << "Version: " << Version::getString() << std::endl;
    std::cout << "==================================" << std::endl;
    
    // Parse command line arguments
    std::string image_path;
    if (argc >= 2) {
        image_path = argv[1];
        std::cout << "Processing image: " << image_path << std::endl;
    } else {
        image_path = "data/04.png";  // Default test image
        std::cout << "No image specified, using default: " << image_path << std::endl;
        std::cout << "Usage: " << argv[0] << " <image_path>" << std::endl;
    }
    
    // Load and validate image
    Mat image = imread(image_path, IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Error: Could not load image '" << image_path << "'" << std::endl;
        std::cerr << "Please check the file path and try again." << std::endl;
        return -1;
    }
    std::cout << "Image loaded successfully (" << image.cols << "x" << image.rows << ")" << std::endl;
    
    // Create detector with enhanced parameters
    DetectionParams params;
    params.detect_method = DetectMethod::TEMPLATE_MATCH_FAST;
    params.corner_type = CornerType::SADDLE_POINT;
    params.corner_threshold = 0.01f;  // Match MATLAB threshold
    params.refine_corners = true;
    params.show_processing = true;
    params.show_debug_images = false;
    
    // Use multiple scales for better detection (参考MATLAB版本)
    params.template_radii = {4, 8, 12};
    params.energy_threshold = -10.0f;
    
    std::cout << "Detection method: " << 
        (params.detect_method == DetectMethod::TEMPLATE_MATCH_FAST ? "Template Matching (Fast)" :
         params.detect_method == DetectMethod::HESSIAN_RESPONSE ? "Hessian Response" :
         "Harris Corners") << std::endl;
    std::cout << "Corner type: " << 
        (params.corner_type == CornerType::SADDLE_POINT ? "Saddle Point" : "Monkey Saddle Point") 
        << std::endl;
    
    ChessboardDetector detector(params);
    
    // Detect chessboards
    std::cout << "Detecting chessboards..." << std::endl;
    auto start = getTickCount();
    
    Chessboards chessboards = detector.detectChessboards(image);
    
    auto end = getTickCount();
    double time_ms = (end - start) / getTickFrequency() * 1000.0;
    
    std::cout << "Detection completed in " << time_ms << " ms" << std::endl;
    
    // For visualization, we also need the corners
    Corners corners = detector.findCorners(image);
    // DEBUG: dump final corner UV coordinates for comparison
    std::cout << "DEBUG: final corners (uv coords) [" << corners.size() << "]:\n";
    for (size_t i = 0; i < corners.size(); ++i) {
        std::cout << "  " << i << ": ("
                  << corners[i].pt.x << "," << corners[i].pt.y << ")\n";
    }
    
    // Draw results
    Mat result_image = image.clone();
    drawResults(result_image, chessboards, corners);
    
    // Show results
    namedWindow("Original Image", WINDOW_AUTOSIZE);
    namedWindow("Detection Results", WINDOW_AUTOSIZE);
    
    imshow("Original Image", image);
    imshow("Detection Results", result_image);
    
    // Save result
    std::string output_path = "result_" + image_path;
    size_t slash_pos = output_path.find_last_of("/\\");
    if (slash_pos != std::string::npos) {
        output_path = "result/" + output_path.substr(slash_pos + 1);
    }
    
    imwrite(output_path, result_image);
    std::cout << "Result saved to: " << output_path << std::endl;
    
    // Wait for key press
    std::cout << "Press any key to exit..." << std::endl;
    waitKey(0);
    
    return 0;
} 