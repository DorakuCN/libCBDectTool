#include <iostream>
#include <opencv2/opencv.hpp>
#include "cbdetect/pipeline.h"
#include "cbdetect/cbdetect.h"

using namespace cv;
using namespace cbdetect;

int main(int argc, char** argv) {
    std::cout << "==================================" << std::endl;
    std::cout << "libcbdetect Pipeline API Demo" << std::endl;
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
    
    // Create detection parameters
    DetectionParams params;
    params.detect_method = DetectMethod::TEMPLATE_MATCH_FAST;
    params.corner_type = CornerType::SADDLE_POINT;
    params.corner_threshold = 0.01f;
    params.refine_corners = true;
    params.show_processing = true;
    params.show_debug_images = false;
    params.template_radii = {4, 8, 12};  // Use MATLAB version parameters
    params.energy_threshold = -10.0f;
    
    // Create pipeline options
    PipelineOptions options;
    options.expand = false;
    options.predict = true;  // Enable polynomial refinement
    options.out_of_image = false;
    options.polynomial_degree = 2;
    
    // Create pipeline
    Pipeline pipeline(params, options);
    
    std::cout << "Pipeline created with polynomial refinement enabled" << std::endl;
    std::cout << "Detection method: " << 
        (params.detect_method == DetectMethod::TEMPLATE_MATCH_FAST ? "Template Matching (Fast)" :
         params.detect_method == DetectMethod::HESSIAN_RESPONSE ? "Hessian Response" :
         "Harris Corners") << std::endl;
    
    // Detect chessboard
    std::cout << "Detecting chessboard with polynomial refinement..." << std::endl;
    auto start = getTickCount();
    
    auto result_tuple = pipeline.detect(image);
    int result = std::get<0>(result_tuple);
    cv::Mat board_uv = std::get<1>(result_tuple);
    cv::Mat board_xy = std::get<2>(result_tuple);
    
    auto end = getTickCount();
    double time_ms = (end - start) / getTickFrequency() * 1000.0;
    
    std::cout << "Detection completed in " << time_ms << " ms" << std::endl;
    
    // Process results
    if (result == 0) {
        std::cout << "No chessboard detected" << std::endl;
        return 0;
    }
    
    std::cout << "Chessboard detected successfully!" << std::endl;
    std::cout << "Result code: " << result << " (0=failure, 1=relative, 2=absolute)" << std::endl;
    std::cout << "Board UV points: " << board_uv.rows << "x" << board_uv.cols << std::endl;
    std::cout << "Board XY points: " << board_xy.rows << "x" << board_xy.cols << std::endl;
    // Debug: dump each detected corner (image uv -> board xy) for comparison with MATLAB
    std::cout << "DEBUG: UV -> XY per corner:\n";
    for (int i = 0; i < board_uv.rows; ++i) {
        double uv_u = board_uv.at<double>(i, 0);
        double uv_v = board_uv.at<double>(i, 1);
        double xy_x = board_xy.at<double>(i, 0);
        double xy_y = board_xy.at<double>(i, 1);
        std::cout << i << ": (" << uv_u << "," << uv_v << ") -> (" << xy_x << "," << xy_y << ")\n";
    }
    
    // Visualize results
    Mat result_image = image.clone();
    
    // Draw detected points
    for (int i = 0; i < board_uv.rows; ++i) {
        Point2d uv(board_uv.at<double>(i, 0), board_uv.at<double>(i, 1));
        Point2d xy(board_xy.at<double>(i, 0), board_xy.at<double>(i, 1));
        
        // Draw point
        circle(result_image, uv, 3, Scalar(0, 255, 0), -1);
        
        // Draw coordinate label
        std::string label = "(" + std::to_string(static_cast<int>(xy.x)) + 
                           "," + std::to_string(static_cast<int>(xy.y)) + ")";
        putText(result_image, label, uv + Point2d(5, -5), 
                FONT_HERSHEY_SIMPLEX, 0.3, Scalar(255, 255, 255), 1);
    }
    
    // Draw grid connections
    if (board_uv.rows > 0) {
        // Find grid dimensions
        double max_x = 0, max_y = 0;
        for (int i = 0; i < board_xy.rows; ++i) {
            max_x = std::max(max_x, board_xy.at<double>(i, 0));
            max_y = std::max(max_y, board_xy.at<double>(i, 1));
        }
        int cols = static_cast<int>(max_x) + 1;
        int rows = static_cast<int>(max_y) + 1;
        
        std::cout << "Grid dimensions: " << rows << "x" << cols << std::endl;
        
        // Draw horizontal lines
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols - 1; ++c) {
                Point2d p1, p2;
                bool found1 = false, found2 = false;
                
                // Find points for this row
                for (int i = 0; i < board_xy.rows; ++i) {
                    if (static_cast<int>(board_xy.at<double>(i, 1)) == r) {
                        if (static_cast<int>(board_xy.at<double>(i, 0)) == c) {
                            p1 = Point2d(board_uv.at<double>(i, 0), board_uv.at<double>(i, 1));
                            found1 = true;
                        } else if (static_cast<int>(board_xy.at<double>(i, 0)) == c + 1) {
                            p2 = Point2d(board_uv.at<double>(i, 0), board_uv.at<double>(i, 1));
                            found2 = true;
                        }
                    }
                }
                
                if (found1 && found2) {
                    line(result_image, p1, p2, Scalar(0, 0, 255), 2);
                }
            }
        }
        
        // Draw vertical lines
        for (int c = 0; c < cols; ++c) {
            for (int r = 0; r < rows - 1; ++r) {
                Point2d p1, p2;
                bool found1 = false, found2 = false;
                
                // Find points for this column
                for (int i = 0; i < board_xy.rows; ++i) {
                    if (static_cast<int>(board_xy.at<double>(i, 0)) == c) {
                        if (static_cast<int>(board_xy.at<double>(i, 1)) == r) {
                            p1 = Point2d(board_uv.at<double>(i, 0), board_uv.at<double>(i, 1));
                            found1 = true;
                        } else if (static_cast<int>(board_xy.at<double>(i, 1)) == r + 1) {
                            p2 = Point2d(board_uv.at<double>(i, 0), board_uv.at<double>(i, 1));
                            found2 = true;
                        }
                    }
                }
                
                if (found1 && found2) {
                    line(result_image, p1, p2, Scalar(0, 0, 255), 2);
                }
            }
        }
    }
    
    // Show results
    namedWindow("Pipeline Detection Results", WINDOW_AUTOSIZE);
    imshow("Pipeline Detection Results", result_image);
    
    // Save result
    std::string output_path = "result/pipeline_" + image_path;
    size_t slash_pos = output_path.find_last_of("/\\");
    if (slash_pos != std::string::npos) {
        output_path = output_path.substr(slash_pos + 1);
    }
    
    imwrite(output_path, result_image);
    std::cout << "Result saved to: " << output_path << std::endl;
    
    // Optional: Create dewarped image
    if (result > 0 && options.predict) {
        std::cout << "Creating dewarped image..." << std::endl;
        Mat dewarped = pipeline.dewarpImage(image, board_uv, board_xy, 20);
        
        if (!dewarped.empty()) {
            namedWindow("Dewarped Image", WINDOW_AUTOSIZE);
            imshow("Dewarped Image", dewarped);
            
            std::string dewarped_path = "result/dewarped_" + image_path;
            slash_pos = dewarped_path.find_last_of("/\\");
            if (slash_pos != std::string::npos) {
                dewarped_path = dewarped_path.substr(slash_pos + 1);
            }
            
            imwrite(dewarped_path, dewarped);
            std::cout << "Dewarped image saved to: " << dewarped_path << std::endl;
        }
    }
    
    // Wait for key press
    std::cout << "Press any key to exit..." << std::endl;
    waitKey(0);
    
    return 0;
} 