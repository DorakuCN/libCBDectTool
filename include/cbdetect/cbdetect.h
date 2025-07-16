#ifndef CBDETECT_H
#define CBDETECT_H

/**
 * @file cbdetect.h
 * @brief Main header file for libcbdetect - Chessboard Detection Library
 * 
 * This library provides robust chessboard detection for camera calibration.
 * It is a C++ port of the MATLAB libcbdetect library by Andreas Geiger.
 * 
 * @author Your Name
 * @date 2024
 * @copyright GPL v3
 */

// Core data structures
#include "corner.h"
#include "chessboard.h"

// Main detection interface  
#include "chessboard_detector.h"

// Convenience namespace
namespace cbdetect {

/**
 * @brief Simple convenience function for chessboard detection
 * @param image Input image (grayscale or color)
 * @param corner_threshold Minimum corner quality threshold (default: 0.01)
 * @param refine_corners Whether to perform subpixel refinement (default: true)
 * @return Detected chessboards
 */
inline Chessboards detect(const cv::Mat& image, 
                         float corner_threshold = 0.01f, 
                         bool refine_corners = true) {
    DetectionParams params;
    params.corner_threshold = corner_threshold;
    params.refine_corners = refine_corners;
    
    ChessboardDetector detector(params);
    return detector.detectChessboards(image);
}

/**
 * @brief Get version information
 */
struct Version {
    static constexpr int MAJOR = 1;
    static constexpr int MINOR = 0;
    static constexpr int PATCH = 0;
    
    static std::string getString() {
        return std::to_string(MAJOR) + "." + 
               std::to_string(MINOR) + "." + 
               std::to_string(PATCH);
    }
};

} // namespace cbdetect

#endif // CBDETECT_H 