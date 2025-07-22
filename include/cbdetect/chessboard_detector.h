#ifndef CBDETECT_CHESSBOARD_DETECTOR_H
#define CBDETECT_CHESSBOARD_DETECTOR_H

#include "corner.h"
#include "chessboard.h"
#include "template_matching.h"
#include "image_preprocessing.h"
#include <opencv2/opencv.hpp>
#include <memory>

namespace cbdetect {

/**
 * @brief Detection method options
 */
enum class DetectMethod {
    HARRIS_CORNER = 0,        // Harris corner detection (fast, basic)
    TEMPLATE_MATCH_FAST,      // Template matching fast mode
    TEMPLATE_MATCH_SLOW,      // Template matching slow mode (higher quality)
    HESSIAN_RESPONSE,         // Hessian response detection
    LOCALIZED_RADON_TRANSFORM // Localized Radon transform (highest quality)
};

/**
 * @brief Corner type for detection
 */
enum class CornerType {
    SADDLE_POINT = 0,         // Regular chessboard corners
    MONKEY_SADDLE_POINT       // Deltille pattern corners
};

/**
 * @brief Configuration parameters for chessboard detection
 */
struct DetectionParams {
    // Detection method and type
    DetectMethod detect_method = DetectMethod::TEMPLATE_MATCH_FAST;
    CornerType corner_type = CornerType::SADDLE_POINT;
    
    // Corner detection parameters
    float corner_threshold = 0.001f;    // Minimum corner quality threshold (lowered for debugging)
    bool refine_corners = true;         // Whether to perform subpixel refinement
    int max_refinement_iterations = 10; // Maximum iterations for corner refinement
    float init_loc_threshold = 0.005f;  // Initial location threshold
    float score_threshold = 0.01f;      // Corner score threshold
    
    // Template matching parameters
    std::vector<int> template_radii = {4, 8, 12};  // Multi-scale template radii
    
    // Non-maximum suppression parameters
    int nms_radius = 3;                 // Radius for non-maximum suppression
    float nms_threshold = 0.025f;       // Threshold for NMS
    int nms_margin = 5;                 // Margin from image border
    
    // Chessboard detection parameters
    float energy_threshold = -10.0f;    // Minimum chessboard energy threshold
    float neighbor_distance_tolerance = 0.3f;  // Tolerance for neighbor distance consistency
    bool strict_grow = true;            // Strict growing policy for boards
    
    // Image processing parameters
    bool normalize_image = true;        // Whether to normalize input image
    bool polynomial_fit = true;         // Use polynomial fitting for subpixel accuracy
    int norm_half_kernel_size = 15;     // Half kernel size for normalization
    int polynomial_fit_half_kernel_size = 4;  // Half kernel size for polynomial fitting
    
    // Debug and visualization options
    bool show_processing = false;       // Show processing steps
    bool show_debug_images = false;     // Show debug images
    bool overlay_results = false;       // Overlay detection results
    
    // Performance options
    bool enable_parallel = true;        // Enable parallel processing
    int num_threads = -1;               // Number of threads (-1 = auto)
    
    // MATLAB matching options (debug)
    bool disable_zero_crossing_filter = false;  // Skip zero-crossing & multi-stage filter to match MATLAB findCorners
    
    // Image preprocessing parameters
    bool enable_image_preprocessing = true;     // Enable image preprocessing
    PreprocessingParams preprocessing_params;   // Image preprocessing parameters
    
    DetectionParams() = default;
};

/**
 * @brief Main chessboard detection class
 */
class ChessboardDetector {
private:
    DetectionParams params_;
    
    // Internal processing images
    cv::Mat img_gray_;
    cv::Mat img_du_;        // Horizontal gradient
    cv::Mat img_dv_;        // Vertical gradient
    cv::Mat img_angle_;     // Gradient angle
    cv::Mat img_weight_;    // Gradient magnitude
    cv::Mat img_corners_;   // Corner response map
    
    // Corner detection algorithms
    std::unique_ptr<TemplateCornerDetector> template_detector_;
    std::unique_ptr<HessianCornerDetector> hessian_detector_;
    
    // Image preprocessor
    std::unique_ptr<ImagePreprocessor> image_preprocessor_;
    
    // Compute quality score for a single corner (like MATLAB's cornerCorrelationScore)
    double computeCornerQualityScore(const Corner& corner, int radius);
    
public:
    explicit ChessboardDetector(const DetectionParams& params = DetectionParams());
    
    // Main detection interface
    Chessboards detectChessboards(const cv::Mat& image);
    
    // Step-by-step detection (for debugging and custom workflows)
    Corners findCorners(const cv::Mat& image);
    Chessboards chessboardsFromCorners(const Corners& corners);
    
    // Configuration
    void setParams(const DetectionParams& params);
    const DetectionParams& getParams() const;
    
    // Utility functions for visualization
    static void drawCorners(cv::Mat& image, const Corners& corners, 
                           const cv::Scalar& color = cv::Scalar(0, 255, 0), int radius = 3);
    static void drawChessboards(cv::Mat& image, const Chessboards& chessboards, 
                               const Corners& corners, const cv::Scalar& color = cv::Scalar(0, 0, 255));
    
    // Post-processing functions
    static void filterDuplicateChessboards(Chessboards& chessboards, const Corners& corners);
    static double computeChessboardOverlap(const Chessboard& cb1, const Chessboard& cb2, const Corners& corners);
    
    // Get internal processing results (for debugging)
    const cv::Mat& getCornerResponse() const { return img_corners_; }
    const cv::Mat& getGradientAngle() const { return img_angle_; }
    const cv::Mat& getGradientMagnitude() const { return img_weight_; }
    
private:
    // Main detection methods
    void preprocessImage(const cv::Mat& image);
    void computeGradients();
    std::vector<cv::Point2d> detectCorners(const cv::Mat& image);
    void detectCornerCandidates();
    Corners extractCorners();
    void refineCorners(Corners& corners);
    void scoreCorners(Corners& corners);
    void filterCorners(Corners& corners);
    
    // libcdetSample-style helper methods
    void createCorrelationPatch(std::vector<cv::Mat>& templates, double angle1, double angle2, int radius);
    std::vector<cv::Point2d> applyNonMaximumSuppression(const cv::Mat& corner_map, int radius, double threshold);
    void polynomialFitValidation(Corners& corners);
    
    // Multi-scale detection functions
    Corners findCornersAtScale(const cv::Mat& image, double scale);
    Corners mergeMultiScaleCorners(const Corners& corners_orig, const Corners& corners_small);
    
    // Chessboard structure recovery and validation
    Chessboard initChessboard(const Corners& corners, int seed_idx);
    Chessboard growChessboard(const Chessboard& chessboard, const Corners& corners, int direction);
    Chessboards recoverStructure(const Corners& corners);  // Main structure recovery function
    float computeChessboardEnergy(const Chessboard& chessboard, const Corners& corners);
    
    // Utility functions
    // Find the best neighbor in a given direction (based on MATLAB's directionalNeighbor)
    int findDirectionalNeighbor(int corner_idx, const cv::Vec2f& direction, 
                               const Chessboard& chessboard, const Corners& corners,
                               double* out_distance = nullptr);
    
    // Optimized version with distance limit for better performance  
    int findDirectionalNeighborFast(int corner_idx, const cv::Vec2f& direction, 
                                   const Chessboard& chessboard, const Corners& corners,
                                   double* out_distance = nullptr, double max_distance = 50.0);
    bool isChessboardValid(const Chessboard& chessboard, const Corners& corners);
};

} // namespace cbdetect

#endif // CBDETECT_CHESSBOARD_DETECTOR_H 