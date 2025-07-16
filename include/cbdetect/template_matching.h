#ifndef CBDETECT_TEMPLATE_MATCHING_H
#define CBDETECT_TEMPLATE_MATCHING_H

#include <opencv2/opencv.hpp>
#include <vector>

namespace cbdetect {

/**
 * @brief Correlation template for corner detection
 */
struct CorrelationTemplate {
    cv::Mat a1, a2, b1, b2;  // Four quadrant templates
    
    CorrelationTemplate() = default;
    
    // Create template for given angles and radius
    static CorrelationTemplate create(double angle1, double angle2, int radius);
    
    bool empty() const { return a1.empty(); }
    void clear() { a1.release(); a2.release(); b1.release(); b2.release(); }
};

/**
 * @brief Template-based corner detector
 */
class TemplateCornerDetector {
public:
    TemplateCornerDetector() = default;
    
    // Main detection function
    cv::Mat detectCorners(const cv::Mat& img, const std::vector<int>& radii);
    
    // Create correlation templates for all scales and orientations
    void createTemplates(const std::vector<int>& radii);
    
    // Apply template matching
    cv::Mat applyTemplateMatching(const cv::Mat& img, const CorrelationTemplate& tmpl);
    
private:
    std::vector<CorrelationTemplate> templates_;
    
    // Template properties: [angle1, angle2, radius] for each template
    std::vector<std::vector<double>> getTemplateProperties(const std::vector<int>& radii);
};

/**
 * @brief Non-maximum suppression for corner detection
 */
class NonMaximumSuppression {
public:
    // Apply NMS and return corner positions
    static std::vector<cv::Point2d> apply(const cv::Mat& response, 
                                         int radius, 
                                         double threshold, 
                                         int margin);
    
    // Check if point is local maximum
    static bool isLocalMaximum(const cv::Mat& response, 
                              int x, int y, 
                              int radius);
};

/**
 * @brief Hessian-based corner detection
 */
class HessianCornerDetector {
public:
    // Compute Hessian response
    static void computeHessianResponse(const cv::Mat& img, cv::Mat& response);
    
    // Detect corners using Hessian determinant
    cv::Mat detectCorners(const cv::Mat& img);
};

} // namespace cbdetect

#endif // CBDETECT_TEMPLATE_MATCHING_H 