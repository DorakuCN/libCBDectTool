#ifndef CBDETECT_CORNER_H
#define CBDETECT_CORNER_H

#include <opencv2/opencv.hpp>
#include <vector>

namespace cbdetect {

/**
 * @brief Corner structure representing detected corner points
 */
struct Corner {
    cv::Point2d pt;           // Sub-pixel corner position
    cv::Vec2f v1, v2, v3;     // Direction vectors (v1, v2 primary; v3 for deltille)
    double radius = 1.0;      // Effective corner radius
    double quality_score = 0.0;  // Corner quality score (like MATLAB's corners.score)
    
    Corner() = default;
    Corner(double x, double y) : pt(x, y) {}
    Corner(const cv::Point2f& p) : pt(p.x, p.y) {}
    Corner(const cv::Point2d& p) : pt(p) {}
};

/**
 * @brief Collection of detected corners
 */
class Corners {
public:
    std::vector<Corner> corners;
    
    void clear();
    size_t size() const;
    bool empty() const;
    
    // Access operators
    Corner& operator[](size_t index);
    const Corner& operator[](size_t index) const;
    
    // Iterator support
    std::vector<Corner>::iterator begin();
    std::vector<Corner>::iterator end();
    std::vector<Corner>::const_iterator begin() const;
    std::vector<Corner>::const_iterator end() const;
    
    // Add corner
    void push_back(const Corner& corner);
    
    // Remove corners with low scores
    void filterByScore(float threshold);
    
    // Convert to OpenCV format for visualization
    std::vector<cv::Point2f> getPoints() const;
    std::vector<cv::Vec2f> getDirections1() const;
    std::vector<cv::Vec2f> getDirections2() const;
    std::vector<float> getScores() const;
};

} // namespace cbdetect

#endif // CBDETECT_CORNER_H 