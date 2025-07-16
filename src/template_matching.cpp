#include "cbdetect/template_matching.h"
#include <cmath>
#include <algorithm>

namespace cbdetect {

// CorrelationTemplate implementation
CorrelationTemplate CorrelationTemplate::create(double angle1, double angle2, int radius) {
    CorrelationTemplate tmpl;
    
    int size = 2 * radius + 1;
    tmpl.a1 = cv::Mat::zeros(size, size, CV_64F);
    tmpl.a2 = cv::Mat::zeros(size, size, CV_64F);
    tmpl.b1 = cv::Mat::zeros(size, size, CV_64F);
    tmpl.b2 = cv::Mat::zeros(size, size, CV_64F);
    
    // Precompute sine and cosine values
    double cos_angle1 = std::cos(angle1);
    double sin_angle1 = std::sin(angle1);
    double cos_angle2 = std::cos(angle2);
    double sin_angle2 = std::sin(angle2);
    
    // Create correlation template
    for (int u = -radius; u <= radius; ++u) {
        for (int v = -radius; v <= radius; ++v) {
            // Compute coordinates in template space
            double x = static_cast<double>(u);
            double y = static_cast<double>(v);
            
            // Distance from center
            double dist = std::sqrt(x * x + y * y);
            if (dist > radius) continue;
            
            // Weights based on distance (Gaussian-like)
            double weight = std::exp(-0.5 * (dist * dist) / (radius * radius / 4.0));
            
            // Compute dot products with principal directions
            double dot1 = x * cos_angle1 + y * sin_angle1;
            double dot2 = x * cos_angle2 + y * sin_angle2;
            
            // Assign to appropriate quadrant
            int row = v + radius;
            int col = u + radius;
            
            if (dot1 >= 0 && dot2 >= 0) {
                tmpl.a1.at<double>(row, col) = weight;
            } else if (dot1 < 0 && dot2 >= 0) {
                tmpl.a2.at<double>(row, col) = weight;
            } else if (dot1 >= 0 && dot2 < 0) {
                tmpl.b1.at<double>(row, col) = weight;
            } else {
                tmpl.b2.at<double>(row, col) = weight;
            }
        }
    }
    
    // Normalize templates
    double sum_a1 = cv::sum(tmpl.a1)[0];
    double sum_a2 = cv::sum(tmpl.a2)[0];
    double sum_b1 = cv::sum(tmpl.b1)[0];
    double sum_b2 = cv::sum(tmpl.b2)[0];
    
    if (sum_a1 > 0) tmpl.a1 /= sum_a1;
    if (sum_a2 > 0) tmpl.a2 /= sum_a2;
    if (sum_b1 > 0) tmpl.b1 /= sum_b1;
    if (sum_b2 > 0) tmpl.b2 /= sum_b2;
    
    return tmpl;
}

// TemplateCornerDetector implementation
std::vector<std::vector<double>> TemplateCornerDetector::getTemplateProperties(const std::vector<int>& radii) {
    std::vector<std::vector<double>> props;
    
    // Template orientations based on MATLAB implementation
    for (int r : radii) {
        // Two main orientations for each radius
        props.push_back({0.0, M_PI_2, static_cast<double>(r)});          // Horizontal/Vertical
        props.push_back({M_PI_4, -M_PI_4, static_cast<double>(r)});      // Diagonal
    }
    
    return props;
}

void TemplateCornerDetector::createTemplates(const std::vector<int>& radii) {
    templates_.clear();
    
    auto props = getTemplateProperties(radii);
    for (const auto& prop : props) {
        double angle1 = prop[0];
        double angle2 = prop[1];
        int radius = static_cast<int>(prop[2]);
        
        templates_.push_back(CorrelationTemplate::create(angle1, angle2, radius));
    }
}

cv::Mat TemplateCornerDetector::applyTemplateMatching(const cv::Mat& img, const CorrelationTemplate& tmpl) {
    cv::Mat img_corners_a1, img_corners_a2, img_corners_b1, img_corners_b2;
    cv::Mat img_corners_mu, img_corners_a, img_corners_b, img_corners_s1, img_corners_s2;
    
    // Filter image with current template
    cv::filter2D(img, img_corners_a1, -1, tmpl.a1, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
    cv::filter2D(img, img_corners_a2, -1, tmpl.a2, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
    cv::filter2D(img, img_corners_b1, -1, tmpl.b1, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
    cv::filter2D(img, img_corners_b2, -1, tmpl.b2, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
    
    // Compute mean
    img_corners_mu = (img_corners_a1 + img_corners_a2 + img_corners_b1 + img_corners_b2) / 4.0;
    
    // Case 1: a=white, b=black
    img_corners_a = cv::min(img_corners_a1, img_corners_a2) - img_corners_mu;
    img_corners_b = img_corners_mu - cv::max(img_corners_b1, img_corners_b2);
    img_corners_s1 = cv::min(img_corners_a, img_corners_b);
    
    // Case 2: b=white, a=black
    img_corners_a = img_corners_mu - cv::max(img_corners_a1, img_corners_a2);
    img_corners_b = cv::min(img_corners_b1, img_corners_b2) - img_corners_mu;
    img_corners_s2 = cv::min(img_corners_a, img_corners_b);
    
    // Combine both cases
    return cv::max(img_corners_s1, img_corners_s2);
}

cv::Mat TemplateCornerDetector::detectCorners(const cv::Mat& img, const std::vector<int>& radii) {
    // Create templates if not already created
    if (templates_.empty()) {
        createTemplates(radii);
    }
    
    cv::Mat img_corners = cv::Mat::zeros(img.size(), CV_64F);
    
    // Apply each template and combine results
    for (const auto& tmpl : templates_) {
        cv::Mat response = applyTemplateMatching(img, tmpl);
        img_corners = cv::max(img_corners, response);
    }
    
    return img_corners;
}

// NonMaximumSuppression implementation
bool NonMaximumSuppression::isLocalMaximum(const cv::Mat& response, int x, int y, int radius) {
    double center_val = response.at<double>(y, x);
    
    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            if (dx == 0 && dy == 0) continue;
            
            int nx = x + dx;
            int ny = y + dy;
            
            if (nx >= 0 && nx < response.cols && ny >= 0 && ny < response.rows) {
                if (response.at<double>(ny, nx) > center_val) {
                    return false;
                }
            }
        }
    }
    
    return true;
}

std::vector<cv::Point2d> NonMaximumSuppression::apply(const cv::Mat& response, 
                                                     int radius, 
                                                     double threshold, 
                                                     int margin) {
    std::vector<cv::Point2d> corners;
    
    for (int y = margin; y < response.rows - margin; ++y) {
        for (int x = margin; x < response.cols - margin; ++x) {
            double val = response.at<double>(y, x);
            
            if (val > threshold && isLocalMaximum(response, x, y, radius)) {
                corners.emplace_back(x, y);
            }
        }
    }
    
    return corners;
}

// HessianCornerDetector implementation
void HessianCornerDetector::computeHessianResponse(const cv::Mat& img, cv::Mat& response) {
    const int rows = img.rows;
    const int cols = img.cols;
    
    response = cv::Mat::zeros(rows, cols, CV_64F);
    
    // Compute Hessian determinant for each pixel
    for (int y = 1; y < rows - 1; ++y) {
        for (int x = 1; x < cols - 1; ++x) {
            // Compute second derivatives using finite differences
            double Lxx = img.at<double>(y, x-1) - 2*img.at<double>(y, x) + img.at<double>(y, x+1);
            double Lyy = img.at<double>(y-1, x) - 2*img.at<double>(y, x) + img.at<double>(y+1, x);
            double Lxy = (img.at<double>(y-1, x-1) - img.at<double>(y-1, x+1) + 
                         img.at<double>(y+1, x+1) - img.at<double>(y+1, x-1)) / 4.0;
            
            // Hessian determinant
            response.at<double>(y, x) = Lxx * Lyy - Lxy * Lxy;
        }
    }
}

cv::Mat HessianCornerDetector::detectCorners(const cv::Mat& img) {
    cv::Mat response;
    computeHessianResponse(img, response);
    return response;
}

} // namespace cbdetect 