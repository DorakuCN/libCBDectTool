#include <opencv2/opencv.hpp>

namespace cbdetect {

// Placeholder implementations for image processing functions

void normalizeImage(cv::Mat& image) {
    cv::normalize(image, image, 0.0, 1.0, cv::NORM_MINMAX);
}

void computeGradient(const cv::Mat& image, cv::Mat& dx, cv::Mat& dy) {
    cv::Sobel(image, dx, CV_64F, 1, 0, 3);
    cv::Sobel(image, dy, CV_64F, 0, 1, 3);
}

} // namespace cbdetect 