#include "cbdetect/subpixel_refinement.h"
#include <opencv2/opencv.hpp>
#include <cmath>

namespace cbdetect {

// Helper for single corner refinement
static void refineCorner(const cv::Mat& image, Corner& corner, int max_iterations) {
    const int window_size = 7; // 7x7 window for refinement
    const int half_window = window_size / 2;
    cv::Point2d refined_pos = corner.pt;
    for (int iter = 0; iter < max_iterations; ++iter) {
        int x = static_cast<int>(std::round(refined_pos.x));
        int y = static_cast<int>(std::round(refined_pos.y));
        if (x < half_window || x >= image.cols - half_window ||
            y < half_window || y >= image.rows - half_window) {
            break;
        }
        cv::Mat window = image(cv::Rect(x - half_window, y - half_window, window_size, window_size));
        // Compute gradients
        cv::Mat grad_x, grad_y;
        cv::Sobel(window, grad_x, CV_64F, 1, 0, 3);
        cv::Sobel(window, grad_y, CV_64F, 0, 1, 3);
        // Compute Hessian and gradient
        cv::Mat hessian = cv::Mat::zeros(2, 2, CV_64F);
        cv::Mat gradient = cv::Mat::zeros(2, 1, CV_64F);
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0) continue;
                int sample_x = half_window + dx;
                int sample_y = half_window + dy;
                if (sample_x < 0 || sample_x >= window_size || sample_y < 0 || sample_y >= window_size) continue;
                double gx = grad_x.at<double>(sample_y, sample_x);
                double gy = grad_y.at<double>(sample_y, sample_x);
                hessian.at<double>(0, 0) += gx * gx;
                hessian.at<double>(0, 1) += gx * gy;
                hessian.at<double>(1, 0) += gy * gx;
                hessian.at<double>(1, 1) += gy * gy;
                gradient.at<double>(0, 0) += gx;
                gradient.at<double>(1, 0) += gy;
            }
        }
        // Solve for displacement: H * delta = -gradient
        cv::Mat delta;
        cv::solve(hessian, -gradient, delta, cv::DECOMP_SVD);
        cv::Point2d displacement(delta.at<double>(0, 0), delta.at<double>(1, 0));
        refined_pos += displacement;
        if (cv::norm(displacement) < 0.01) {
            break;
        }
    }
    corner.pt = refined_pos;
}

void refineCorners(const cv::Mat& image, Corners& corners, int max_iterations) {
    if (corners.empty() || image.empty()) return;
    cv::Mat img_gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, img_gray, cv::COLOR_BGR2GRAY);
    } else {
        img_gray = image.clone();
    }
    img_gray.convertTo(img_gray, CV_64F, 1.0 / 255.0);
    for (auto& corner : corners) {
        refineCorner(img_gray, corner, max_iterations);
    }
}

} // namespace cbdetect 