#pragma once

#include <opencv2/opencv.hpp>
#include "corner.h"

namespace cbdetect {

/**
 * @brief Refine corner locations and orientations to subpixel accuracy.
 *        This implements a simplified version of MATLABs refineCorners.m.
 * @param image Grayscale image used for refinement (CV_64F)
 * @param corners Corner list to refine (modified in place)
 * @param max_iterations Maximum iterations for the refinement process
 */
void refineCorners(const cv::Mat& image, Corners& corners, int max_iterations = 10);

} // namespace cbdetect
 