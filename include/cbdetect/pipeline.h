#ifndef CBDETECT_PIPELINE_H
#define CBDETECT_PIPELINE_H

#include "cbdetect/chessboard_detector.h"
#include <opencv2/opencv.hpp>
#include <tuple>

namespace cbdetect {

struct PipelineOptions {
    bool expand = false;        // whether to expand detected board
    bool predict = false;       // whether to predict full grid
    bool out_of_image = false;  // allow predictions outside image
    int polynomial_degree = 2;  // polynomial degree for refinement
};

class Pipeline {
public:
    Pipeline(const DetectionParams& det_params = DetectionParams(),
             const PipelineOptions& options = PipelineOptions());

    /**
     * @brief Detect checkerboard and optionally refine with polynomial model.
     * @param image Input image (grayscale or color).
     * @param size Optional checkerboard size (rows, cols).
     * @return tuple: result code (0=failure,1=relative,2=absolute),
     *         board_uv (Nx2), board_xy (Nx2).
     */
    std::tuple<int, cv::Mat, cv::Mat> detect(const cv::Mat& image,
                                             const cv::Size& size = cv::Size());

    /**
     * @brief Dewarp image using polynomial model predicted from board.
     */
    cv::Mat dewarpImage(const cv::Mat& image, const cv::Mat& board_uv,
                        const cv::Mat& board_xy, int res_factor = 50);

    PipelineOptions& options();
    const PipelineOptions& options() const;

private:
    ChessboardDetector detector_;
    PipelineOptions options_;

    // polynomial coefficients
    cv::Mat coeff_u_;
    cv::Mat coeff_v_;

    void fitPolynomial(const cv::Mat& xy, const cv::Mat& uv);
    cv::Point2d predictPoint(double x, double y) const;
};

} // namespace cbdetect

#endif // CBDETECT_PIPELINE_H 