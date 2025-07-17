#include "cbdetect/pipeline.h"
#include <opencv2/opencv.hpp>
#include <cmath>

namespace cbdetect {

Pipeline::Pipeline(const DetectionParams& det_params,
                   const PipelineOptions& options)
    : detector_(det_params), options_(options) {}

PipelineOptions& Pipeline::options() { return options_; }
const PipelineOptions& Pipeline::options() const { return options_; }

std::tuple<int, cv::Mat, cv::Mat> Pipeline::detect(const cv::Mat& image,
                                                   const cv::Size& size) {
    Chessboards boards = detector_.detectChessboards(image);
    if (boards.empty()) {
        return std::make_tuple(0, cv::Mat(), cv::Mat());
    }

    // Use the largest chessboard
    auto best = boards[0];
    int max_count = best->getCornerCount();
    for (size_t i = 1; i < boards.size(); ++i) {
        if (boards[i]->getCornerCount() > max_count) {
            best = boards[i];
            max_count = boards[i]->getCornerCount();
        }
    }

    // Retrieve corners from detector
    Corners corners = detector_.findCorners(image);

    // Map board grid to UV and XY matrices
    int rows = best->rows();
    int cols = best->cols();
    cv::Mat board_uv(max_count, 2, CV_64F);
    cv::Mat board_xy(max_count, 2, CV_64F);
    int cnt = 0;
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            int idx = best->getCornerIndex(r, c);
            if (idx >= 0 && idx < static_cast<int>(corners.size())) {
                board_uv.at<double>(cnt, 0) = corners[idx].pt.x;
                board_uv.at<double>(cnt, 1) = corners[idx].pt.y;
                board_xy.at<double>(cnt, 0) = c;
                board_xy.at<double>(cnt, 1) = r;
                cnt++;
            }
        }
    }
    board_uv = board_uv.rowRange(0, cnt).clone();
    board_xy = board_xy.rowRange(0, cnt).clone();

    if (cnt == 0) {
        return std::make_tuple(0, cv::Mat(), cv::Mat());
    }

    // Fit polynomial for refinement
    if (options_.predict) {
        fitPolynomial(board_xy, board_uv);

        int cols_all = cols;
        int rows_all = rows;
        if (options_.expand && size.width > 0 && size.height > 0) {
            cols_all = size.width;
            rows_all = size.height;
        }
        cv::Mat xy_grid(rows_all * cols_all, 2, CV_64F);
        cnt = 0;
        for (int r = 0; r < rows_all; ++r) {
            for (int c = 0; c < cols_all; ++c) {
                xy_grid.at<double>(cnt, 0) = c;
                xy_grid.at<double>(cnt, 1) = r;
                cnt++;
            }
        }
        cv::Mat uv_pred(rows_all * cols_all, 2, CV_64F);
        for (int i = 0; i < xy_grid.rows; ++i) {
            cv::Point2d p = predictPoint(xy_grid.at<double>(i,0), xy_grid.at<double>(i,1));
            uv_pred.at<double>(i,0) = p.x;
            uv_pred.at<double>(i,1) = p.y;
        }
        board_uv = uv_pred;
        board_xy = xy_grid;
    }

    int result = 1;
    if (size.width > 0 && size.height > 0 &&
        size.width == cols && size.height == rows) {
        result = 2;
    }

    return std::make_tuple(result, board_uv, board_xy);
}

void Pipeline::fitPolynomial(const cv::Mat& xy, const cv::Mat& uv) {
    // Build design matrix
    int deg = options_.polynomial_degree;
    int terms = (deg + 1) * (deg + 2) / 2;
    cv::Mat A(xy.rows, terms, CV_64F);
    for (int i = 0; i < xy.rows; ++i) {
        double x = xy.at<double>(i,0);
        double y = xy.at<double>(i,1);
        int col = 0;
        for (int d = 0; d <= deg; ++d) {
            for (int j = 0; j <= d; ++j) {
                int k = d - j;
                A.at<double>(i,col++) = std::pow(x,j) * std::pow(y,k);
            }
        }
    }
    cv::Mat AtA = A.t() * A;
    cv::Mat AtB_u = A.t() * uv.col(0);
    cv::Mat AtB_v = A.t() * uv.col(1);
    cv::solve(AtA, AtB_u, coeff_u_, cv::DECOMP_SVD);
    cv::solve(AtA, AtB_v, coeff_v_, cv::DECOMP_SVD);
}

cv::Point2d Pipeline::predictPoint(double x, double y) const {
    int deg = options_.polynomial_degree;
    int terms = (deg + 1) * (deg + 2) / 2;
    cv::Mat t(1, terms, CV_64F);
    int col = 0;
    for (int d = 0; d <= deg; ++d) {
        for (int j = 0; j <= d; ++j) {
            int k = d - j;
            t.at<double>(0,col++) = std::pow(x,j) * std::pow(y,k);
        }
    }
    cv::Mat u_mat = t * coeff_u_;
    cv::Mat v_mat = t * coeff_v_;
    double u = u_mat.at<double>(0, 0);
    double v = v_mat.at<double>(0, 0);
    return cv::Point2d(u,v);
}

cv::Mat Pipeline::dewarpImage(const cv::Mat& image, const cv::Mat& board_uv,
                              const cv::Mat& board_xy, int res_factor) {
    if (board_uv.empty() || board_xy.empty()) return cv::Mat();
    fitPolynomial(board_xy, board_uv);
    int cols = static_cast<int>(cv::norm(board_xy.col(0), cv::NORM_INF)) + 1;
    int rows = static_cast<int>(cv::norm(board_xy.col(1), cv::NORM_INF)) + 1;
    int res_u = res_factor * cols;
    int res_v = res_factor * rows;
    cv::Mat dewarped(res_v, res_u, image.type());
    for (int j = 0; j < res_v; ++j) {
        for (int i = 0; i < res_u; ++i) {
            double x = static_cast<double>(i) / res_factor;
            double y = static_cast<double>(res_v - 1 - j) / res_factor;
            cv::Point2d p = predictPoint(x, y);
            int u = static_cast<int>(std::round(p.x));
            int v = static_cast<int>(std::round(p.y));
            if (u >= 0 && u < image.cols && v >= 0 && v < image.rows) {
                dewarped.at<cv::Vec3b>(j,i) = image.at<cv::Vec3b>(v,u);
            } else {
                dewarped.at<cv::Vec3b>(j,i) = cv::Vec3b(0,0,0);
            }
        }
    }
    return dewarped;
}

} // namespace cbdetect 