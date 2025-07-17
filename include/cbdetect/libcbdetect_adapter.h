#ifndef CBDETECT_LIBCBDETECT_ADAPTER_H
#define CBDETECT_LIBCBDETECT_ADAPTER_H

#include <opencv2/opencv.hpp>
#include <vector>

namespace cbdetect {

// libcdetSample compatible enums
enum DetectMethod {
    TemplateMatchFast = 0,
    TemplateMatchSlow,
    HessianResponse,
    LocalizedRadonTransform
};

enum CornerType {
    SaddlePoint = 0,
    MonkeySaddlePoint
};

// libcdetSample compatible parameter structure
struct Params {
    bool show_processing;
    bool show_debug_image;
    bool show_grow_processing;
    bool norm;
    bool polynomial_fit;
    int norm_half_kernel_size;
    int polynomial_fit_half_kernel_size;
    double init_loc_thr;
    double score_thr;
    bool strict_grow;
    bool overlay;
    bool occlusion;
    DetectMethod detect_method;
    CornerType corner_type;
    std::vector<int> radius;

    Params()
        : show_processing(true)
        , show_debug_image(false)
        , show_grow_processing(false)
        , norm(false)
        , polynomial_fit(true)
        , norm_half_kernel_size(31)
        , polynomial_fit_half_kernel_size(4)
        , init_loc_thr(0.01)
        , score_thr(0.01)
        , strict_grow(true)
        , overlay(false)
        , occlusion(true)
        , detect_method(HessianResponse)
        , corner_type(SaddlePoint)
        , radius({5, 7}) {}
};

// libcdetSample compatible corner structure
struct Corner {
    std::vector<cv::Point2d> p;
    std::vector<int> r;
    std::vector<cv::Point2d> v1;
    std::vector<cv::Point2d> v2;
    std::vector<cv::Point2d> v3;
    std::vector<double> score;
};

struct Board {
    std::vector<std::vector<int>> idx;
    std::vector<std::vector<std::vector<double>>> energy;
    int num;

    Board() : num(0) {}
};

// Main libcdetSample compatible functions
void find_corners(const cv::Mat& img, Corner& corners, const Params& params = Params());
void boards_from_corners(const cv::Mat& img, const Corner& corners, std::vector<Board>& boards, const Params& params = Params());

// Core algorithm functions following libcdetSample exactly
void image_normalization_and_gradients(cv::Mat& img, cv::Mat& img_du, cv::Mat& img_dv,
                                      cv::Mat& img_angle, cv::Mat& img_weight, const Params& params);
void get_init_location(const cv::Mat& img, const cv::Mat& img_du, const cv::Mat& img_dv,
                      Corner& corners, const Params& params);
void filter_corners(const cv::Mat& img, const cv::Mat& img_angle, const cv::Mat& img_weight, 
                   Corner& corners, const Params& params);
void refine_corners(const cv::Mat& img_du, const cv::Mat& img_dv, const cv::Mat& img_angle, 
                   const cv::Mat& img_weight, Corner& corners, const Params& params);
void polynomial_fit(const cv::Mat& img, Corner& corners, const Params& params);
void score_corners(const cv::Mat& img, const cv::Mat& img_weight, Corner& corners, const Params& params);
void non_maximum_suppression_sparse(Corner& corners, int radius, const cv::Size& img_size, const Params& params);

// Helper functions
void create_correlation_patch(std::vector<cv::Mat>& templates, double angle1, double angle2, int radius);
void non_maximum_suppression(const cv::Mat& corner_map, int radius, double threshold, int template_radius, Corner& corners);
void hessian_response(const cv::Mat& img_in, cv::Mat& img_out);
void box_filter(const cv::Mat& img, cv::Mat& blur_img, int kernel_size_x, int kernel_size_y = -1);

} // namespace cbdetect

#endif // CBDETECT_LIBCBDETECT_ADAPTER_H 