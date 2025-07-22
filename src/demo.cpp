#include <iostream>
#include <opencv2/opencv.hpp>
#include "cbdetect/chessboard_detector.h"
#include "cbdetect/image_preprocessing.h"

void run_detection(const std::string& image_path, const std::string& output_image) {
    std::cout << "\n加载图像: " << image_path << std::endl;
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "无法加载图像: " << image_path << std::endl;
        return;
    }
    std::cout << "图像尺寸: " << img.cols << "x" << img.rows << std::endl;

    // 配置预处理参数
    cbdetect::PreprocessingParams prep_params;
    prep_params.enable_adaptive_lighting = true;
    prep_params.enable_shadow_highlight_recovery = true;
    prep_params.auto_gamma = true;
    prep_params.gamma_correction = 1.0;
    prep_params.clahe_clip_limit = 3.0;
    prep_params.clahe_tile_size = 8;
    prep_params.enable_local_contrast = true;
    prep_params.enable_denoising = true;
    prep_params.enable_sharpening = true;

    // 配置检测参数
    cbdetect::DetectionParams det_params;
    det_params.corner_type = cbdetect::CornerType::SADDLE_POINT;
    det_params.detect_method = cbdetect::DetectMethod::TEMPLATE_MATCH_FAST;
    det_params.enable_image_preprocessing = true;
    det_params.preprocessing_params = prep_params;
    det_params.show_processing = true;
    det_params.show_debug_images = true;

    // 检测
    cbdetect::ChessboardDetector detector(det_params);
    auto chessboards = detector.detectChessboards(img);

    // 输出结果
    std::cout << "检测到棋盘格数量: " << chessboards.size() << std::endl;
    if (chessboards.size() > 0) {
        std::cout << "第一个棋盘格角点数: " << chessboards[0]->getCornerCount() << std::endl;
    }

    // 可视化结果
    cv::Mat vis = img.clone();
    auto corners = detector.findCorners(img);
    cbdetect::ChessboardDetector::drawCorners(vis, corners);
    cbdetect::ChessboardDetector::drawChessboards(vis, chessboards, corners);
    cv::imwrite(output_image, vis);
    std::cout << "结果已保存: " << output_image << std::endl;
}

int main() {
    run_detection("data/imiSample/IR.bmp", "IR_detect_result.png");
    run_detection("data/imiSample/Color.bmp", "Color_detect_result.png");
    return 0;
} 