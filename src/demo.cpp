#include "cbdetect/chessboard_detector.h"
#include "cbdetect/chessboard.h"
#include "cbdetect/corner.h"
#include <chrono>
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <string>

using namespace std::chrono;

/**
 * 棋盘格检测演示函数
 * @param image_path 图像文件路径
 * @param corner_type 角点类型 (SADDLE_POINT 或 MONKEY_SADDLE_POINT)
 * @param debug_mode 是否启用调试模式
 */
void detect_chessboard(const std::string& image_path, 
                      cbdetect::CornerType corner_type = cbdetect::CornerType::SADDLE_POINT,
                      bool debug_mode = false) {
    
    // 读取图像
    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    
    if (img.empty()) {
        std::cerr << "Error: Could not load image '" << image_path << "'" << std::endl;
        return;
    }

    std::cout << "Processing image: " << image_path << " (size: " << img.cols << "x" << img.rows << ")" << std::endl;

    // 创建检测参数
    cbdetect::DetectionParams params;
    params.corner_type = corner_type;
    params.show_processing = debug_mode;
    params.show_debug_images = debug_mode;
    params.overlay_results = debug_mode;

    // 创建检测器
    cbdetect::ChessboardDetector detector(params);

    // 执行检测
    auto t1 = high_resolution_clock::now();
    
    cbdetect::Chessboards chessboards = detector.detectChessboards(img);
    
    auto t2 = high_resolution_clock::now();
    
    // 计算执行时间
    auto duration = duration_cast<microseconds>(t2 - t1);
    double total_time_ms = duration.count() / 1000.0;

    // 获取角点信息（用于可视化）
    cbdetect::Corners corners = detector.findCorners(img);

    // 输出结果
    std::cout << "Detection " << (!chessboards.empty() ? "SUCCESS" : "FAILED") << std::endl;
    std::cout << "Total execution time: " << total_time_ms << " ms" << std::endl;
    std::cout << "Detected " << corners.size() << " corners and " << chessboards.size() << " boards" << std::endl;

    if (debug_mode) {
        std::cout << "Debug mode enabled - detailed information:" << std::endl;
        for (size_t i = 0; i < corners.size(); ++i) {
            std::cout << "  Corner " << i << ": (" << corners[i].pt.x << ", " << corners[i].pt.y 
                      << ") score: " << corners[i].quality_score << std::endl;
        }
        
        for (size_t i = 0; i < chessboards.size(); ++i) {
            const auto& board = chessboards[i];
            std::cout << "  Board " << i << ": " << board->rows() << "x" << board->cols() 
                      << " corners, energy: " << board->energy << std::endl;
        }
    }

    // 可视化结果
    cv::Mat result_img = img.clone();
    
    // 绘制角点
    cbdetect::ChessboardDetector::drawCorners(result_img, corners, cv::Scalar(0, 255, 0), 3);
    
    // 绘制棋盘格
    cbdetect::ChessboardDetector::drawChessboards(result_img, chessboards, corners, cv::Scalar(255, 0, 0));

    // 保存结果图像
    std::string output_path = "result_" + std::to_string(time(nullptr)) + ".png";
    cv::imwrite(output_path, result_img);
    std::cout << "Result saved to: " << output_path << std::endl;

    // 显示结果（可选）
    if (debug_mode) {
        cv::imshow("Chessboard Detection Result", result_img);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
}

/**
 * 显示使用说明
 */
void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options] [image_path]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -h, --help          Show this help message" << std::endl;
    std::cout << "  -d, --debug         Enable debug mode" << std::endl;
    std::cout << "  -t, --type TYPE     Corner type (saddle/monkey) [default: saddle]" << std::endl;
    std::cout << "  image_path          Path to input image [default: data/04.png]" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  " << program_name << " data/04.png" << std::endl;
    std::cout << "  " << program_name << " -d data/05.png" << std::endl;
    std::cout << "  " << program_name << " -t monkey data/06.png" << std::endl;
}

int main(int argc, char* argv[]) {
    std::string image_path = "data/04.png";  // 默认图像路径
    cbdetect::CornerType corner_type = cbdetect::CornerType::SADDLE_POINT;  // 默认角点类型
    bool debug_mode = false;

    // 解析命令行参数
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "-d" || arg == "--debug") {
            debug_mode = true;
        } else if (arg == "-t" || arg == "--type") {
            if (i + 1 < argc) {
                std::string type = argv[++i];
                if (type == "monkey") {
                    corner_type = cbdetect::CornerType::MONKEY_SADDLE_POINT;
                } else if (type == "saddle") {
                    corner_type = cbdetect::CornerType::SADDLE_POINT;
                } else {
                    std::cerr << "Error: Unknown corner type '" << type << "'" << std::endl;
                    std::cerr << "Valid types: saddle, monkey" << std::endl;
                    return 1;
                }
            } else {
                std::cerr << "Error: Missing argument for --type" << std::endl;
                return 1;
            }
        } else if (arg[0] != '-') {
            // 非选项参数作为图像路径
            image_path = arg;
        } else {
            std::cerr << "Error: Unknown option '" << arg << "'" << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }

    std::cout << "=== Chessboard Detection Demo ===" << std::endl;
    std::cout << "Image: " << image_path << std::endl;
    std::cout << "Corner type: " << (corner_type == cbdetect::CornerType::SADDLE_POINT ? "SaddlePoint" : "MonkeySaddle") << std::endl;
    std::cout << "Debug mode: " << (debug_mode ? "ON" : "OFF") << std::endl;
    std::cout << "=================================" << std::endl;

    // 执行检测
    detect_chessboard(image_path, corner_type, debug_mode);

    return 0;
} 