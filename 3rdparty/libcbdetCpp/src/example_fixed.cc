#include "libcbdetect/boards_from_corners.h"
#include "libcbdetect/config.h"
#include "libcbdetect/find_corners.h"
#include "libcbdetect/plot_boards.h"
#include "libcbdetect/plot_corners.h"
#include <chrono>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace std::chrono;

void detect(const char* str, cbdetect::CornerType corner_type) {
  cbdetect::Corner corners;
  std::vector<cbdetect::Board> boards;
  cbdetect::Params params;
  params.corner_type = corner_type;

  cv::Mat img = cv::imread(str, cv::IMREAD_COLOR);
  
  if (img.empty()) {
    printf("Error: Could not load image '%s'\n", str);
    return;
  }

  printf("Processing image: %s (size: %dx%d)\n", str, img.cols, img.rows);

  auto t1 = high_resolution_clock::now();
  cbdetect::find_corners(img, corners, params);
  auto t2 = high_resolution_clock::now();
  cbdetect::plot_corners(img, corners);
  auto t3 = high_resolution_clock::now();
  cbdetect::boards_from_corners(img, corners, boards, params);
  auto t4 = high_resolution_clock::now();
  printf("Find corners took: %.3f ms\n", duration_cast<microseconds>(t2 - t1).count() / 1000.0);
  printf("Find boards took: %.3f ms\n", duration_cast<microseconds>(t4 - t3).count() / 1000.0);
  printf("Total took: %.3f ms\n", duration_cast<microseconds>(t2 - t1).count() / 1000.0 + duration_cast<microseconds>(t4 - t3).count() / 1000.0);
  printf("Detected %zu corners and %zu boards\n", corners.p.size(), boards.size());
  cbdetect::plot_boards(img, corners, boards, params);
}

int main(int argc, char* argv[]) {
  char* path = nullptr;
  
  if (argc >= 2) {
    path = argv[1];
  } else {
    // 默认使用example_data中的04.png
    path = (char*)"example_data/04.png";
  }
  
  printf("Testing image: %s\n", path);
  detect(path, cbdetect::SaddlePoint);
  return 0;
} 