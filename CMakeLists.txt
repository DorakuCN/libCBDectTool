cmake_minimum_required(VERSION 3.10)
project(libcbdetect VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
# For macOS with Homebrew
if(APPLE)
    set(CMAKE_PREFIX_PATH "/opt/homebrew;${CMAKE_PREFIX_PATH}")
    list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew/lib/cmake")
endif()

find_package(OpenCV REQUIRED)
# find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})
# include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(include)

# Add compile options
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra")

# Add subdirectories
add_subdirectory(src)

# Create demo executables
# 保证demo.cpp被编译为可执行文件demo
add_executable(demo src/demo.cpp)
target_link_libraries(demo PRIVATE opencv_core opencv_imgcodecs opencv_imgproc opencv_highgui)
target_link_libraries(demo PRIVATE cbdetect)

add_executable(pipeline_demo src/pipeline_demo.cpp)
target_link_libraries(pipeline_demo cbdetect ${OpenCV_LIBS})

# libcdetSample compatible debug test
add_executable(debug_libcbdet src/debug_libcbdet_test.cpp)
target_link_libraries(debug_libcbdet cbdetect ${OpenCV_LIBS})

# Parameter optimization test
add_executable(optimized_debug src/optimized_debug_test.cpp)
target_link_libraries(optimized_debug cbdetect ${OpenCV_LIBS})

# Final optimized implementation test
add_executable(final_optimized src/final_optimized_test.cpp)
target_link_libraries(final_optimized cbdetect ${OpenCV_LIBS})

# Simple validation and comparison test
add_executable(validate src/simple_validation.cpp)
target_link_libraries(validate cbdetect ${OpenCV_LIBS})

# Coordinate analysis tool
add_executable(coord_analysis src/coordinate_analysis.cpp)
target_link_libraries(coord_analysis cbdetect ${OpenCV_LIBS})

# Region-focused detection analysis
add_executable(region_analysis src/region_focused_detection.cpp)
target_link_libraries(region_analysis cbdetect ${OpenCV_LIBS})

# MATLAB-targeted detection solution
add_executable(matlab_targeted src/matlab_targeted_detection.cpp)
target_link_libraries(matlab_targeted cbdetect ${OpenCV_LIBS})

# Fine-tuned detection solution
add_executable(fine_tuned src/fine_tuned_detection.cpp)
target_link_libraries(fine_tuned cbdetect ${OpenCV_LIBS})

# Final perfect detection solution
add_executable(perfect_detection src/final_perfect_detection.cpp)
target_link_libraries(perfect_detection cbdetect ${OpenCV_LIBS})

# Detailed debug comparison with MATLAB
add_executable(debug_comparison src/detailed_debug_comparison.cpp)
target_link_libraries(debug_comparison cbdetect ${OpenCV_LIBS})

# MATLAB vs C++ comprehensive comparison
add_executable(matlab_cpp_comparison src/matlab_cpp_comparison.cpp)
target_link_libraries(matlab_cpp_comparison cbdetect ${OpenCV_LIBS})

# Direction vector debug test
add_executable(direction_debug src/debug_direction_vectors.cpp)
target_link_libraries(direction_debug cbdetect ${OpenCV_LIBS})

# Histogram debug test
add_executable(histogram_debug src/debug_histogram.cpp)
target_link_libraries(histogram_debug cbdetect ${OpenCV_LIBS})

# Angle delta debug test
add_executable(angle_delta_debug src/debug_angle_delta.cpp)
target_link_libraries(angle_delta_debug cbdetect ${OpenCV_LIBS})

# detG debug test
add_executable(detG_debug src/debug_detG.cpp)
target_link_libraries(detG_debug cbdetect ${OpenCV_LIBS})

# Matrix A debug test
add_executable(matrix_A_debug src/debug_matrix_A.cpp)
target_link_libraries(matrix_A_debug cbdetect ${OpenCV_LIBS})

# Chessboard detection debug test
add_executable(chessboard_debug src/debug_chessboard_detection.cpp)
target_link_libraries(chessboard_debug cbdetect ${OpenCV_LIBS})

# Image preprocessing demo
add_executable(image_preprocessing_demo src/image_preprocessing_demo.cpp)
target_link_libraries(image_preprocessing_demo cbdetect ${OpenCV_LIBS})

# Comprehensive image preprocessing test
add_executable(image_preprocessing_test src/image_preprocessing_comprehensive_test.cpp)
target_link_libraries(image_preprocessing_test cbdetect ${OpenCV_LIBS})

# Gamma enhancement demo
add_executable(gamma_enhancement_demo src/gamma_enhancement_demo.cpp src/gamma_enhancement.cpp)
target_link_libraries(gamma_enhancement_demo cbdetect ${OpenCV_LIBS})

# Lighting analysis and detection optimization
add_executable(lighting_analysis_detection src/lighting_analysis_detection.cpp)
target_link_libraries(lighting_analysis_detection cbdetect ${OpenCV_LIBS})

# Parameter optimization test
add_executable(parameter_optimization_test src/parameter_optimization_test.cpp)
target_link_libraries(parameter_optimization_test cbdetect ${OpenCV_LIBS})

# Auto corner debug
add_executable(auto_corner_debug src/auto_corner_debug.cpp)
target_link_libraries(auto_corner_debug cbdetect ${OpenCV_LIBS})

# OpenCV comparison optimization
add_executable(opencv_comparison_optimization src/opencv_comparison_optimization.cpp)
target_link_libraries(opencv_comparison_optimization cbdetect ${OpenCV_LIBS})

# Corner detection optimizer
add_executable(corner_detection_optimizer src/corner_detection_optimizer.cpp)
target_link_libraries(corner_detection_optimizer cbdetect ${OpenCV_LIBS})

# Automated testing debug
add_executable(automated_testing_debug src/automated_testing_debug.cpp)
target_link_libraries(automated_testing_debug cbdetect ${OpenCV_LIBS})

# Add filesystem library for C++17
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_link_libraries(image_preprocessing_test stdc++fs)
    target_link_libraries(gamma_enhancement_demo stdc++fs)
    target_link_libraries(lighting_analysis_detection stdc++fs)
    target_link_libraries(parameter_optimization_test stdc++fs)
    target_link_libraries(auto_corner_debug stdc++fs)
    target_link_libraries(opencv_comparison_optimization stdc++fs)
    target_link_libraries(corner_detection_optimizer stdc++fs)
    target_link_libraries(automated_testing_debug stdc++fs)
endif()

# Create library
add_library(cbdetect STATIC
    src/corner.cpp
    src/chessboard.cpp
    src/chessboard_detector.cpp
    src/template_matching.cpp
    src/correlation_template.cpp
    src/image_processing.cpp
    src/image_preprocessing.cpp
    src/non_maximum_suppression.cpp
    src/subpixel_refinement.cpp
    src/corner_scoring.cpp
    src/zero_crossing_filter.cpp
    src/correlation_scoring.cpp
    src/chessboard_energy.cpp
    src/structure_recovery.cpp
    src/libcbdetect_adapter.cpp
    src/pipeline.cpp
)

target_link_libraries(cbdetect ${OpenCV_LIBS})

# Install targets
install(TARGETS cbdetect demo pipeline_demo
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/ DESTINATION include) 