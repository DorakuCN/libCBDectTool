cmake_minimum_required(VERSION 3.10)
project(libcbdetect VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
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
add_executable(demo src/demo.cpp)
target_link_libraries(demo cbdetect ${OpenCV_LIBS})

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

# Create library
add_library(cbdetect STATIC
    src/corner.cpp
    src/chessboard.cpp
    src/chessboard_detector.cpp
    src/template_matching.cpp
    src/correlation_template.cpp
    src/image_processing.cpp
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