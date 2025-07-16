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

# Create demo executable
add_executable(demo src/demo.cpp)
target_link_libraries(demo cbdetect ${OpenCV_LIBS})

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
)

target_link_libraries(cbdetect ${OpenCV_LIBS})

# Install targets
install(TARGETS cbdetect demo
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/ DESTINATION include) 