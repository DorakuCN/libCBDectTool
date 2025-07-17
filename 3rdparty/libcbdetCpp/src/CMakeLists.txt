include_directories(${PROJECT_SOURCE_DIR}/include)
add_executable(example example.cc)
target_link_libraries(example cbdetect ${OpenCV_LIBS})

add_executable(example_fixed example_fixed.cc)
target_link_libraries(example_fixed cbdetect ${OpenCV_LIBS})
