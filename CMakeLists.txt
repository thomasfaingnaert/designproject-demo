cmake_minimum_required(VERSION 3.13.1)
project(Test)
find_package(OpenCV REQUIRED)
add_executable(Test main.cpp)
target_link_libraries(Test ${OpenCV_LIBS})
