cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_BUILD_TYPE Debug)
project(EigenFaces)
find_package( OpenCV REQUIRED )



set(TRAINING_DATA_PATH "${CMAKE_SOURCE_DIR}/DATA/training/" CACHE INTERNAL "")
set(TEST_DATA_PATH "${CMAKE_SOURCE_DIR}/DATA/test/" CACHE INTERNAL "")

include_directories(
    "${CMAKE_SOURCE_DIR}/include"
    "${CMAKE_BINARY_DIR}/include"
)

include_directories("include")
configure_file(
    "${CMAKE_SOURCE_DIR}/include/common.hpp.in"
    "${CMAKE_BINARY_DIR}/include/common.hpp"
)

add_executable(${PROJECT_NAME}
    "src/main.cpp"
    "include/common.hpp"
)
target_link_libraries( EigenFaces ${OpenCV_LIBS} )
