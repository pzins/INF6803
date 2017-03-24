cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_BUILD_TYPE Debug)
project(EigenFaces)
find_package( OpenCV REQUIRED )
add_executable(${PROJECT_NAME} "main.cpp")
target_link_libraries( EigenFaces ${OpenCV_LIBS} )


#aux_source_directory(. SRC_LIST)
