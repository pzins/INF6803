#####################
## cmake config below
#set(CMAKE_BUILD_TYPE Debug)

cmake_minimum_required(VERSION 2.8.0 FATAL_ERROR)
set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE INTERNAL "")
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
foreach(OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES})
    string(TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG} "${CMAKE_BINARY_DIR}/bin")
endforeach(OUTPUTCONFIG CMAKE_CONFIGURATION_TYPES)
if(WIN32)
    list(APPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/external/opencv")
endif()
set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE INTERNAL "")
#set(DATA_ROOT_PATH "${CMAKE_SOURCE_DIR}/data" CACHE INTERNAL "")


########################
## solution config below
project(inf6803_h2017)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set_property(DIRECTORY APPEND PROPERTY COMPILE_DEFINITIONS $<$<CONFIG:Debug>:DEBUG>)
set_property(DIRECTORY APPEND PROPERTY COMPILE_DEFINITIONS $<$<CONFIG:Debug>:_DEBUG>)
find_package(OpenCV 3.0 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
add_definitions(${OpenCV_DEFINITIONS})
mark_as_advanced(OPENCV_FOUND OpenCV_FOUND OpenCV_DIR)
if((CMAKE_CXX_COMPILER_ID STREQUAL "GNU") OR (CMAKE_CXX_COMPILER_ID STREQUAL "Clang"))
    add_definitions(-march=native)
    add_definitions(-Wall)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif("x${CMAKE_CXX_COMPILER_ID}" STREQUAL "xMSVC")
    if((MSVC_VERSION GREATER 1899) AND (CMAKE_SIZEOF_VOID_P EQUAL 8) AND ("${OpenCV_DIR}" STREQUAL "${CMAKE_SOURCE_DIR}/external/opencv"))
        file(GLOB OpenCV_DEBUG_DLLS "${CMAKE_SOURCE_DIR}/external/opencv/x64/vc14/bin/*310d.dll")
        list(APPEND OpenCV_DEBUG_DLLS "${CMAKE_SOURCE_DIR}/external/opencv/x64/vc14/bin/opencv_ffmpeg310_64.dll")
        file(COPY ${OpenCV_DEBUG_DLLS} DESTINATION "${CMAKE_BINARY_DIR}/bin/")
        file(GLOB OpenCV_RELEASE_DLLS "${CMAKE_SOURCE_DIR}/external/opencv/x64/vc14/bin/*310.dll")
        list(APPEND OpenCV_RELEASE_DLLS "${CMAKE_SOURCE_DIR}/external/opencv/x64/vc14/bin/opencv_ffmpeg310_64.dll")
        file(COPY ${OpenCV_RELEASE_DLLS} DESTINATION "${CMAKE_BINARY_DIR}/bin/")
    endif()
    add_definitions(-DUNICODE -D_UNICODE)
    add_definitions(-D_SCL_SECURE_NO_WARNINGS)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    add_definitions(/W1)
    add_definitions(/openmp)
    add_definitions(/arch:AVX)
    add_definitions(/W4)
endif()
include_directories(
    "${CMAKE_SOURCE_DIR}/include"
    "${CMAKE_BINARY_DIR}/include"
)

#############################
## TP2 projects configs below
include_directories("include")
configure_file(
    "${CMAKE_SOURCE_DIR}/include/tp2/common.hpp.conf"
    "${CMAKE_BINARY_DIR}/include/tp2/common.hpp"
)
    add_executable(tp2_main
        "src/tp2/main.cpp"
        "src/tp2/MyTracker.cpp"
        "include/tp2/common.hpp"
    )
    target_link_libraries(tp2_main ${OpenCV_LIBRARIES})
