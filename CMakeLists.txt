cmake_minimum_required(VERSION 3.5)
project(gst_dsexample_sources2)
#set(CMAKE_CXX_FLAGS “-Wall -g”)
set(CMAKE_CXX_STANDARD 11)

find_package(PkgConfig)

find_package(OpenCV REQUIRED)
#pkg_check_modules( OpenCV REQUIRED opencv)
#pkg_check_modules(GST REQUIRED gstreamer-1.0>=1.4)
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0)
pkg_check_modules(GSTREAMER-APP REQUIRED gstreamer-app-1.0)
pkg_check_modules(GSTREAMER-AUDIO REQUIRED gstreamer-audio-1.0)
pkg_check_modules(GSTREAMER-PBUTILS REQUIRED gstreamer-pbutils-1.0)
pkg_check_modules(GSTREAMER-FFT REQUIRED gstreamer-fft-1.0)
pkg_check_modules(GSTREAMER-VIDEO REQUIRED gstreamer-video-1.0)
pkg_check_modules(GSTREAMER-BASE REQUIRED gstreamer-base-1.0)

include_directories(
        .
        …/nvgstiva-app_sources/nvgstiva-app/includes
        dsexample_lib
        {GSTREAMER_INCLUDE_DIRS} {GSTREAMER-APP_INCLUDE_DIRS}
        {GSTREAMER-AUDIO_INCLUDE_DIRS} {GSTREAMER-PBUTILS_INCLUDE_DIRS}
        {GSTREAMER-FFT_INCLUDE_DIRS} {GSTREAMER-VIDEO_INCLUDE_DIRS}
        {GSTREAMER-BASE_INCLUDE_DIRS} {CMAKE_CURRENT_SOURCE_DIR}
)

link_directories(
        {CMAKE_CURRENT_SOURCE_DIR} /usr/lib/aarch64-linux-gnu/tegra ../nvgstiva-app_sources/nvgstiva-app/src dsexample_lib {GSTREAMER_LIBRARY_DIRS}
        {GSTREAMER-APP_LIBRARY_DIRS} {GSTREAMER-AUDIO_LIBRARY_DIRS}
        {GSTREAMER-PBUTILS_LIBRARY_DIRS} {GSTREAMER-FFT_LIBRARY_DIRS}
        {GSTREAMER-VIDEO_LIBRARY_DIRS} {GSTREAMER-BASE_LIBRARY_DIRS}
)

link_libraries(
        dsexample
        nvbuf_utils
        gstnvivameta
        EGL
        {GSTREAMER_LIBRARIES} {GSTREAMER-APP_LIBRARIES}
        {GSTREAMER-AUDIO_LIBRARIES} {GSTREAMER-PBUTILS_LIBRARIES}
        {GSTREAMER-FFT_LIBRARIES} {GSTREAMER-VIDEO_LIBRARIES}
        {GSTREAMER-BASE_LIBRARIES} {OpenCV_LIBS}

)
#set(CMAKE_C_FLAGS “${CMAKE_C_FLAGS} -fPIC”)

set (CMAKE_SHARED_LINKER_FLAGS “${CMAKE_SHARED_LINKER_FLAGS} -L/usr/lib/aarch64-linux-gnu/tegra -lnvbuf_utils -lgstnvivameta -Wl,-no-undefined -L dsexample_lib -ldsexample -lEGL”)

add_library(gst_dsexample_sources2
        dsexample_lib/dsexample_lib.c
        dsexample_lib/dsexample_lib.h
        gstdsexample.cpp
        gstdsexample.h)