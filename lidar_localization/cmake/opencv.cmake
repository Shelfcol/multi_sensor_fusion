find_package(OpenCV REQUIRED)
include_directories(${OPENCV_INCLUDE_DIRS})
list(APPEND ALL_TARGET_LIBRARIES ${OpenCV_LIBS})
