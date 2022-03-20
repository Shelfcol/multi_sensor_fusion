cmake_minimum_required(VERSION 2.8.3)
project(lidar_localization)

SET(CMAKE_BUILD_TYPE "Debug")
# SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
add_compile_options(-std=c++14)
add_definitions(-std=c++14)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  pcl_ros
  geometry_msgs
  tf
  eigen_conversions
  message_generation 
  std_srvs
  pcl_ros
  ndt_omp
  ndt_cpu
  
)

add_service_files(
   FILES
   saveMap.srv
)

generate_messages(
   DEPENDENCIES
   std_msgs
)

set(ALL_TARGET_LIBRARIES "")

include(cmake/glog.cmake)
include(cmake/YAML.cmake)
include(cmake/ceres.cmake)
include(cmake/g2o.cmake)
include(cmake/eigen.cmake)
include(cmake/geographic.cmake)
include(cmake/opencv.cmake)


include_directories(include ${catkin_INCLUDE_DIRS})
include(cmake/global_definition.cmake)
catkin_package()

file(GLOB_RECURSE ALL_SRCS "*.cpp")
file(GLOB_RECURSE NODE_SRCS "src/*_node.cpp")
file(GLOB_RECURSE THIRD_PARTY_SRCS "third_party/*.cpp")
list(REMOVE_ITEM ALL_SRCS ${NODE_SRCS})
list(REMOVE_ITEM ALL_SRCS ${THIRD_PARTY_SRCS})
add_definitions(-w)

add_executable(test_frame_node src/test_frame_node.cpp ${ALL_SRCS})
target_link_libraries(test_frame_node ${catkin_LIBRARIES} ${ALL_TARGET_LIBRARIES})

add_executable(save_pose_node src/save_pose_node.cpp ${ALL_SRCS})
target_link_libraries(save_pose_node ${catkin_LIBRARIES} ${ALL_TARGET_LIBRARIES})


add_executable(ceres_learning_node src/ceres_learning_node.cpp ${ALL_SRCS})
add_dependencies(ceres_learning_node ${catkin_EXPORTED_TARGETS})
target_link_libraries(ceres_learning_node ${catkin_LIBRARIES} ${ALL_TARGET_LIBRARIES})

add_executable(g2o_learning_node src/g2o_learning_node.cpp ${ALL_SRCS})
add_dependencies(g2o_learning_node ${catkin_EXPORTED_TARGETS})
target_link_libraries(g2o_learning_node ${catkin_LIBRARIES} ${ALL_TARGET_LIBRARIES})


add_executable(front_end_node src/front_end_node.cpp ${ALL_SRCS})
add_dependencies(front_end_node ${catkin_EXPORTED_TARGETS} saveMap_gencpp)
target_link_libraries(front_end_node ${catkin_LIBRARIES} ${ALL_TARGET_LIBRARIES})

add_executable(lidar_gnss_cal_node src/lidar_gnss_cal_node.cpp ${ALL_SRCS})
add_dependencies(lidar_gnss_cal_node ${catkin_EXPORTED_TARGETS})
target_link_libraries(lidar_gnss_cal_node ${catkin_LIBRARIES} ${ALL_TARGET_LIBRARIES})