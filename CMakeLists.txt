cmake_minimum_required(VERSION 3.8)
project(radar_odom)

# Establece políticas para suprimir advertencias de CMake
if(POLICY CMP0144)
  cmake_policy(SET CMP0144 NEW)
endif()

if(POLICY CMP0167)
  cmake_policy(SET CMP0167 NEW)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(angles REQUIRED)
find_package(Ceres REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)

include_directories(
  include
  ${CERES_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

add_executable(radar_pcl_processor src/radar_pcl_processor.cpp)
add_executable(optimizer src/optimizer.cpp)
add_executable(record src/record.cpp)
add_executable(baselink_tf src/baselink_tf.cpp)

ament_target_dependencies(radar_pcl_processor rclcpp sensor_msgs geometry_msgs pcl_conversions pcl_ros Eigen3 OpenCV angles)
ament_target_dependencies(optimizer rclcpp nav_msgs sensor_msgs geometry_msgs Eigen3 tf2_ros tf2_geometry_msgs Ceres pcl_ros pcl_conversions)  
ament_target_dependencies(record rclcpp nav_msgs geometry_msgs)
ament_target_dependencies(baselink_tf rclcpp nav_msgs tf2_ros)

target_link_libraries(optimizer ${CERES_LIBRARIES} ${PCL_LIBRARIES})

install(TARGETS
  radar_pcl_processor
  optimizer
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/launch
)

install(DIRECTORY
  config/
  DESTINATION share/${PROJECT_NAME}/config
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
