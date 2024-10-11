# Radar based odometry for Ground Vehicles

**Radar_Odom** is an open-source ROS2 Humble package designed to estimate the trajectory of a ground vehicle equipped with a 4D radar (x, y, z, Doppler) and an IMU. It is a 3D odometry system aimed at contributing to the development of odometry algorithms in adverse situations where commonly used sensors such as LiDAR, cameras, etc., are not applicable, for instance, in environments with fog or rain in 3D scenarios.

## 1. Dependencies

### System Requirements:
- **Ubuntu 22.04 (Jammy)**
- **ROS2 Humble**

This package depends on several libraries and ROS2 packages. Below is a list of the required packages and libraries:

### ROS2 Packages:
- `ament_cmake`
- `rclcpp`
- `geometry_msgs`
- `nav_msgs`
- `sensor_msgs`
- `tf2_eigen`
- `pcl_conversions`
- `pcl_ros`
- `tf2_ros`
- `tf2_geometry_msgs`

### External Libraries:
- **Eigen3**
- **OpenCV**
- **Ceres**
- **angles**
- **PCL**
