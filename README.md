<a id="readme-top"></a>
<!-- PROJECT LOGO -->
<br />
<div>

# Radar Based Odometry for Ground Vehicles

</div>
<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#system-requirements">System Requirements</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#tuning">Tuning</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

**Radar_Odom** is an open-source ROS2 Humble package designed to estimate the trajectory of a ground vehicle equipped with a 4D radar (x, y, z, Doppler) and an IMU. It is a 3D odometry system aimed at contributing to the development of odometry algorithms in adverse situations where commonly used sensors such as LiDAR, cameras, etc., are not applicable, for instance, in environments with fog or rain in 3D scenarios.

<p align="center">
  <img src="figures/WF.png" alt="System Structure" />
</p>
<p align="right">(<a href="#readme-top">back to top</a>)</p>

The `Radar_Odom` package is structured around two main nodes that work together to estimate the vehicle's trajectory using radar and IMU data.

### 1. Filtering & Doppler Velocity Estimation Node
This node processes data from both the radar and the IMU. It performs several tasks:
- **PointCloud Preprocessing**: Prepares and filters the raw radar data.
- **Doppler-Based Velocity Estimation**: Estimates the vehicle's ego-velocity by leveraging the Doppler effect detected at radar points.
- **Outliers Rejection**: Uses techniques like RANSAC to detect and exclude non-physically feasible movements and dynamic objects (e.g., moving vehicles or pedestrians), ensuring only static elements contribute to the final velocity estimation.

### 2. Optimization Node
Once the radar and IMU data have been refined, this node handles:
- **Pose Estimation**: Using the ego-velocity and filtered data, the node estimates the vehicle's pose.
- **KeyFrame Management**: Manages key frames and ensures optimization over a sliding window.
- **Graph Construction & Optimization**: Builds and optimizes a graph of the vehicle's poses using scan matching (GICP) and IMU constraints.

This process ensures the vehicle's trajectory is continuously updated and optimized. The nodes work together to provide accurate pose estimation even in challenging environments where traditional sensors may not perform well.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

### System Requirements
- **Ubuntu 22.04 (Jammy)**
- **ROS2 Humble**

### ROS2 Packages:

    tf2_eigen
    pcl_conversions
    pcl_ros
    tf2_ros
    tf2_geometry_msgs

### External Libraries:

    Eigen3
    Ceres
    angles
    PCL
    OpenCV

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Installation

Follow these steps to install the `radar_odom` package from the `Radar_Odom` repository:

1. **Clone the repository** (excluding `README.md` and the `figures` folder):

    ```bash
    mkdir radar_odom
    cd pose_slam
    git clone https://github.com/LuciaCoto/Radar_Odom.git .
    ```

2. **Build the package**:

    After cloning the repository and ensuring all the dependencies mentioned in the previous section are installed, build the package:

    ```bash
    colcon build
    ```

3. **Source your workspace**:

    After building, source your workspace to ensure the package is recognized:

    ```bash
    source install/setup.bash
    ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Tuning 

## Tuning

This section outlines the key parameters for configuring the radar-based odometry system. The configuration file is divided into two main nodes: `radar_pcl_processor` and `graph_slam`. Below is a brief description of the key parameters you can adjust:

### 1. **radar_pcl_processor Node**:

This node is responsible for processing radar and IMU data, estimating Doppler-based velocity, and filtering dynamic objects. Key parameters include:

- **IMU and Radar Topics**:
  - `imu_topic`: Specify the topic name where IMU data is published.
  - `radar_topic`: Specify the radar point cloud topic to process.

- **Vehicle Dynamics**:
  - `holonomic_vehicle`: Set to `true` for holonomic vehicles or `false` for non-holonomic ones.
  
- **Filtering Parameters**:
  - `distance_near_thresh` / `distance_far_thresh`: Defines the range of radar points to consider for processing.
  - `z_low_thresh` / `z_high_thresh`: Z-axis limits to filter out noise or irrelevant points based on height.

- **Dynamic Object Removal**:
  - `enable_dynamic_object_removal`: Enable or disable the removal of dynamic objects from the radar data. Setting this to `true` helps reject moving objects, like pedestrians or other vehicles.

The Doppler-based velocity estimator uses additional internal parameters for its operation (e.g., `doppler_velocity_correction_factor`, `allowed_outlier_percentage`), but these are set automatically based on system requirements and generally do not need manual tuning.

### 2. **graph_slam Node**:

This node handles the pose graph optimization using radar and IMU data, where keyframes are managed and graph optimization is performed.

- **IMU Bias**:
  - `bias_rpy`: Adjust the IMU bias for roll, pitch, and yaw angles, helping correct any drift or offset in the IMU data.

- **Keyframe Management**:
  - `keyframe_delta_trans`: The minimum translation (in meters) required between keyframes.
  - `keyframe_delta_angle`: The minimum rotation (in radians) required between keyframes.

- **Optimization Window**:
  - `max_window_size`: Defines the size of the sliding window for graph optimization, determining how many keyframes are optimized at once.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

