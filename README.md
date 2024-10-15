<a id="readme-top"></a>
<!-- PROJECT LOGO -->
<br />
<div>

<h1>Radar Based Odometry for Ground Vehicles</h1>

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
This is an example of how you may give instructions on setting up your project locally.

### System Requirements
- **Ubuntu 22.04 (Jammy)**
- **ROS2 Humble**

ROS2 Packages:

    tf2_eigen
    pcl_conversions
    pcl_ros
    tf2_ros
    tf2_geometry_msgs

External Libraries:

    Eigen3
    Ceres
    angles
    PCL
    OpenCV (optional, used for matrix operations in the transformation)

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

The following section provides guidance on tuning the parameters for the radar-based odometry system. 

*(Continue here with your specific tuning instructions)*

<p align="right">(<a href="#readme-top">back to top</a>)</p>
