// rio_utils/radar_point_cloud.h

#pragma once

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/pcl_macros.h>

#include <sensor_msgs/msg/point_cloud2.hpp>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <angles/angles.h>

struct RadarPointCloudType
{
  PCL_ADD_POINT4D   // x, y, z position in [m]
  PCL_ADD_INTENSITY
  union
    {
      struct
      {
        float doppler;
      };
      float data_c[4];
    };
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT
(
    RadarPointCloudType,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, intensity, intensity)
    (float, doppler, doppler)
)


struct EulerAngles : public Eigen::Vector3d
{
  EulerAngles() : Eigen::Vector3d(0, 0, 0) {}

  EulerAngles(const double roll, const double pitch, const double yaw) : Eigen::Vector3d(roll, pitch, yaw) {}

  EulerAngles(const Eigen::Vector3d& eul_n_b) : Eigen::Vector3d(eul_n_b) {}

  EulerAngles from_degrees(const Eigen::Vector3d& eul_rad)
  {
    x() = angles::from_degrees(eul_rad.x());
    y() = angles::from_degrees(eul_rad.y());
    z() = angles::from_degrees(eul_rad.z());
    return EulerAngles(x(), y(), z());
  }

  Eigen::Vector3d to_degrees() { return Eigen::Vector3d(angles::to_degrees(x()), angles::to_degrees(y()), angles::to_degrees(z())); }

  double& roll() { return Eigen::Vector3d::x(); }
  double roll() const { return Eigen::Vector3d::x(); }

  double& pitch() { return Eigen::Vector3d::y(); }
  double pitch() const { return Eigen::Vector3d::y(); }

  double& yaw() { return Eigen::Vector3d::z(); }
  double yaw() const { return Eigen::Vector3d::z(); }
};