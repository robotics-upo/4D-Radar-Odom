#pragma once
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <Eigen/Dense>
#include <unsupported/Eigen/NonLinearOptimization>
#include <iostream>
#include "RadarEgoVelConfig.hpp"

namespace rio {
  using Vector11 = Eigen::Matrix<double, 11, 1>;

  struct RadarEgoVelocityEstimatorIndices {
    uint x_r = 0;
    uint y_r = 1;
    uint z_r = 2;
    uint snr_db = 3;
    uint doppler = 4;
    uint range = 5;
    uint azimuth = 6;
    uint elevation = 7;
    uint normalized_x = 8;
    uint normalized_y = 9;
    uint normalized_z = 10;
  };

  RadarPointCloudType toRadarPointCloudType(const Vector11& item, const RadarEgoVelocityEstimatorIndices& idx) {
    RadarPointCloudType point;
    point.x = item[idx.x_r];
    point.y = item[idx.y_r];
    point.z = item[idx.z_r];
    point.doppler = -item[idx.doppler];
    point.intensity = item[idx.snr_db];
    return point;
  }

  class RadarEgoVel {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    RadarEgoVel(const RadarEgoVelocityEstimatorConfig& config) : config_(config) {
      setRansacIter();
    }

    bool estimate(const sensor_msgs::msg::PointCloud2& radar_scan_msg,
                  const double& pitch,
                  const double& roll,
                  Eigen::Vector3d& v_r,
                  Eigen::Vector3d& sigma_v_r,
                  sensor_msgs::msg::PointCloud2& inlier_radar_msg,
                  sensor_msgs::msg::PointCloud2& outlier_radar_msg);

  private:
    const RadarEgoVelocityEstimatorIndices idx_;
    RadarEgoVelocityEstimatorConfig config_;
    uint ransac_iter_ = 0;

    void setRansacIter() {
      ransac_iter_ = static_cast<uint>(std::log(1.0 - config_.success_prob) /
                                       std::log(1.0 - std::pow(1.0 - config_.outlier_prob, config_.N_ransac_points)));
    }

    bool solve3DFullRansac(const Eigen::MatrixXd& radar_data, const double& pitch, const double& roll,
                           Eigen::Vector3d& v_r, Eigen::Vector3d& sigma_v_r,
                           std::vector<uint>& inlier_idx_best, std::vector<uint>& outlier_idx_best);
    bool solve3DFull(const Eigen::MatrixXd& radar_data, const double pitch, const double roll, Eigen::Vector3d& v_r);
  };

  bool RadarEgoVel::estimate(const sensor_msgs::msg::PointCloud2& radar_scan_msg, const double& pitch, const double& roll, Eigen::Vector3d& v_r, Eigen::Vector3d& sigma_v_r, sensor_msgs::msg::PointCloud2& inlier_radar_msg, sensor_msgs::msg::PointCloud2& outlier_radar_msg) {
    auto radar_scan = std::make_unique<pcl::PointCloud<RadarPointCloudType>>();
    auto radar_scan_inlier = std::make_unique<pcl::PointCloud<RadarPointCloudType>>();
    auto radar_scan_outlier = std::make_unique<pcl::PointCloud<RadarPointCloudType>>();
    
    bool success = false;
    pcl::fromROSMsg(radar_scan_msg, *radar_scan);
    std::vector<Vector11> valid_targets;

    for (uint i = 0; i < radar_scan->size(); ++i) {
      const auto target = radar_scan->at(i);
      const double r = Eigen::Vector3d(target.x, target.y, target.z).norm();

      double azimuth = std::atan2(target.y, target.x);
      double elevation = std::atan2(std::sqrt(target.x * target.x + target.y * target.y), target.z) - M_PI_2;

      if (r > config_.min_dist && r < config_.max_dist && target.intensity > config_.min_db &&
          std::fabs(azimuth) < angles::from_degrees(config_.azimuth_thresh_deg) &&
          std::fabs(elevation) < angles::from_degrees(config_.elevation_thresh_deg)) {
        
        Vector11 v_pt;
        v_pt << target.x, target.y, target.z, target.intensity, -target.doppler * config_.doppler_velocity_correction_factor,
            r, azimuth, elevation, target.x / r, target.y / r, target.z / r;
        valid_targets.emplace_back(v_pt);
      }
    }
    
    if (valid_targets.size() > 2) {
      std::vector<double> v_dopplers;
      for (const auto &v_pt : valid_targets)
        v_dopplers.emplace_back(std::fabs(v_pt[idx_.doppler]));
      
      const size_t n = v_dopplers.size() * (1.0 - config_.allowed_outlier_percentage); 
      std::nth_element(v_dopplers.begin(), v_dopplers.begin() + n, v_dopplers.end()); 
      const auto median = v_dopplers[n];

      if (median < config_.thresh_zero_velocity) {
        v_r = Eigen::Vector3d(0, 0, 0);
        sigma_v_r = Eigen::Vector3d(config_.sigma_zero_velocity_x, config_.sigma_zero_velocity_y, config_.sigma_zero_velocity_z);
        for (const auto &item : valid_targets)
          if (std::fabs(item[idx_.doppler]) < config_.thresh_zero_velocity)
            radar_scan_inlier->push_back(toRadarPointCloudType(item, idx_));
        success = true;
      } else {
        Eigen::MatrixXd radar_data(valid_targets.size(), 4);
        uint idx = 0;
        for (const auto &v_pt : valid_targets)
          radar_data.row(idx++) = Eigen::Vector4d(v_pt[idx_.normalized_x], v_pt[idx_.normalized_y], v_pt[idx_.normalized_z], v_pt[idx_.doppler]);

        std::vector<uint> inlier_idx_best;
        std::vector<uint> outlier_idx_best;
        success = solve3DFullRansac(radar_data, pitch, roll, v_r, sigma_v_r, inlier_idx_best, outlier_idx_best);

        for (const auto& idx : inlier_idx_best)
          radar_scan_inlier->push_back(toRadarPointCloudType(valid_targets.at(idx), idx_));
        for (const auto& idx : outlier_idx_best)
          radar_scan_outlier->push_back(toRadarPointCloudType(valid_targets.at(idx), idx_));
      }
    }

    radar_scan_inlier->height = 1;
    radar_scan_inlier->width = radar_scan_inlier->size();

    pcl::PCLPointCloud2 tmp;
    pcl::toPCLPointCloud2<RadarPointCloudType>(*radar_scan_inlier, tmp);
    pcl_conversions::fromPCL(tmp, inlier_radar_msg);
    inlier_radar_msg.header = radar_scan_msg.header;

    radar_scan_outlier->height = 1;
    radar_scan_outlier->width = radar_scan_outlier->size();

    pcl::PCLPointCloud2 tmp_o;
    pcl::toPCLPointCloud2<RadarPointCloudType>(*radar_scan_outlier, tmp_o);
    pcl_conversions::fromPCL(tmp_o, outlier_radar_msg);
    outlier_radar_msg.header = radar_scan_msg.header;

    return success;
  }

bool RadarEgoVel::solve3DFullRansac(const Eigen::MatrixXd& radar_data, const double& pitch, const double& roll, Eigen::Vector3d& v_r, Eigen::Vector3d& sigma_v_r, std::vector<uint>& inlier_idx_best, std::vector<uint>& outlier_idx_best) {
    // Matriz con los datos del radar
    Eigen::MatrixXd H_all(radar_data.rows(), 3);
    H_all.col(0) = radar_data.col(0);
    H_all.col(1) = radar_data.col(1);
    H_all.col(2) = radar_data.col(2);
    const Eigen::VectorXd y_all = radar_data.col(3);

    std::vector<uint> idx(radar_data.rows());
    for (uint k = 0; k < radar_data.rows(); ++k) idx[k] = k;
    
    std::random_device rd;
    std::mt19937 g(rd());

    // Solo continuar si hay suficientes puntos para realizar RANSAC
    if (radar_data.rows() >= config_.N_ransac_points) {
        for (uint k = 0; k < ransac_iter_; ++k) {
            std::shuffle(idx.begin(), idx.end(), g);
            Eigen::MatrixXd radar_data_iter;
            radar_data_iter.resize(config_.N_ransac_points, 4);

            for (uint i = 0; i < config_.N_ransac_points; ++i)
                radar_data_iter.row(i) = radar_data.row(idx.at(i));

            // Llamada a solve3DFull con las nuevas ecuaciones
            bool rtn = solve3DFull(radar_data_iter, pitch, roll, v_r);
            
            if (rtn) {
                Eigen::VectorXd err(radar_data.rows());
                for (int i = 0; i < radar_data.rows(); ++i) {

                    // Cálculo de expected_vr usando las nuevas ecuaciones:
                    double expected_vr = H_all(i, 0) * std::cos(pitch) + 
                                         H_all(i, 1) * std::cos(roll) +  
                                         H_all(i, 2) * (std::sin(pitch) * std::sin(roll));
                    err(i) = std::abs(y_all(i) - expected_vr);
                }

                // Identificación de inliers y outliers
                std::vector<uint> inlier_idx;
                std::vector<uint> outlier_idx;
                for (uint j = 0; j < err.rows(); ++j) {
                    if (err(j) < config_.inlier_thresh)
                        inlier_idx.emplace_back(j);
                    else
                        outlier_idx.emplace_back(j);
                }

                if (float(outlier_idx.size()) / (inlier_idx.size() + outlier_idx.size()) > 0.05) {
                    inlier_idx.insert(inlier_idx.end(), outlier_idx.begin(), outlier_idx.end());
                    outlier_idx.clear();
                }

                // Actualizar los mejores índices de inliers y outliers
                if (inlier_idx.size() > inlier_idx_best.size()) {
                    inlier_idx_best = inlier_idx;
                }
                if (outlier_idx.size() > outlier_idx_best.size()) {
                    outlier_idx_best = outlier_idx;
                }
            }
        }
    }

    // Resolver usando los inliers finales
    if (!inlier_idx_best.empty()) {
        Eigen::MatrixXd radar_data_inlier(inlier_idx_best.size(), 4);
        for (uint i = 0; i < inlier_idx_best.size(); ++i)
            radar_data_inlier.row(i) = radar_data.row(inlier_idx_best.at(i));

        // Llamada final a solve3DFull con los inliers
        bool rtn = solve3DFull(radar_data_inlier, pitch, roll, v_r);
        return rtn;
    }

    return false;
}


bool RadarEgoVel::solve3DFull(const Eigen::MatrixXd& radar_data, const double pitch, const double roll, Eigen::Vector3d& v_r) {
  
    Eigen::MatrixXd H_all(radar_data.rows(), 3);
    H_all.col(0) = radar_data.col(0);  
    H_all.col(1) = radar_data.col(1);  
    H_all.col(2) = radar_data.col(2);  
    const Eigen::VectorXd y_all = radar_data.col(3);

    Eigen::MatrixXd A(radar_data.rows(), 2);  
    Eigen::VectorXd b(radar_data.rows());

    for (int i = 0; i < H_all.rows(); ++i) {
     
      double term_x = H_all(i, 0) * std::cos(pitch);           
      double term_y = H_all(i, 1) * std::cos(roll);            
      double term_z_x = H_all(i, 2) * std::sin(pitch);        // r_z * sin(pitch) para Vx
      double term_z_y = H_all(i, 2) * std::sin(roll);         // r_z * sin(roll) para Vy

      A(i, 0) = term_x + term_z_x; 
      A(i, 1) = term_y + term_z_y; 
      b(i) = y_all(i);
    }

   
    Eigen::Vector2d v_xy = (A.transpose() * A).ldlt().solve(A.transpose() * b);

    v_r.x() = v_xy(0);  
    v_r.y() = v_xy(1); 
    v_r.z() = std::sin(pitch) * v_r.x() + std::sin(roll) * v_r.y();  

    return true;
}


}
