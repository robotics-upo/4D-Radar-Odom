#pragma once

namespace rio {
  struct RadarEgoVelocityEstimatorConfig {
    float min_dist;
    float max_dist;
    float min_db;
    float elevation_thresh_deg;
    float azimuth_thresh_deg;
    float doppler_velocity_correction_factor;
    float thresh_zero_velocity;
    float allowed_outlier_percentage;
    float sigma_zero_velocity_x;
    float sigma_zero_velocity_y;
    float sigma_zero_velocity_z;
    float sigma_offset_radar_x;
    float sigma_offset_radar_y;
    float sigma_offset_radar_z;
    float max_sigma_x;
    float max_sigma_y;
    float max_sigma_z;
    float max_r_cond;
    bool use_cholesky_instead_of_bdcsvd;
    bool use_ransac;
    float outlier_prob;
    float success_prob;
    float N_ransac_points;
    float inlier_thresh;
  };
}
