#include <ceres/ceres.h>
#include <Eigen/Dense>
#include "ceres/autodiff_cost_function.h"
#include <utils.hpp>

class CostFunction_Ori {
public:
    CostFunction_Ori( Eigen::Quaterniond q_ab, double w = 1.0)
        : q_ab_icp_(q_ab), w_(w) {}


    template <typename T>
    bool operator()(const T* const pose_a_roll_ptr,
                    const T* const pose_a_pitch_ptr,
                    const T* const pose_a_yaw_ptr,
                    const T* const pose_b_roll_ptr,
                    const T* const pose_b_pitch_ptr,
                    const T* const pose_b_yaw_ptr,
                    T* residuals_ptr) const {

        Eigen::Quaternion<T> q_a = CreateQuaternion(pose_a_roll_ptr, pose_a_pitch_ptr, pose_a_yaw_ptr);
        Eigen::Quaternion<T> q_b = CreateQuaternion(pose_b_roll_ptr, pose_b_pitch_ptr, pose_b_yaw_ptr);
        Eigen::Quaternion<T> q_a_normalized = q_a.normalized();
        Eigen::Quaternion<T> q_b_normalized = q_b.normalized();
        Eigen::Quaternion<T> q_ab_icp_T = q_ab_icp_.template cast<T>();

        Eigen::Quaternion<T> q_ab_est_ = q_a_normalized.inverse() * q_b_normalized;
        Eigen::Quaternion<T> q_ab_est_normalized = q_ab_est_.normalized();
        Eigen::Quaternion<T> error_rot = q_ab_est_normalized * q_ab_icp_T.inverse();
        Eigen::Quaternion<T> error_rot_norm = error_rot.normalized();

        Eigen::Map<Eigen::Matrix<T, 4, 1>> residuals(residuals_ptr);
        residuals(0) = error_rot_norm.x() * T(w_) * T(1.0);
        residuals(1) = error_rot_norm.y() * T(w_) * T(1.0);
        residuals(2) = error_rot_norm.z() * T(w_) * T(1.0);
        residuals(3) = (T(1.0) - error_rot_norm.w() * error_rot_norm.w()) * T(w_) * T(1.0);

        return true;
    }

    static ceres::CostFunction* Create(Eigen::Quaterniond q_ab, double w = 1.0) {
        return new ceres::AutoDiffCostFunction<CostFunction_Ori, 4, 1, 1,1,1,1,1>(
            new CostFunction_Ori(q_ab, w));
    }

private:

template <typename T>
Eigen::Quaternion<T> CreateQuaternion(const T* const roll, const T* const pitch, const T* const yaw) const {
   
    Eigen::AngleAxis<T> roll_angle(*roll, Eigen::Matrix<T, 3, 1>::UnitX());
    Eigen::AngleAxis<T> pitch_angle(*pitch, Eigen::Matrix<T, 3, 1>::UnitY());
    Eigen::AngleAxis<T> yaw_angle(*yaw, Eigen::Matrix<T, 3, 1>::UnitZ());
    
    Eigen::Quaternion<T> q = yaw_angle * pitch_angle * roll_angle; 

    return q;
}
   
    const Eigen::Quaterniond q_ab_icp_;
    double w_;
};

