#include <ceres/ceres.h>
#include <Eigen/Dense>
#include "ceres/autodiff_cost_function.h"
#include <utils.hpp>

class CostFunction_Odom {
public:
    CostFunction_Odom(const Eigen::Vector3d& t_ab, const double& w = 1.0)
        : t_ab_odom_(t_ab), w_(w) {
    }

    template <typename T>
    bool operator()(const T* const pose_a_roll_ptr, const T* const pose_a_pitch_ptr, const T* const pose_a_yaw_ptr,
                    const T* const pose_a_tx_ptr, const T* const pose_a_ty_ptr, const T* const pose_a_tz_ptr,
                    const T* const pose_b_tx_ptr, const T* const pose_b_ty_ptr, const T* const pose_b_tz_ptr,
                    T* residuals_ptr) const {
        
        Eigen::Quaternion<T> q_a = CreateQuaternion(*pose_a_roll_ptr, *pose_a_pitch_ptr, *pose_a_yaw_ptr);
        Eigen::Quaternion<T> q_a_normalized = q_a.normalized();
        Eigen::Matrix<T, 3, 1> t_a(*pose_a_tx_ptr, *pose_a_ty_ptr, *pose_a_tz_ptr);
        Eigen::Matrix<T, 3, 1> t_b(*pose_b_tx_ptr, *pose_b_ty_ptr, *pose_b_tz_ptr);
        Eigen::Matrix<T, 3, 1> t_ab_odomT = t_ab_odom_.template cast<T>();

        Eigen::Matrix<T, 3, 1> t_ab_ = q_a_normalized.inverse() * (t_b - t_a);
        Eigen::Matrix<T, 3, 1> error_pos = t_ab_ - t_ab_odomT;

        Eigen::Map<Eigen::Matrix<T, 1, 1>> residuals(residuals_ptr);
        residuals(0) = error_pos[2] * T(1.0) * T(w_);

        return true;
    }

    static ceres::CostFunction* Create(const Eigen::Vector3d& t_ab, const double& w = 1.0) {
        return new ceres::AutoDiffCostFunction<CostFunction_Odom, 1, 1, 1, 1, 1, 1, 1, 1, 1,1>(
            new CostFunction_Odom(t_ab, w));
    }

private:
    template <typename T>
    Eigen::Quaternion<T> CreateQuaternion(const T& roll, const T& pitch, const T& yaw) const {
        Eigen::AngleAxis<T> roll_angle(roll, Eigen::Matrix<T, 3, 1>::UnitX());
        Eigen::AngleAxis<T> pitch_angle(pitch, Eigen::Matrix<T, 3, 1>::UnitY());
        Eigen::AngleAxis<T> yaw_angle(yaw, Eigen::Matrix<T, 3, 1>::UnitZ());

        Eigen::Quaternion<T> q = yaw_angle * pitch_angle * roll_angle;
        
        return q;
    }

    const Eigen::Vector3d t_ab_odom_;
    const double w_;
};
