#include <ceres/ceres.h>
#include <Eigen/Dense>
#include "ceres/autodiff_cost_function.h"
#include <utils.hpp>

class CostFunction_Imu {
public:
    CostFunction_Imu(double roll, double pitch, double w = 1.0)
        : roll_(roll), pitch_(pitch), w_(w) {}

    template <typename T>
    bool operator()(const T* const pose_a_roll_ptr,
                    const T* const pose_a_pitch_ptr,
                    T* residuals_ptr) const {

        Eigen::Quaternion<T> q_a = CreateQuaternion(*pose_a_roll_ptr, *pose_a_pitch_ptr, T(0.0));
        Eigen::Quaternion<T> q_a_normalized = q_a.normalized();
        Eigen::Quaternion<T> q_imu = CreateQuaternion(T(roll_), T(pitch_), T(0.0));
        Eigen::Quaternion<T> q_imu_normalized = q_imu.normalized();

        Eigen::Quaternion<T> q_error = q_a_normalized * q_imu_normalized.inverse();
        Eigen::Quaternion<T> q_error_norm = q_error.normalized();

        Eigen::Map<Eigen::Matrix<T, 4, 1>> residuals(residuals_ptr);
        residuals(0) = q_error_norm.x() * T(w_);
        residuals(1) = q_error_norm.y() * T(w_);
        residuals(2) = q_error_norm.z() * T(w_);
        residuals(3) = (T(1.0) - q_error_norm.w() * q_error_norm.w()) * T(w_);

        return true;
    }

    static ceres::CostFunction* Create(double roll, double pitch, double w = 1.0) {
        return new ceres::AutoDiffCostFunction<CostFunction_Imu, 4, 1, 1>(
            new CostFunction_Imu(roll, pitch, w));
    }

private:
    template <typename T>
    Eigen::Quaternion<T> CreateQuaternion(T roll, T pitch, T yaw) const {
        Eigen::AngleAxis<T> roll_angle(roll, Eigen::Matrix<T, 3, 1>::UnitX());
        Eigen::AngleAxis<T> pitch_angle(pitch, Eigen::Matrix<T, 3, 1>::UnitY());
        Eigen::AngleAxis<T> yaw_angle(yaw, Eigen::Matrix<T, 3, 1>::UnitZ());

        Eigen::Quaternion<T> q = yaw_angle * pitch_angle * roll_angle;
        return q;
    }

    double roll_, pitch_;
    double w_;
};