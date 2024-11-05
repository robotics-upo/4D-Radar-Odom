#include <ceres/ceres.h>
#include <Eigen/Dense>
#include "ceres/autodiff_cost_function.h"
#include <utils.hpp>

class CostFunction_Trans {
public:
    CostFunction_Trans(const Eigen::Vector3d& t_ab, const double& w = 1.0)
        : t_ab_icp_(t_ab), w_(w) {}

    template <typename T>
    bool operator()(const T* const pose_a_tx_ptr,
                    const T* const pose_a_ty_ptr,
                    const T* const pose_a_tz_ptr,
                    const T* const pose_b_tx_ptr,
                    const T* const pose_b_ty_ptr,
                    const T* const pose_b_tz_ptr,
                    const T* const pose_a_roll_ptr,
                    const T* const pose_a_pitch_ptr,
                    const T* const pose_a_yaw_ptr,
                    T* residuals_ptr) const {

        Eigen::Quaternion<T> q_a = CreateQuaternion(pose_a_roll_ptr, pose_a_pitch_ptr, pose_a_yaw_ptr);
        Eigen::Quaternion<T> q_a_normalized = q_a.normalized();

        Eigen::Matrix<T, 3, 1> t_a(*pose_a_tx_ptr, *pose_a_ty_ptr, *pose_a_tz_ptr);
        Eigen::Matrix<T, 3, 1> t_b(*pose_b_tx_ptr, *pose_b_ty_ptr, *pose_b_tz_ptr);
        Eigen::Matrix<T, 3, 1> t_ab_icp_T = t_ab_icp_.template cast<T>();

        Eigen::Matrix<T, 3, 1> t_ab_ = q_a_normalized.inverse() * (t_b - t_a);
        Eigen::Matrix<T, 3, 1> error_pos =  t_ab_-t_ab_icp_T ;

        Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals(residuals_ptr);
        residuals(0) = error_pos[0] * T(0.1) * T(w_);
        residuals(1) = error_pos[1] * T(0.1) * T(w_);
        residuals(2) = error_pos[2] * T(0.1) * T(w_);

        return true;
    }

    static ceres::CostFunction* Create(const Eigen::Vector3d& t_ab, const double& w = 1.0) {
        return new ceres::AutoDiffCostFunction<CostFunction_Trans, 3, 1,1,1,1,1,1,1,1,1>(
            new CostFunction_Trans(t_ab, w));
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


    const Eigen::Vector3d t_ab_icp_;
    const double w_;
};
