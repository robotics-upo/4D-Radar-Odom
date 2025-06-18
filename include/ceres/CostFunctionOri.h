
#include <ceres/ceres.h>
#include <Eigen/Dense>
#include "ceres/autodiff_cost_function.h"
#include <utils.hpp>

class CostFunctionOri
{
public:
    CostFunctionOri(const Eigen::Quaterniond& q_ab, double w = 1.0)
        : q_ab_icp_(q_ab), w_(w) {}

    template <typename T>
    bool operator()(const T* const q_a_ptr,     // [qw,qx,qy,qz]
                    const T* const q_b_ptr,     // [qw,qx,qy,qz]
                    T* residuals_ptr) const
    {
        Eigen::Quaternion<T> q_a(q_a_ptr[0], q_a_ptr[1],
                                 q_a_ptr[2], q_a_ptr[3]);
        Eigen::Quaternion<T> q_b(q_b_ptr[0], q_b_ptr[1],
                                 q_b_ptr[2], q_b_ptr[3]);
        q_a.normalize();
        q_b.normalize();

        Eigen::Quaternion<T> q_ab_meas = q_ab_icp_.template cast<T>();

        Eigen::Quaternion<T> q_ab_est  = q_a.inverse() * q_b;
        q_ab_est.normalize();

        Eigen::Quaternion<T> q_err = q_ab_est * q_ab_meas.inverse();
        q_err.normalize();

        Eigen::Map<Eigen::Matrix<T,4,1>> r(residuals_ptr);
        r(0) = q_err.x() * T(w_);
        r(1) = q_err.y() * T(w_);
        r(2) = q_err.z() * T(w_);
        r(3) = (T(1.0) - q_err.w()*q_err.w()) * T(w_);

        return true;
    }

   
    static ceres::CostFunction* Create(const Eigen::Quaterniond& q_ab,
                                       double w = 1.0)
    {
        return new ceres::AutoDiffCostFunction< CostFunctionOri,
                                                4,   
                                                4,   // q_a
                                                4    // q_b
                                              >( new CostFunctionOri(q_ab, w) );
    }

private:
    const Eigen::Quaterniond q_ab_icp_;
    double w_;
};
