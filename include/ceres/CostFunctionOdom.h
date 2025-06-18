#include <ceres/ceres.h>
#include <Eigen/Dense>
#include "ceres/autodiff_cost_function.h"
#include <utils.hpp>

class CostFunctionOdom
{
public:
    CostFunctionOdom(const Eigen::Vector3d& t_ab, double w = 1.0)
        : t_ab_odom_(t_ab), w_(w) {}

    template <typename T>
    bool operator()(const T* const t_a_ptr,      // tx,ty,tz  (3)
                    const T* const t_b_ptr,      // tx,ty,tz  (3)
                    const T* const q_a_ptr,      // qw,qx,qy,qz (4)
                    T* residuals_ptr) const
    {
        Eigen::Quaternion<T> q_a(q_a_ptr[0], q_a_ptr[1],
                                 q_a_ptr[2], q_a_ptr[3]);
        q_a.normalize();

        Eigen::Matrix<T,3,1> t_a(t_a_ptr[0], t_a_ptr[1], t_a_ptr[2]);
        Eigen::Matrix<T,3,1> t_b(t_b_ptr[0], t_b_ptr[1], t_b_ptr[2]);

        Eigen::Matrix<T,3,1> t_ab_meas = t_ab_odom_.template cast<T>();

        Eigen::Matrix<T,3,1> t_ab_est = q_a.inverse() * (t_b - t_a);

        Eigen::Map<Eigen::Matrix<T,1,1>> r(residuals_ptr);
        r(0) = (t_ab_est(2) - t_ab_meas(2)) * T(w_);

        return true;
    }

   
    static ceres::CostFunction* Create(const Eigen::Vector3d& t_ab,
                                       double w = 1.0)
    {
        return new ceres::AutoDiffCostFunction< CostFunctionOdom,
                                                1,   
                                                3,   // t_a
                                                3,   // t_b
                                                4    // q_a
                                              >( new CostFunctionOdom(t_ab, w) );
    }

private:
    const Eigen::Vector3d t_ab_odom_;
    double w_;
};
