#include <ceres/ceres.h>
#include <Eigen/Dense>
#include "ceres/autodiff_cost_function.h"
#include <utils.hpp>

class CostFunctionTrans
{
public:
    CostFunctionTrans(const Eigen::Vector3d& t_ab, double w = 1.0)
        : t_ab_icp_(t_ab), w_(w) {}

   
    template <typename T>
    bool operator()(const T* const t_a_ptr,        // tx,ty,tz   (3)
                    const T* const t_b_ptr,        // tx,ty,tz   (3)
                    const T* const q_a_ptr,        // qw,qx,qy,qz (4)
                    T* residuals_ptr) const
    {
        Eigen::Quaternion<T> q_a(q_a_ptr[0], q_a_ptr[1],
                                 q_a_ptr[2], q_a_ptr[3]);
        q_a.normalize();

        Eigen::Matrix<T,3,1> t_a(t_a_ptr[0], t_a_ptr[1], t_a_ptr[2]);
        Eigen::Matrix<T,3,1> t_b(t_b_ptr[0], t_b_ptr[1], t_b_ptr[2]);

        Eigen::Matrix<T,3,1> t_ab_meas = t_ab_icp_.template cast<T>();

        Eigen::Matrix<T,3,1> t_ab_est = q_a.inverse() * (t_b - t_a);

        Eigen::Matrix<T,3,1> err = t_ab_est - t_ab_meas;

        Eigen::Map<Eigen::Matrix<T,3,1>> r(residuals_ptr);
        r = err * T(0.1) * T(w_);  

        return true;
    }

    // ------------------------------------------------------------
    //  NUEVO Create: bloques 3, 3, 4
    // ------------------------------------------------------------
    static ceres::CostFunction* Create(const Eigen::Vector3d& t_ab,
                                       double w = 1.0)
    {
        return new ceres::AutoDiffCostFunction< CostFunctionTrans,
                                                3,   
                                                3,   // t_a
                                                3,   // t_b
                                                4    // q_a (qw,qx,qy,qz)
                                              >( new CostFunctionTrans(t_ab, w) );
    }

private:
    const Eigen::Vector3d t_ab_icp_;
    double w_;
};
