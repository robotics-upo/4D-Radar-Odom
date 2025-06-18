#ifndef COST_FUNCTION_IMU_H
#define COST_FUNCTION_IMU_H

#include <ceres/ceres.h>
#include <Eigen/Dense>

class CostFunctionImu {
public:
    CostFunctionImu(double roll, double pitch, double w = 1.0)
        : roll_(roll), pitch_(pitch), w_(w) {}

    template <typename T>
    bool operator()(const T* const q_ptr,  // [qw, qx, qy, qz]
                    T* residuals_ptr) const {
        Eigen::Quaternion<T> q_a(q_ptr[0],
                                 q_ptr[1],
                                 q_ptr[2],
                                 q_ptr[3]);
        q_a.normalize();

        Eigen::Matrix<T,3,3> Ra = q_a.toRotationMatrix();
        T yaw_a = ceres::atan2(Ra(1,0), Ra(0,0));

        Eigen::AngleAxis<T> rx(T(roll_),  Eigen::Matrix<T,3,1>::UnitX());
        Eigen::AngleAxis<T> ry(T(pitch_), Eigen::Matrix<T,3,1>::UnitY());
        Eigen::AngleAxis<T> rz(yaw_a,      Eigen::Matrix<T,3,1>::UnitZ());
        Eigen::Quaternion<T> q_imu = rz * ry * rx;
        q_imu.normalize();

        Eigen::Quaternion<T> q_err =  q_a*q_imu.inverse();
        q_err.normalize();

        Eigen::Map<Eigen::Matrix<T, 4, 1>> residuals(residuals_ptr);
        residuals(0) = q_err.x() * T(w_);
        residuals(1) = q_err.y() * T(w_);
        residuals(2) = q_err.z() * T(w_);
        residuals(3) = (T(1.0) - q_err.w() * q_err.w()) * T(w_);

        return true;
    }

    static ceres::CostFunction* Create(double roll,
                                       double pitch,
                                       double w = 1.0) {
        return new ceres::AutoDiffCostFunction<
            CostFunctionImu,
            4,  
            4   
        >(new CostFunctionImu(roll, pitch, w));
    }

private:
    double roll_, pitch_, w_;
};

#endif  
