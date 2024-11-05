

#ifndef UTILS_HPP
#define UTILS_HPP

#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/LinearMath/Transform.h>
#include <rclcpp/rclcpp.hpp>
#include <Eigen/Dense>
#include <pcl/registration/icp.h>
#include <pcl/registration/transformation_estimation_point_to_plane.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/search/kdtree.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <Eigen/Dense>
#include <pcl/features/normal_3d.h>



// Print Transform
void print_tf(const tf2::Transform& transform, const std::string& name) {
    tf2::Vector3 translation = transform.getOrigin();
    tf2::Quaternion rotation = transform.getRotation();

    std::cerr << "Transform " << name << ":" << std::endl;
    std::cerr << "  - Posición: (" << translation.getX() << ", " << translation.getY() << ", " << translation.getZ() << ")" << std::endl;
    std::cerr << "  - Orientación: (" << rotation.getX() << ", " << rotation.getY() << ", " << rotation.getZ() << ", " << rotation.getW() << ")" << std::endl;
}

// Conversion from Eigen::Matrix4d to tf2::Transform
tf2::Transform matrixToTransform(const Eigen::Matrix4d& matrix) {
    // Obtener la parte de rotación y la parte de translación de la matriz
    tf2::Matrix3x3 rotation_tf2(matrix(0, 0), matrix(0, 1), matrix(0, 2),
                                matrix(1, 0), matrix(1, 1), matrix(1, 2),
                                matrix(2, 0), matrix(2, 1), matrix(2, 2));

    tf2::Vector3 translation_tf2(matrix(0, 3), matrix(1, 3), matrix(2, 3));

    // Convertir la parte de rotación a un cuaternión
    tf2::Quaternion rotation_quaternion;
    rotation_tf2.getRotation(rotation_quaternion);
    
    // Normalizar el cuaternión de rotación
    rotation_quaternion.normalize();

    // Crear un objeto tf2::Transform
    tf2::Transform transform;
    transform.setRotation(rotation_quaternion);
    transform.setOrigin(translation_tf2);

    return transform;
}

// Conversion from tf2::Transform to Eigen::

Eigen::Matrix4d transformToMatrix(const tf2::Transform& transform) {
    Eigen::Matrix4d matrix;

    // Obtener la rotación y la traslación del tf2::Transform
    tf2::Quaternion rotation = transform.getRotation();
    tf2::Vector3 translation = transform.getOrigin();

    // Construir la matriz de transformación
    matrix.block<3, 3>(0, 0) = Eigen::Quaterniond(rotation.w(), rotation.x(), rotation.y(), rotation.z()).toRotationMatrix();
    matrix.block<3, 1>(0, 3) = Eigen::Vector3d(translation.x(), translation.y(), translation.z());
    matrix.row(3) << 0, 0, 0, 1;


    return matrix;
}


#endif // UTILS_HPP