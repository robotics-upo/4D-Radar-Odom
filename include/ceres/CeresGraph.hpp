#include "keyframe.h"

#include "ceres/CostFunction_Ori.h"
#include "ceres/CostFunction_Imu.h"
#include "ceres/CostFunction_Trans.h"
#include "ceres/CostFunction_Odom.h"
#include <ceres/ceres.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <Eigen/Dense>
#include <pcl/features/normal_3d.h>
#include <pcl/surface/organized_fast_mesh.h>
#include <pcl/registration/gicp.h>
#include <pcl/features/from_meshes.h>
#include <pcl/common/common.h>

#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Transform.h>
#include <algorithm>
#include <iostream>
#include <omp.h>
#include <memory>
#include <utils.hpp>
class CeresGraph {
public:
    CeresGraph() : max_window_size_(0), keyframes_(nullptr) {}
    CeresGraph(std::vector<KeyFrame>* keyframes, int max_window_size)
        : max_window_size_(max_window_size), keyframes_(keyframes) {}

    using PointT = pcl::PointXYZI;

    // Update constraints for the optimization graph based on recent keyframes
    void update_constraints(int window_size) {
        // Ensure window size does not exceed the maximum
        if (window_size > max_window_size_) {
            window_size = max_window_size_;
        }

        // Setup the Generalized ICP (GICP) algorithm
        pcl::GeneralizedIterativeClosestPoint<PointT, PointT> reg;
        reg.setMaximumIterations(100);
        reg.setTransformationEpsilon(1e-6);
        reg.setMaxCorrespondenceDistance(0.3);
        reg.setRANSACIterations(15);
        reg.setRANSACOutlierRejectionThreshold(1.5);

        int n = keyframes_->size() - 1;
        int id_source = (*keyframes_)[n].getId();
        Eigen::Matrix4d pose_source = (*keyframes_)[n].getPose_matrix4d();

        // Loop over the previous keyframes within the window
        for (int i = 1; i <= window_size - 1; i++) {
            int id_target = (*keyframes_)[n - i].getId();

            // Downsample the point clouds for source and target keyframes
            pcl::PointCloud<PointT>::Ptr downsampled_source_cloud(new pcl::PointCloud<PointT>);
            pcl::PointCloud<PointT>::Ptr downsampled_target_cloud(new pcl::PointCloud<PointT>);

            pcl::VoxelGrid<PointT> voxel_grid;
            voxel_grid.setLeafSize(0.1f, 0.1f, 0.1f);

            voxel_grid.setInputCloud((*keyframes_)[n].getPointCloud());
            voxel_grid.filter(*downsampled_source_cloud);

            voxel_grid.setInputCloud((*keyframes_)[n - i].getPointCloud());
            voxel_grid.filter(*downsampled_target_cloud);

            // Set the input point clouds for GICP
            reg.setInputSource(downsampled_source_cloud);
            reg.setInputTarget(downsampled_target_cloud);

            // Initial guess for GICP based on poses
            Eigen::Matrix4d pose_target = (*keyframes_)[n - i].getPose_matrix4d();
            Eigen::Matrix4d initial_guess = pose_target.inverse() * pose_source;
            Eigen::Matrix4f initial_guess_float = initial_guess.cast<float>();

            // Align the point clouds using GICP
            auto aligned_cloud = pcl::make_shared<pcl::PointCloud<PointT>>();
            reg.align(*aligned_cloud, initial_guess_float);

            if (reg.hasConverged()) {
                // Obtain the transformation from GICP
                Eigen::Matrix4d icp_transformation = reg.getFinalTransformation().cast<double>();

                // Extract translation and rotation
                Eigen::Vector3d translation = icp_transformation.block<3, 1>(0, 3);
                Eigen::Matrix3d rotation_matrix = icp_transformation.block<3, 3>(0, 0);
                Eigen::Quaterniond rotation(rotation_matrix);

                double fitness_score = reg.getFitnessScore();
                double weight = 1.0 / (fitness_score * 2 + 1e-6);

                // If the fitness score is acceptable, add the constraint
                if (fitness_score < 3.5) {
                    Constraint constraint;
                    constraint.id = id_target;
                    constraint.q = rotation.normalized();
                    constraint.t = translation;
                    constraint.w = weight;
                    (*keyframes_)[n].addConstraint(constraint);
                }
            } else {
                std::cerr << "ICP did not converge for nodes: " << id_source << " and " << id_target << std::endl;
            }
        }
    }

    // Create the optimization graph based on the current window of keyframes
    void create_graph(int window_size) {
        if (window_size > max_window_size_) {
            window_size = max_window_size_;
        }

        // Initialize parameters matrix
        params_.resize(window_size, 7);
        params_.setZero();

        for (int i = 0; i < window_size; ++i) {
            KeyFrame& node = (*keyframes_)[keyframes_->size() - window_size + i];
            size_t constraints_number = node.getConstraints().size();

            int window_position_node;
            if (window_size < max_window_size_) {
                window_position_node = node.getId() - 1;
            } else {
                window_position_node = node.getId() - (keyframes_->size() - window_size) - 1;
            }

            // Set initial parameter values for the node
            params_(window_position_node, 0) = node.get_roll();
            params_(window_position_node, 1) = node.get_pitch();
            params_(window_position_node, 2) = node.get_yaw();
            params_(window_position_node, 3) = node.get_position()[0];
            params_(window_position_node, 4) = node.get_position()[1];
            params_(window_position_node, 5) = node.get_position()[2];
            params_(window_position_node, 6) = 1.0; // Flag to indicate valid parameters

            // Add IMU constraint
            ceres::CostFunction* cost_function_imu = CostFunction_Imu::Create(node.get_roll_imu(), node.get_pitch_imu(), 1.0);
            graph_.AddResidualBlock(cost_function_imu, nullptr,
                &params_(window_position_node, 0), &params_(window_position_node, 1));

            // Add odometry constraint if not the first node
            if (window_position_node > 0) {
                KeyFrame& node_prev = (*keyframes_)[keyframes_->size() - window_size + i - 1];
                int window_position_related = window_position_node - 1;

                // Set initial parameter values for the related node
                params_(window_position_related, 0) = node_prev.get_roll();
                params_(window_position_related, 1) = node_prev.get_pitch();
                params_(window_position_related, 2) = node_prev.get_yaw();
                params_(window_position_related, 3) = node_prev.get_position()[0];
                params_(window_position_related, 4) = node_prev.get_position()[1];
                params_(window_position_related, 5) = node_prev.get_position()[2];
                params_(window_position_related, 6) = 1.0;

                // Add odometry constraint
                ceres::CostFunction* cost_function_odom = CostFunction_Odom::Create(node.get_Odom_tf().block<3, 1>(0, 3), 1.0);
                graph_.AddResidualBlock(cost_function_odom, nullptr,
                    &params_(window_position_related, 0), &params_(window_position_related, 1), &params_(window_position_related, 2),
                    &params_(window_position_related, 3), &params_(window_position_related, 4), &params_(window_position_related, 5),
                    &params_(window_position_node, 3), &params_(window_position_node, 4), &params_(window_position_node, 5));

                // Fix the parameters of the first node to anchor the graph
                if (window_position_node == 1) {
                    graph_.SetParameterBlockConstant(&params_(window_position_related, 0));
                    graph_.SetParameterBlockConstant(&params_(window_position_related, 1));
                    graph_.SetParameterBlockConstant(&params_(window_position_related, 2));
                    graph_.SetParameterBlockConstant(&params_(window_position_related, 3));
                    graph_.SetParameterBlockConstant(&params_(window_position_related, 4));
                    graph_.SetParameterBlockConstant(&params_(window_position_related, 5));
                }
            }

            // Add constraints from GICP (loop closures)
            for (size_t j = 0; j < constraints_number; ++j) {
                Constraint constraint = node.getConstraints()[j];
                int constraint_id = constraint.id;

                if (constraint_id >= (*keyframes_)[keyframes_->size() - window_size].getId()) {
                    // Find the related node
                    auto it = std::find_if(keyframes_->begin(), keyframes_->end(),
                        [constraint_id](const KeyFrame& kf) { return kf.getId() == constraint_id; });
                    KeyFrame& node_related = *it;

                    int window_position_related;
                    if (window_size < max_window_size_) {
                        window_position_related = node_related.getId() - 1;
                    } else {
                        window_position_related = node_related.getId() - (keyframes_->size() - window_size) - 1;
                    }

                    // Set initial parameter values for the related node
                    params_(window_position_related, 0) = node_related.get_roll();
                    params_(window_position_related, 1) = node_related.get_pitch();
                    params_(window_position_related, 2) = node_related.get_yaw();
                    params_(window_position_related, 3) = node_related.get_position()[0];
                    params_(window_position_related, 4) = node_related.get_position()[1];
                    params_(window_position_related, 5) = node_related.get_position()[2];
                    params_(window_position_related, 6) = 1.0;

                    // Add orientation constraint from GICP
                    ceres::LossFunction* loss_function_ori = new ceres::TukeyLoss(1.0);
                    ceres::CostFunction* cost_function_ori_gicp = CostFunction_Ori::Create(constraint.q, constraint.w);
                    graph_.AddResidualBlock(cost_function_ori_gicp, loss_function_ori,
                        &params_(window_position_related, 0), &params_(window_position_related, 1), &params_(window_position_related, 2),
                        &params_(window_position_node, 0), &params_(window_position_node, 1), &params_(window_position_node, 2));

                    // Add translation constraint from GICP
                    ceres::LossFunction* loss_function_trans = new ceres::TukeyLoss(0.5);
                    ceres::CostFunction* cost_function_trans_gicp = CostFunction_Trans::Create(constraint.t, constraint.w);
                    graph_.AddResidualBlock(cost_function_trans_gicp, loss_function_trans,
                        &params_(window_position_related, 3), &params_(window_position_related, 4), &params_(window_position_related, 5),
                        &params_(window_position_node, 3), &params_(window_position_node, 4), &params_(window_position_node, 5),
                        &params_(window_position_related, 0), &params_(window_position_related, 1), &params_(window_position_related, 2));
                }
            }
        }
    }

    // Optimize the graph using Ceres Solver
    void optimize_graph(int window_size) {
        ceres::Solver::Options options;
        options.max_num_iterations = 1000;
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;

        ceres::Solver::Summary summary;
        ceres::Solve(options, &graph_, &summary);

        if (window_size > max_window_size_) {
            window_size = max_window_size_;
        }
        update_params(window_size);
    }

    // Update keyframe parameters after optimization
    void update_params(int window_size) {
        for (int i = 0; i < window_size; ++i) {
            if (params_(i, 6) == 1.0) { // Check if parameters are valid
                KeyFrame& node = (*keyframes_)[keyframes_->size() - window_size + i];
                node.update_params(params_(i, 0), params_(i, 1), params_(i, 2),
                                   params_(i, 3), params_(i, 4), params_(i, 5));
            }
        }
    }

private:
    int max_window_size_;
    std::vector<KeyFrame>* keyframes_;
    ceres::Problem graph_;
    Eigen::MatrixXd params_;

    pcl::PointCloud<pcl::PointXYZI>::Ptr icp_cloud_ = pcl::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
};