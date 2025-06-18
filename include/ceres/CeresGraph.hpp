#include "keyframe.h"

#include "ceres/CostFunctionOri.h"
#include "ceres/CostFunctionImu.h"
#include "ceres/CostFunctionTrans.h"
#include "ceres/CostFunctionOdom.h"
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
        #pragma omp parallel for num_threads(20)
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
                   #pragma omp critical
                    {
                        (*keyframes_)[n].addConstraint(constraint);
                    }
                }
            } else {
                std::cerr << "ICP did not converge for nodes: " << id_source << " and " << id_target << std::endl;
            }
        }
    }
void create_graph(int window_size)
{
    
    if (window_size > max_window_size_)
        window_size = max_window_size_;

    params_.resize(window_size, 8);      // qw qx qy qz  tx ty tz flag
    params_.setZero();

    std::unordered_set<int> quat_done;
    std::unordered_set<int> pos_done;

    for (int i = 0; i < window_size; ++i)
    {
        KeyFrame& node = (*keyframes_)[keyframes_->size() - window_size + i];
        size_t constraints_number = node.getConstraints().size();

        int window_position_node =
            (window_size < max_window_size_)
                ? node.getId() - 1
                : node.getId() - (keyframes_->size() - window_size) - 1;

        Eigen::Quaterniond q = rpyToQuat(node.get_roll(),
                                         node.get_pitch(),
                                         node.get_yaw());
        params_(window_position_node,0) = q.w();
        params_(window_position_node,1) = q.x();
        params_(window_position_node,2) = q.y();
        params_(window_position_node,3) = q.z();
        params_(window_position_node,4) = node.get_position()[0];
        params_(window_position_node,5) = node.get_position()[1];
        params_(window_position_node,6) = node.get_position()[2];
        params_(window_position_node,7) = 1.0;

        bool firstQuat = quat_done.insert(window_position_node).second;
        if (firstQuat) {
            
            graph_.AddParameterBlock(&params_(window_position_node,0), 4);
            graph_.SetManifold(&params_(window_position_node,0),
                               new ceres::QuaternionManifold());
        } 

        bool firstPos = pos_done.insert(window_position_node).second;
        if (firstPos) {
            
            graph_.AddParameterBlock(&params_(window_position_node,4), 3);
        } 

        // ── IMU ────────────────────────────────────────────────
        ceres::CostFunction* cf_imu =
            CostFunctionImu::Create(node.get_roll_imu(),
                                     node.get_pitch_imu(), 1.0);
        graph_.AddResidualBlock(cf_imu, nullptr,
                              &params_(window_position_node,0));  // q_a (4)

        if (window_position_node > 0)
        {
            KeyFrame& node_prev =
                (*keyframes_)[keyframes_->size() - window_size + i - 1];
            int window_position_related = window_position_node - 1;

            Eigen::Quaterniond q_prev = rpyToQuat(node_prev.get_roll(),
                                                  node_prev.get_pitch(),
                                                  node_prev.get_yaw());
            params_(window_position_related,0)=q_prev.w();
            params_(window_position_related,1)=q_prev.x();
            params_(window_position_related,2)=q_prev.y();
            params_(window_position_related,3)=q_prev.z();
            params_(window_position_related,4)=node_prev.get_position()[0];
            params_(window_position_related,5)=node_prev.get_position()[1];
            params_(window_position_related,6)=node_prev.get_position()[2];
            params_(window_position_related,7)=1.0;

            if (quat_done.insert(window_position_related).second) {
               
                graph_.AddParameterBlock(&params_(window_position_related,0),4);
                graph_.SetManifold(&params_(window_position_related,0),
                                   new ceres::QuaternionManifold());
            }
            if (pos_done.insert(window_position_related).second) {
               
                graph_.AddParameterBlock(&params_(window_position_related,4),3);
            }

            ceres::CostFunction* cf_odom =
                CostFunctionOdom::Create(node.get_Odom_tf().block<3,1>(0,3),5.0);
            graph_.AddResidualBlock(cf_odom, nullptr,
                &params_(window_position_related,4),   // t_a (3)
                &params_(window_position_node,   4),   // t_b (3)
                &params_(window_position_related,0));  // q_a (4)

            if (window_position_node == 1) {
                graph_.SetParameterBlockConstant(&params_(window_position_related,0));
                graph_.SetParameterBlockConstant(&params_(window_position_related,4));
            }
        }

        // ── LOOP-CLOSURE (GICP) ───────────────────────────────
        for (size_t j = 0; j < constraints_number; ++j)
        {
            const Constraint& c = node.getConstraints()[j];
            if (c.id < (*keyframes_)[keyframes_->size() - window_size].getId())
                continue;

            auto it = std::find_if(keyframes_->begin(), keyframes_->end(),
                      [&](const KeyFrame& kf){ return kf.getId()==c.id; });
            KeyFrame& node_related = *it;

            int window_position_related =
                (window_size < max_window_size_)
                    ? node_related.getId()-1
                    : node_related.getId()
                      - (keyframes_->size()-window_size) - 1;

            Eigen::Quaterniond q_rel = rpyToQuat(node_related.get_roll(),
                                                 node_related.get_pitch(),
                                                 node_related.get_yaw());
            params_(window_position_related,0)=q_rel.w();
            params_(window_position_related,1)=q_rel.x();
            params_(window_position_related,2)=q_rel.y();
            params_(window_position_related,3)=q_rel.z();
            params_(window_position_related,4)=node_related.get_position()[0];
            params_(window_position_related,5)=node_related.get_position()[1];
            params_(window_position_related,6)=node_related.get_position()[2];
            params_(window_position_related,7)=1.0;

            if (quat_done.insert(window_position_related).second) {
               
                graph_.AddParameterBlock(&params_(window_position_related,0),4);
                graph_.SetManifold(&params_(window_position_related,0),
                                   new ceres::QuaternionManifold());
            }
            if (pos_done.insert(window_position_related).second) {
               
                graph_.AddParameterBlock(&params_(window_position_related,4),3);
            }

            // ORI
            ceres::LossFunction* loss_function_ori = new ceres::TukeyLoss(1.0);
            ceres::CostFunction* cf_ori =
                CostFunctionOri::Create(c.q, c.w);
            graph_.AddResidualBlock(cf_ori, loss_function_ori,
                &params_(window_position_related,0),   // q_a (4)
                &params_(window_position_node,   0));  // q_b (4)

            // TRANS
            ceres::LossFunction* loss_function_trans = new ceres::TukeyLoss(1.0);
            ceres::CostFunction* cf_tr =
                CostFunctionTrans::Create(c.t, c.w);
            graph_.AddResidualBlock(cf_tr, loss_function_trans,
                &params_(window_position_related,4),   // t_a (3)
                &params_(window_position_node,   4),   // t_b (3)
                &params_(window_position_related,0));  // q_a (4)
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
    void update_params(int window_size)
    {
        for (int i = 0; i < window_size; ++i)
        {
            if (params_(i,7) != 1.0) continue;

            KeyFrame& node =
                (*keyframes_)[keyframes_->size() - window_size + i];

            Eigen::Quaterniond q(params_(i,0), params_(i,1),
                                params_(i,2), params_(i,3)); // w,x,y,z

            Eigen::Vector3d p(params_(i,4),
                            params_(i,5),
                            params_(i,6));

            node.update_params(q, p);
        }
    }

private:
    int max_window_size_;
    std::vector<KeyFrame>* keyframes_;
    ceres::Problem graph_;
    Eigen::Matrix<double,
              Eigen::Dynamic,
              Eigen::Dynamic,
              Eigen::RowMajor> params_;

    pcl::PointCloud<pcl::PointXYZI>::Ptr icp_cloud_ = pcl::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
};