#include "keyframe.h"
#include "ceres/CostFunction_.h"

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

    CeresGraph() : max_window_size_(0), Keyframes_(nullptr) {} // Constructor por defecto
    CeresGraph(std::vector<KeyFrame>* keyframes, int max_window_size) : Keyframes_(keyframes), max_window_size_(max_window_size) {}
    using PointT = pcl::PointXYZI;

    // Matching Constraints  ------------------------------------------
    void update_constraints(int window_size) {

        if (window_size > max_window_size_) {
            window_size = max_window_size_;
        }

        pcl::GeneralizedIterativeClosestPoint<PointT, PointT> reg;
        reg.setMaximumIterations(100);  
        reg.setTransformationEpsilon(1e-6);  
        reg.setMaxCorrespondenceDistance(0.3);
        reg.setRANSACIterations(15);
        reg.setRANSACOutlierRejectionThreshold(1.5); 

        int n = Keyframes_->size() - 1;
        int id_source = (*Keyframes_)[n].getId();
        Eigen::Matrix4d pose2 = (*Keyframes_)[n].getPose_matrix4d();

        for (size_t i = 1; i <= window_size - 1; i++) {

            int id_target = (*Keyframes_)[n - i].getId();

            pcl::PointCloud<PointT>::Ptr downsampled_source_cloud(new pcl::PointCloud<PointT>);
            pcl::PointCloud<PointT>::Ptr downsampled_target_cloud(new pcl::PointCloud<PointT>);

            pcl::VoxelGrid<PointT> sor;
            sor.setInputCloud((*Keyframes_)[n].getPointCloud());
            sor.setLeafSize(0.1f, 0.1f, 0.1f);
            sor.filter(*downsampled_source_cloud);

            sor.setInputCloud((*Keyframes_)[n - i].getPointCloud());
            sor.filter(*downsampled_target_cloud);

            reg.setInputSource(downsampled_source_cloud);
            reg.setInputTarget(downsampled_target_cloud);

            Eigen::Matrix4d pose1 = (*Keyframes_)[n - i].getPose_matrix4d();
            Eigen::Matrix4d guess = pose1.inverse() * pose2;
            Eigen::Matrix4f guess_float = guess.cast<float>();

            auto aligned = pcl::make_shared<pcl::PointCloud<PointT>>();
            reg.align(*aligned, guess_float);
            
            if (reg.hasConverged()) {

                Eigen::Matrix4d icp_matrix = reg.getFinalTransformation().cast<double>();

                Eigen::Vector3d translation = icp_matrix.block<3, 1>(0, 3);
                Eigen::Matrix3d rotation = icp_matrix.block<3, 3>(0, 0);
                Eigen::Quaterniond rot(rotation);

                double fitness_score = reg.getFitnessScore();
                double w = 1.0 / (fitness_score*2 + 1e-6);

               
                if (fitness_score < 3.5) {
                    std::cerr<<"Contstraint added: " << id_source << " y " << id_target << std::endl;

                    Constraint constraint;
                    constraint.id = id_target;
                    constraint.q = rot.normalized();
                    constraint.t = translation;
                    constraint.w = w;  
                    (*Keyframes_)[n].addConstraint(constraint);
                    
                }
            } else {
                std::cerr << "ICP no not converged for nodes: " << id_source << " y " << id_target << std::endl;
            }
        }
    }

    // Graph Creation --------------------------------------------------
    void create_graph(int window_size) {
        std::cerr<<"Grafo creandose..."<<std::endl;
        if (window_size > max_window_size_) {
            window_size = max_window_size_;
        }
        bool first_constraint = true;
        params.resize(window_size, 7);
        params.setZero();

        for (int i = 0; i < window_size; ++i) {

            KeyFrame& node = (*Keyframes_)[Keyframes_->size() - window_size + i];
            double constraints_number = node.getConstraints().size();

            int window_position_node;
            if(window_size<max_window_size_){window_position_node = node.getId()-1;}else{window_position_node = node.getId() - (Keyframes_->size() - window_size)-1;}
            params(window_position_node, 0) = node.get_roll();
            params(window_position_node, 1) = node.get_pitch();
            params(window_position_node, 2) = node.get_yaw();
            params(window_position_node, 3) = node.get_position()[0];
            params(window_position_node, 4) = node.get_position()[1];
            params(window_position_node, 5) = node.get_position()[2];
            params(window_position_node, 6) = 1.0;

            ceres::LossFunction* loss_function_imu(new ceres::TukeyLoss(1.0));
            ceres::CostFunction* cost_function_imu = CostFunction_Imu::Create(node.get_roll_imu(),node.get_pitch_imu(),0.9);
            graph_.AddResidualBlock(cost_function_imu,loss_function_imu,
                &params(window_position_node,0),&params(window_position_node,1));

            if(window_position_node > 0){

                KeyFrame& node_prev = (*Keyframes_)[Keyframes_->size() - window_size + i - 1];
                int window_position_related = window_position_node - 1;
                params(window_position_related, 0) = node_prev.get_roll();
                params(window_position_related, 1) = node_prev.get_pitch();
                params(window_position_related, 2) = node_prev.get_yaw();
                params(window_position_related, 3) = node_prev.get_position()[0];
                params(window_position_related, 4) = node_prev.get_position()[1];
                params(window_position_related, 5) = node_prev.get_position()[2];
                params(window_position_related, 6) = 1.0;

                ceres::LossFunction* loss_function_odom(new ceres::TukeyLoss(1.0));
                ceres::CostFunction* cost_function_odom = CostFunction_Odom::Create((node.get_Odom_tf()).block<3, 1>(0, 3),1.0);
                graph_.AddResidualBlock(cost_function_odom,loss_function_odom,
                    &params(window_position_related,5),&params(window_position_node,5),
                    &params(window_position_related,0),&params(window_position_related,1),&params(window_position_related,2));
                
                if(window_position_node == 1){
                    graph_.SetParameterBlockConstant(&params(window_position_related,0));
                    graph_.SetParameterBlockConstant(&params(window_position_related,1));
                    graph_.SetParameterBlockConstant(&params(window_position_related,2));
                    graph_.SetParameterBlockConstant(&params(window_position_related,5));
                }
            }
            for (size_t j = 0; j < constraints_number; ++j) {
                
                Constraint constraint = node.getConstraints()[j];
                int constraint_id = constraint.id;
                
                if (constraint_id>=(*Keyframes_)[Keyframes_->size() - window_size].getId()) {

                    auto it = std::find_if(Keyframes_->begin(), Keyframes_->end(), [constraint_id](const KeyFrame& kf) {
                        return kf.getId() == constraint_id;
                    });
                    KeyFrame& node_related = *it;

                    int window_position_related;
                    if(window_size<max_window_size_){window_position_related = node_related.getId()-1;}else{window_position_related = node_related.getId() - (Keyframes_->size() - window_size)-1;}
                    params(window_position_related, 0) = node_related.get_roll();
                    params(window_position_related, 1) = node_related.get_pitch();
                    params(window_position_related, 2) = node_related.get_yaw();
                    params(window_position_related, 3) = node_related.get_position()[0];
                    params(window_position_related, 4) = node_related.get_position()[1];
                    params(window_position_related, 5) = node_related.get_position()[2];
                    params(window_position_related, 6) = 1.0;

                    ceres::LossFunction* loss_function_ori(new ceres::TukeyLoss(1.0));
                    ceres::CostFunction* cost_function_ori_gicp = CostFunction_Ori::Create(constraint.q,constraint.w);
                    graph_.AddResidualBlock(cost_function_ori_gicp,loss_function_ori,
                        &params(window_position_related,0),&params(window_position_related,1),&params(window_position_related,2),
                        &params(window_position_node,0),&params(window_position_node,1),&params(window_position_node,2));

                    ceres::LossFunction* loss_function_trans(new ceres::TukeyLoss(0.5));
                    ceres::CostFunction* cost_function_trans_gicp = CostFunction_Trans::Create(constraint.t,constraint.w);
                    graph_.AddResidualBlock(cost_function_trans_gicp, loss_function_trans,
                        &params(window_position_related, 3),&params(window_position_related, 4),&params(window_position_related,5),
                        &params(window_position_node, 3),&params(window_position_node, 4),&params(window_position_node, 5),
                        &params(window_position_related, 0),&params(window_position_related, 1),&params(window_position_related, 2));
                       
                        if(first_constraint){
                        graph_.SetParameterBlockConstant(&params(window_position_related,0));
                        graph_.SetParameterBlockConstant(&params(window_position_related,1));
                        graph_.SetParameterBlockConstant(&params(window_position_related,2));
                        graph_.SetParameterBlockConstant(&params(window_position_related,3));
                        graph_.SetParameterBlockConstant(&params(window_position_related,4));
                       
                        first_constraint=false;
                    } 

                }

            }
 
         
            
        }

    }  

    // Graph Optimization ----------------------------------------------
    void optimize_graph(int window_size) {
  
        ceres::Solver::Options options;
        options.max_num_iterations = 1000;

        options.minimizer_progress_to_stdout = true;
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;

       ceres::Solver::Summary summary; 
       ceres::Solve(options, &graph_, &summary);
      // std::cerr << summary.FullReport() << "\n"<<std::endl;  
        
        if (window_size > max_window_size_) {
            window_size = max_window_size_;
        }

        update_params(window_size);
    }

    void update_params(int window_size){

        
        for (int i = 0; i < window_size; ++i) {

            if (params(i, 6) == 1.0) { // Solo actualizar si el flag está marcado
                KeyFrame& node = (*Keyframes_)[Keyframes_->size() - window_size + i];
                node.update_params(params(i, 0),params(i, 1),params(i, 2), params(i, 3), params(i, 4), params(i, 5));
                std::cerr << "Pose nodo después: " 
                        << node.get_position()[0] << " " 
                        << node.get_position()[1] << " " 
                        << node.get_position()[2] << std::endl;
            }
        }

    }

  

    private:
    
    int max_window_size_;
    std::vector<KeyFrame>* Keyframes_;
    ceres::Problem graph_;
    Eigen::MatrixXd params;
    pcl::PointCloud<pcl::PointXYZI>::Ptr icp_cloud_ = pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>);

};

    

/* void create_graph(int window_size) {
        
        if (window_size > max_window_size_) {
            window_size = max_window_size_;
        }
        bool first_constraint = true;
    
        for (int i = 0; i < window_size; ++i) {

            KeyFrame& node = (*Keyframes_)[Keyframes_->size() - window_size + i];
            double constraints_number = node.getConstraints().size();

            for (size_t j = 0; j < constraints_number; ++j) {
                
                Constraint constraint = node.getConstraints()[j];
                int constraint_id = constraint.id;

                if (constraint_id>=(*Keyframes_)[Keyframes_->size() - window_size].getId()) {

                    auto it = std::find_if(Keyframes_->begin(), Keyframes_->end(), [constraint_id](const KeyFrame& kf) {
                        return kf.getId() == constraint_id;
                    });
  
                    KeyFrame& node_related = *it;
                    ceres::LossFunction* loss_function_ori(new ceres::TukeyLoss(1.0));
                    ceres::LossFunction* loss_function_trans(new ceres::TukeyLoss(0.2));
                    ceres::LossFunction* loss_function_gicp(new ceres::TukeyLoss(0.5));
                   // Eigen::Quaterniond q_a(node_related.getPose_q()[3], node_related.getPose_q()[0], node_related.getPose_q()[1], node_related.getPose_q()[2]);
                    
                    Eigen::Vector3d constraint_t;
                    constraint_t << constraint.w, constraint.w, constraint.w; 
                    ceres::CostFunction* cost_function_ori_gicp = CostFunctionOri::Create(constraint.q,constraint.w);
                    ceres::CostFunction* cost_function_trans_gicp = CostFunctionTrans::Create(constraint.t,constraint_t);
                    ceres::CostFunction* cost_function_gicp = CostFunction::Create(constraint.t,constraint.q,constraint.w);
                  //  graph_.AddResidualBlock(cost_function_gicp,loss_function_gicp,node_related.getPose_t(),node_related.getPose_q(),node.getPose_t(),node.getPose_q());
                    graph_.AddResidualBlock(cost_function_ori_gicp,loss_function_ori,node_related.getPose_q(),node.getPose_q());
                   graph_.AddResidualBlock(cost_function_trans_gicp, loss_function_trans,node_related.getPose_t(),node.getPose_t(),node_related.getPose_q()); 
                    
                    if(first_constraint){
                        graph_.SetParameterBlockConstant(node_related.getPose_t());
                        graph_.SetParameterBlockConstant(node_related.getPose_q());
                        first_constraint==false;
                    }  
                }

            }

            if(i>0)
            {   
                

                    KeyFrame& prev_node = (*Keyframes_)[Keyframes_->size() - window_size + i -1];
                    ceres::LossFunction* loss_function_ori_odom(new ceres::TukeyLoss(1.0));
                    ceres::LossFunction* loss_function_trans_odom(new ceres::TukeyLoss(1.0));

                    Eigen::Matrix4d odom_tf = node.get_Odom_tf();
                    Eigen::Vector3d odom_tf_t = odom_tf.block<3,1>(0,3);
                    Eigen::Matrix3d rotation_matrix = odom_tf.block<3,3>(0,0);
                    Eigen::Quaterniond odom_tf_q(rotation_matrix);
                    Eigen::Vector3d constraint_t;
                    constraint_t << 1.0, 1.0, 1.0; 
                   // ceres::CostFunction* cost_function_ori_odom = CostFunctionOri::Create(odom_tf_q.normalized(),1.0);
                    //ceres::CostFunction* cost_function_trans_odom = CostFunctionTrans::Create(odom_tf_t,constraint_t);

                  //  graph_.AddResidualBlock(cost_function_ori_odom,loss_function_ori_odom,prev_node.getPose_q(),node.getPose_q());
                  // graph_.AddResidualBlock(cost_function_trans_odom, loss_function_trans_odom,prev_node.getPose_t(),node.getPose_t(),prev_node.getPose_q()); 
                    
                    if(first_constraint){

                       // graph_.SetParameterBlockConstant(prev_node.getPose_t());
                        //graph_.SetParameterBlockConstant(prev_node.getPose_q());
                        first_constraint = false;

                    } 
                
            }  
         
            
        }

    }  */