#include <utility>
#include <vector>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Vector3.h>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <Eigen/Dense>
#include <pcl/features/normal_3d.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>
#include <pcl/common/io.h>
#include <pcl/search/kdtree.h>
using PointT = pcl::PointXYZI;

struct Constraint {
   
    int id; 
    Eigen::Quaterniond q;
    Eigen::Vector3d t;
    double w;


};

class KeyFrame {
    public:
        // Constructor
        KeyFrame(const int id, Eigen::Matrix4d pose, const pcl::PointCloud<pcl::PointXYZI>::Ptr& point_cloud, const rclcpp::Time& timestamp)
        : id_(id), point_cloud_(point_cloud), timestamp_(timestamp) {

        position_ = pose.block<3, 1>(0, 3);

        tf2::Matrix3x3 tf_rotation;
        tf_rotation.setValue(
            pose(0, 0), pose(0, 1), pose(0, 2),
            pose(1, 0), pose(1, 1), pose(1, 2),
            pose(2, 0), pose(2, 1), pose(2, 2)
        );

        tf_rotation.getRPY(roll_, pitch_, yaw_);
        roll_imu_ = roll_;
        pitch_imu_ = pitch_;

        tf2::Quaternion tf_quat;
        tf_rotation.getRotation(tf_quat);

        orientation_ = Eigen::Quaterniond(tf_quat.w(), tf_quat.x(), tf_quat.y(), tf_quat.z());
    }

        // Node ID
        int getId() const {
            return id_;
        }

        rclcpp::Time getTimestamp() const {
            return timestamp_;
        }

        void addConstraint(const Constraint& constraint) {

            constraints_.push_back(constraint);
        }
        void add_Odom_tf(const Eigen::Matrix4d odom_tf)
        {
            odom_tf_ = odom_tf;
        }


        void update_params(const Eigen::Quaterniond& q, const Eigen::Vector3d& position) {
            orientation_ = q.normalized();
            position_ = position;

            tf2::Quaternion tf_quat(q.x(), q.y(), q.z(), q.w());

            double roll, pitch, yaw;
            tf2::Matrix3x3(tf_quat).getRPY(roll, pitch, yaw);

            roll_ = roll;
            pitch_ = pitch;
            yaw_ = yaw;
        }
         Eigen::Matrix4d get_Odom_tf() const {
            return odom_tf_;
        }

        // Struct of constrains
        const std::vector<Constraint>& getConstraints() const {
            return constraints_;
        }
        
        // Associated point_cloud
        const pcl::PointCloud<pcl::PointXYZI>::Ptr& getPointCloud() const {
            return point_cloud_;
        }

        // Associated Pose
        double* getPose_t() { 
            return position_.data();
        }

        double* getPose_q() {
            return orientation_.coeffs().data();
        }

        // Associated Pose
        Eigen::Matrix4d getPose_matrix4d() const {

            Eigen::Matrix4d result = Eigen::Matrix4d::Identity();
            Eigen::Quaterniond normalized_orientation = orientation_.normalized();
            result.block<3, 3>(0, 0) = normalized_orientation.toRotationMatrix();
            result.block<3, 1>(0, 3) = position_;
            return result;
        }

        Eigen::Vector3d get_position(){
            return position_;
        }
        double get_yaw() {
            return yaw_;
        }
        double get_roll(){
            return roll_;
        }
        double get_pitch(){
            return pitch_;
        }
        double get_roll_imu(){
            return roll_imu_;
        }
        double get_pitch_imu(){
            return pitch_imu_;
        }

    private:

        int id_; 
        Eigen::Quaterniond orientation_;
        Eigen::Vector3d position_;
        std::vector<Constraint> constraints_;
        pcl::PointCloud<pcl::PointXYZI>::Ptr point_cloud_;
        rclcpp::Time timestamp_;
        Eigen::Matrix4d odom_tf_;
        double roll_,pitch_,yaw_;
        double roll_imu_,pitch_imu_;

};
