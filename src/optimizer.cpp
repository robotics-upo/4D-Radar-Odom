#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist_with_covariance_stamped.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <Eigen/Dense>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/LinearMath/Transform.h>
#include <ceres/ceres.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/registration/icp.h>
#include <ceres/CeresGraph.hpp>
#include <utils.hpp>
#include <queue>
#include <mutex>
#include <thread>

class GraphSlam : public rclcpp::Node {
public:
    using PointT = pcl::PointXYZI;

    GraphSlam() : Node("graph_slam"), stop_processing_(false) {
        // Subscribers
        ego_vel_subscriber_ = this->create_subscription<geometry_msgs::msg::TwistWithCovarianceStamped>(
            "/Ego_Vel_Twist", 300,
            [this](const geometry_msgs::msg::TwistWithCovarianceStamped::SharedPtr msg) {
                std::lock_guard<std::mutex> lock(queue_mutex_);
                twist_queue_.push(msg);
            });

        imu_subscriber_ = this->create_subscription<sensor_msgs::msg::Imu>(
            "/vectornav/imu", 1024,
            [this](const sensor_msgs::msg::Imu::SharedPtr msg) {
                std::lock_guard<std::mutex> lock(queue_mutex_);
                imu_queue_.push(transformImuData(msg));
            });

        point_cloud_subscriber_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/filtered_points", 300,
            [this](const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
                std::lock_guard<std::mutex> lock(queue_mutex_);
                point_cloud_queue_.push(msg);
            });

        // Publishers
        odometry_publisher_ = this->create_publisher<nav_msgs::msg::Odometry>("/odometry", 10);
        point_cloud_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/keyframe_cloud", 10);

        // Initialize parameters and start processing thread
        InitializeParams();
        processing_thread_ = std::thread(&GraphSlam::processMessages, this);
    }

    ~GraphSlam() {
        stop_processing_ = true;

        if (!Keyframes.empty()) {
            publishRemainingKeyframes();
        }
        if (processing_thread_.joinable()) {
            processing_thread_.join();
        }
    }

private:
    // Transform IMU data to desired coordinate frame
    sensor_msgs::msg::Imu::SharedPtr transformImuData(const sensor_msgs::msg::Imu::SharedPtr imu_msg) {
        auto imu_data = std::make_shared<sensor_msgs::msg::Imu>();
        imu_data->header.stamp = imu_msg->header.stamp;
        imu_data->header.frame_id = "imu_frame";

        Eigen::Quaterniond q_ahrs(
            imu_msg->orientation.w,
            imu_msg->orientation.x,
            imu_msg->orientation.y,
            imu_msg->orientation.z);

        Eigen::Quaterniond q_r =
            Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitZ()) *
            Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitY()) *
            Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitX());

        Eigen::Quaterniond q_rr =
            Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitZ()) *
            Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitY()) *
            Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX());

        Eigen::Quaterniond q_out = q_r * q_ahrs * q_rr;

        imu_data->orientation.w = q_out.w();
        imu_data->orientation.x = q_out.x();
        imu_data->orientation.y = q_out.y();
        imu_data->orientation.z = q_out.z();

        imu_data->angular_velocity.x = imu_msg->angular_velocity.x;
        imu_data->angular_velocity.y = -imu_msg->angular_velocity.y;
        imu_data->angular_velocity.z = -imu_msg->angular_velocity.z;

        imu_data->linear_acceleration.x = imu_msg->linear_acceleration.x;
        imu_data->linear_acceleration.y = -imu_msg->linear_acceleration.y;
        imu_data->linear_acceleration.z = -imu_msg->linear_acceleration.z;

        return imu_data;
    }

    // Main processing loop
    void processMessages() {
        while (!stop_processing_) {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            if (!twist_queue_.empty() && !imu_queue_.empty() && !point_cloud_queue_.empty()) {
                auto twist_msg = twist_queue_.front();
                auto imu_msg = imu_queue_.front();
                auto cloud_msg = point_cloud_queue_.front();
                lock.unlock();

                if (isTwistBeforeImu(twist_msg, imu_msg) && isImuBeforePointCloud(imu_msg, cloud_msg)) {
                    {
                        std::lock_guard<std::mutex> lock(queue_mutex_);
                        twist_queue_.pop();
                        imu_queue_.pop();
                        point_cloud_queue_.pop();
                    }
                    processOdometry(twist_msg, imu_msg, cloud_msg);
                } else {
                    std::lock_guard<std::mutex> lock(queue_mutex_);
                    if (!isTwistBeforeImu(twist_msg, imu_msg)) {
                        imu_queue_.pop();
                    } else if (!isImuBeforePointCloud(imu_msg, cloud_msg)) {
                        point_cloud_queue_.pop();
                    }
                }
            } else {
                lock.unlock();
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }
    }

    // Check message timestamps for synchronization
    bool isTwistBeforeImu(
        const geometry_msgs::msg::TwistWithCovarianceStamped::SharedPtr& twist_msg,
        const sensor_msgs::msg::Imu::SharedPtr& imu_msg) {
        auto twist_time = twist_msg->header.stamp;
        auto imu_time = imu_msg->header.stamp;
        return (twist_time.sec < imu_time.sec) ||
               (twist_time.sec == imu_time.sec && twist_time.nanosec < imu_time.nanosec);
    }

    bool isImuBeforePointCloud(
        const sensor_msgs::msg::Imu::SharedPtr& imu_msg,
        const sensor_msgs::msg::PointCloud2::SharedPtr& cloud_msg) {
        auto imu_time = imu_msg->header.stamp;
        auto cloud_time = cloud_msg->header.stamp;
        return (imu_time.sec < cloud_time.sec) ||
               (imu_time.sec == cloud_time.sec && imu_time.nanosec < cloud_time.nanosec);
    }

    // Process odometry calculations
    void processOdometry(
        const geometry_msgs::msg::TwistWithCovarianceStamped::SharedPtr& twist_msg,
        const sensor_msgs::msg::Imu::SharedPtr& imu_msg,
        const sensor_msgs::msg::PointCloud2::SharedPtr& cloud_msg) {

        pcl::PointCloud<pcl::PointXYZI>::Ptr actual_cloud_ = pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::fromROSMsg(*cloud_msg, *actual_cloud_);

        tf2::fromMsg(imu_msg->orientation, q_imu);
        q_imu.normalize();

        if (initialization) {
            initialization = false;
            q_prev = q_imu;
        }

        double this_time = imu_msg->header.stamp.sec + imu_msg->header.stamp.nanosec * 1e-9;
        static double last_time = this_time;
        double dt_imu = this_time - last_time;
        last_time = this_time;

        double yaw_vel = imu_msg->angular_velocity.z * dt_imu;
        yaw_integrated += yaw_vel;
        yaw_integrated_kf += yaw_integrated;

        update_estimated_pose(twist_msg, dt_imu);

        if (decide(estimated_pose)) {
            keyframe_pose = estimated_pose;
            keyframe_index++;
            KeyFrame actual_keyframe(keyframe_index, transformToMatrix(keyframe_pose), actual_cloud_, cloud_msg->header.stamp);
            Keyframes.push_back(actual_keyframe);
            yaw_integrated_kf = 0.0;

            if (Keyframes.size() > 1) {
                Eigen::Matrix4d prev_pose = Keyframes[Keyframes.size() - 2].getPose_matrix4d();
                Eigen::Matrix4d actual_pose = Keyframes.back().getPose_matrix4d();
                Eigen::Matrix4d odom_tf = prev_pose.inverse() * actual_pose;
                Keyframes.back().add_Odom_tf(odom_tf);

                CeresGraph graph(&Keyframes, max_window_size);
                graph.update_constraints(Keyframes.size());
                graph.create_graph(Keyframes.size());
                graph.optimize_graph(Keyframes.size());

                tf2::Transform optimized_transform = matrixToTransform(Keyframes.back().getPose_matrix4d());
                tf2::Transform no_opt_transform = keyframe_pose;
                tf2::Quaternion q_diff = no_opt_transform.getRotation().inverse() * optimized_transform.getRotation();
                q_diff.normalize();

                double roll_diff, pitch_diff, yaw_diff;
                tf2::Matrix3x3(q_diff).getRPY(roll_diff, pitch_diff, yaw_diff);

                bool exceeds_threshold = std::abs(roll_diff) > 0.349 ||
                                         std::abs(pitch_diff) > 0.349 ||
                                         std::abs(yaw_diff) > 0.349;

                if (!exceeds_threshold) {
                    estimated_pose = matrixToTransform(Keyframes.back().getPose_matrix4d());
                } else {
                    std::cerr << "Exceeded Threshold" << std::endl;
                }

                nav_msgs::msg::Odometry odometry_msg;
                odometry_msg.header.frame_id = "map";

                if (Keyframes.size() > static_cast<size_t>(max_window_size)) {
                    KeyFrame published_keyframe = Keyframes[Keyframes.size() - max_window_size];
                    odometry_msg.header.stamp = published_keyframe.getTimestamp();
                    tf2::Transform odom_opt = matrixToTransform(published_keyframe.getPose_matrix4d());
                    odometry_msg.pose.pose.position.x = odom_opt.getOrigin().getX();
                    odometry_msg.pose.pose.position.y = odom_opt.getOrigin().getY();
                    odometry_msg.pose.pose.position.z = odom_opt.getOrigin().getZ();
                    odometry_msg.pose.pose.orientation = tf2::toMsg(odom_opt.getRotation().normalize());

                    odometry_publisher_->publish(odometry_msg);

                    sensor_msgs::msg::PointCloud2 keyframe_cloud;
                    pcl::toROSMsg(*published_keyframe.getPointCloud(), keyframe_cloud);
                    keyframe_cloud.header.frame_id = "base_link";
                    keyframe_cloud.header.stamp = imu_msg->header.stamp;

                    point_cloud_publisher_->publish(keyframe_cloud);
                } else {
                    KeyFrame published_keyframe = Keyframes[0];
                }
            }
        }

        yaw_integrated = 0.0;
        q_prev = q_imu;
    }

    // Update the estimated pose using velocity and IMU data
    void update_estimated_pose(
        const geometry_msgs::msg::TwistWithCovarianceStamped::SharedPtr& twist_msg,
        double dt) {

        double translation_x = twist_msg->twist.twist.linear.x * dt;
        double translation_y = twist_msg->twist.twist.linear.y * dt;
        double translation_z = twist_msg->twist.twist.linear.z * dt;
        tf2::Vector3 translation(translation_x, translation_y, translation_z);

        q_rot = q_prev.inverse() * q_imu;
        q_rot.normalize();
        q_prev = q_imu;

        tf2::Matrix3x3 m(q_rot);
        double roll, pitch, yaw;
        m.getRPY(roll, pitch, yaw);

        tf2::Quaternion q_yaw;
        q_yaw.setRPY(roll, pitch, yaw_integrated);
        q_yaw.normalize();

        tf2::Transform transform;
        transform.setRotation(q_yaw);
        transform.setOrigin(translation);
        estimated_pose = estimated_pose * transform;

        tf2::Quaternion q_current;
        tf2::Matrix3x3(estimated_pose.getRotation()).getRotation(q_current);
        q_current.normalize();

        double roll_current, pitch_current, yaw_current;
        tf2::Matrix3x3(q_current).getRPY(roll_current, pitch_current, yaw_current);

        double roll_imu, pitch_imu, yaw_imu;
        tf2::Matrix3x3(q_imu).getRPY(roll_imu, pitch_imu, yaw_imu);

        tf2::Quaternion q_combined;
        q_combined.setRPY(roll_imu - bias.x(), pitch_imu - bias.y(), yaw_current);
        q_combined.normalize();

        estimated_pose.setRotation(q_combined);

        if (!validatePose(estimated_pose)) {
            std::cerr << "Error: estimated pose contains NaN values, skipping update." << std::endl;
        }
    }

    // Validate that the pose does not contain NaN values
    bool validatePose(const tf2::Transform& pose) {
        return !(std::isnan(pose.getOrigin().getX()) ||
                 std::isnan(pose.getOrigin().getY()) ||
                 std::isnan(pose.getOrigin().getZ()) ||
                 std::isnan(pose.getRotation().x()) ||
                 std::isnan(pose.getRotation().y()) ||
                 std::isnan(pose.getRotation().z()) ||
                 std::isnan(pose.getRotation().w()));
    }

    // Initialize parameters from the ROS parameter server
    void InitializeParams() {
        this->declare_parameter<std::vector<double>>("bias_rpy", {0.0, 0.0, 0.0});
        this->declare_parameter<double>("keyframe_delta_trans", 1.0);
        this->declare_parameter<double>("keyframe_delta_angle", 0.1);
        this->declare_parameter<int>("max_window_size", 10);

        std::vector<double> bias_values;
        this->get_parameter("bias_rpy", bias_values);
        bias(0) = bias_values[0];
        bias(1) = bias_values[1];
        bias(2) = bias_values[2];

        this->get_parameter("keyframe_delta_trans", keyframe_delta_trans);
        this->get_parameter("keyframe_delta_angle", keyframe_delta_angle);
        this->get_parameter("max_window_size", max_window_size);

        dt_imu = 0.0;
        initialization = true;
        is_first = true;
        estimated_pose = tf2::Transform::getIdentity();
        yaw_integrated = 0.0;
        keyframe_index = 0;
    }

    // Decide whether to create a new keyframe based on movement thresholds
    bool decide(const tf2::Transform& Pose) {
        if (is_first) {
            is_first = false;
            return true;
        }

        tf2::Transform delta = keyframe_pose.inverse() * Pose;
        tf2::Vector3 translation = delta.getOrigin();
        tf2::Quaternion rotation = delta.getRotation();
        double dx = translation.length();

        double roll, pitch, yaw;
        tf2::Matrix3x3(rotation).getRPY(roll, pitch, yaw);

        return (dx >= keyframe_delta_trans || std::abs(yaw) >= keyframe_delta_angle);
    }

    // Publish remaining keyframes before shutdown
    void publishRemainingKeyframes() {
        size_t start_idx = std::max(0, static_cast<int>(Keyframes.size()) - static_cast<int>(max_window_size));

        for (size_t i = start_idx; i < Keyframes.size(); ++i) {
            nav_msgs::msg::Odometry odometry_msg;
            odometry_msg.header.frame_id = "map";
            odometry_msg.header.stamp = Keyframes[i].getTimestamp();

            tf2::Transform odom_opt = matrixToTransform(Keyframes[i].getPose_matrix4d());
            odometry_msg.pose.pose.position.x = odom_opt.getOrigin().getX();
            odometry_msg.pose.pose.position.y = odom_opt.getOrigin().getY();
            odometry_msg.pose.pose.position.z = odom_opt.getOrigin().getZ();
            odometry_msg.pose.pose.orientation = tf2::toMsg(odom_opt.getRotation().normalize());

            odometry_publisher_->publish(odometry_msg);
        }
    }

    // Member variables
    rclcpp::Subscription<geometry_msgs::msg::TwistWithCovarianceStamped>::SharedPtr ego_vel_subscriber_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_subscriber_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr point_cloud_subscriber_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odometry_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr point_cloud_publisher_;

    std::queue<geometry_msgs::msg::TwistWithCovarianceStamped::SharedPtr> twist_queue_;
    std::queue<sensor_msgs::msg::Imu::SharedPtr> imu_queue_;
    std::queue<sensor_msgs::msg::PointCloud2::SharedPtr> point_cloud_queue_;
    std::mutex queue_mutex_;

    geometry_msgs::msg::TwistWithCovarianceStamped::SharedPtr twist_msg_;
    tf2::Transform estimated_pose;
    tf2::Transform keyframe_pose;
    double yaw_integrated;
    double yaw_integrated_kf;

    tf2::Quaternion q_imu;
    tf2::Quaternion q_prev;
    tf2::Quaternion q_rot;
    tf2::Quaternion q_bias;
    std::vector<KeyFrame> Keyframes;
    Eigen::Vector3d bias;

    bool initialization;
    bool is_first;
    double keyframe_delta_trans;
    double keyframe_delta_angle;
    size_t keyframe_index;
    int max_window_size;
    double dt_imu;
    double pcum;

    std::thread processing_thread_;
    std::atomic<bool> stop_processing_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<GraphSlam>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}