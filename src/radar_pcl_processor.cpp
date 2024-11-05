#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <geometry_msgs/msg/twist_with_covariance_stamped.hpp>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include "Eigen/Dense"
#include <opencv2/core/core.hpp>
#include "rio_utils/radar_point_cloud.hpp"
#include "RadarEgoVel.hpp"
#include <chrono>
#include <vector>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/LinearMath/Transform.h>

// RadarProcessor class handles point cloud processing, transformation between sensor frames,
// and ego-velocity estimation using radar data.
class RadarProcessor : public rclcpp::Node {
public:
    RadarProcessor() : Node("Radar_pcl_processor") {
        // Retrieve parameters and set up communication channels.
        getParams();
        setupSubscribersAndPublishers();
        initializeTransformation();
    }

private:
    // Function to load parameters from the ROS2 parameter server.
    void getParams() {
        rio::RadarEgoVelocityEstimatorConfig config;

        // Retrieve ROS parameters
        this->get_parameter("imu_topic", imu_topic_);
        this->get_parameter("radar_topic", radar_topic_);
        this->get_parameter("enable_dynamic_object_removal", enable_dynamic_object_removal_);
        this->get_parameter("holonomic_vehicle", holonomic_vehicle_);
        this->get_parameter("distance_near_thresh", distance_near_thresh_);
        this->get_parameter("distance_far_thresh", distance_far_thresh_);
        this->get_parameter("z_low_thresh", z_low_thresh_);
        this->get_parameter("z_high_thresh", z_high_thresh_);

        // Retrieve estimator configuration parameters
        this->get_parameter("min_dist", config.min_dist);
        this->get_parameter("max_dist", config.max_dist);
        this->get_parameter("min_db", config.min_db);
        this->get_parameter("elevation_thresh_deg", config.elevation_thresh_deg);
        this->get_parameter("azimuth_thresh_deg", config.azimuth_thresh_deg);
        this->get_parameter("doppler_velocity_correction_factor", config.doppler_velocity_correction_factor);
        this->get_parameter("thresh_zero_velocity", config.thresh_zero_velocity);
        this->get_parameter("allowed_outlier_percentage", config.allowed_outlier_percentage);
        this->get_parameter("sigma_zero_velocity_x", config.sigma_zero_velocity_x);
        this->get_parameter("sigma_zero_velocity_y", config.sigma_zero_velocity_y);
        this->get_parameter("sigma_zero_velocity_z", config.sigma_zero_velocity_z);
        this->get_parameter("sigma_offset_radar_x", config.sigma_offset_radar_x);
        this->get_parameter("sigma_offset_radar_y", config.sigma_offset_radar_y);
        this->get_parameter("sigma_offset_radar_z", config.sigma_offset_radar_z);
        this->get_parameter("max_sigma_x", config.max_sigma_x);
        this->get_parameter("max_sigma_y", config.max_sigma_y);
        this->get_parameter("max_sigma_z", config.max_sigma_z);
        this->get_parameter("max_r_cond", config.max_r_cond);
        this->get_parameter("use_cholesky_instead_of_bdcsvd", config.use_cholesky_instead_of_bdcsvd);
        this->get_parameter("use_ransac", config.use_ransac);
        this->get_parameter("outlier_prob", config.outlier_prob);
        this->get_parameter("success_prob", config.success_prob);
        this->get_parameter("N_ransac_points", config.N_ransac_points);
        this->get_parameter("inlier_thresh", config.inlier_thresh);

        // Initialize the radar ego-velocity estimator with the retrieved configuration
        estimator_ = std::make_shared<rio::RadarEgoVel>(config);
    }

    // Set up ROS2 subscribers and publishers for IMU and radar point clouds.
    void setupSubscribersAndPublishers() {
        // Subscribe to IMU data
        imu_subscriber_ = create_subscription<sensor_msgs::msg::Imu>(
            imu_topic_, 10, [this](const sensor_msgs::msg::Imu::SharedPtr msg) {
                imuCallback(msg);
            });

        // Subscribe to radar point cloud data
        radar_subscriber_ = create_subscription<sensor_msgs::msg::PointCloud>(
            radar_topic_, 10, [this](const sensor_msgs::msg::PointCloud::SharedPtr msg) {
                cloudCallback(msg);
            });

        // Publishers for processed data
        twist_publisher_ = create_publisher<geometry_msgs::msg::TwistWithCovarianceStamped>("/Ego_Vel_Twist", 5);
        radar_filtered_publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>("/filtered_pointcloud", 10);
        inlier_pc2_publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>("/inlier_pointcloud", 5);
        outlier_pc2_publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>("/outlier_pointcloud", 5);
        raw_pc2_publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>("/raw_pointcloud", 10);
    }

    // Initialize transformation matrices between different sensor frames
    void initializeTransformation() {
        // Transformation from Livox to RGB frame (change if needed)
        livox_to_rgb_ = (cv::Mat_<double>(4,4) <<
            -0.006878330000, -0.999969000000, 0.003857230000, 0.029164500000,
            -7.737180000000E-05, -0.003856790000, -0.999993000000, 0.045695200000,
            0.999976000000, -0.006878580000, -5.084110000000E-05, -0.19018000000,
            0, 0, 0, 1);
        rgb_to_livox_ = livox_to_rgb_.inv();

        thermal_to_rgb_ = (cv::Mat_<double>(4,4) <<
            0.9999526089706319, 0.008963747151337641, -0.003798822163962599, 0.18106962419014,
            -0.008945181135788245, 0.9999481006917174, 0.004876439015823288, -0.04546324090016857,
            0.00384233617405678, -0.004842226763999368, 0.999980894463835, 0.08046453079998771,
            0, 0, 0, 1);

        radar_to_thermal_ = (cv::Mat_<double>(4,4) <<
            0.999665, 0.00925436, -0.0241851, -0.0248342,
            -0.00826999, 0.999146, 0.0404891, 0.0958317,
            0.0245392, -0.0402755, 0.998887, 0.0268037,
            0, 0, 0, 1);

        change_radar_frame_ = (cv::Mat_<double>(4,4) <<
            0, -1, 0, 0,
            0, 0, -1, 0,
            1, 0, 0, 0,
            0, 0, 0, 1);

        // Compute the transformation from radar to Livox frame
        radar_to_livox_ = rgb_to_livox_ * thermal_to_rgb_ * radar_to_thermal_ * change_radar_frame_;
    }

    // IMU callback for processing orientation and updating quaternions (change if needed)
    void imuCallback(const sensor_msgs::msg::Imu::SharedPtr imu_msg) {
        // Convert IMU orientation to Eigen quaternion
        Eigen::Quaterniond q_ahrs(
            imu_msg->orientation.w,
            imu_msg->orientation.x,
            imu_msg->orientation.y,
            imu_msg->orientation.z);

        // Apply rotation adjustments
        Eigen::Quaterniond q_r =
            Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitZ()) *
            Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitY()) *
            Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitX());
        Eigen::Quaterniond q_rr =
            Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitZ()) *
            Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitY()) *
            Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX());

        Eigen::Quaterniond q_out = q_r * q_ahrs * q_rr;

        // Update the current orientation quaternion
        q_current_ = tf2::Quaternion(q_out.x(), q_out.y(), q_out.z(), q_out.w());
    }

    // Radar point cloud callback for filtering and processing radar data
    void cloudCallback(const sensor_msgs::msg::PointCloud::SharedPtr pcl_msg) {
        // Define a custom point type with Doppler velocity
        RadarPointCloudType radar_point_raw;
        pcl::PointCloud<RadarPointCloudType>::Ptr radar_cloud_raw(new pcl::PointCloud<RadarPointCloudType>);

        // Filter and transform radar data from the radar frame to Livox frame
        for (size_t i = 0; i < pcl_msg->points.size(); ++i) {
            if (pcl_msg->channels[2].values[i] > 0.0) {
                if (std::isnan(pcl_msg->points[i].x) || std::isinf(pcl_msg->points[i].y) || std::isinf(pcl_msg->points[i].z))
                    continue;

                // Transform point from radar frame to Livox frame
                cv::Mat pt_mat = (cv::Mat_<double>(4, 1) << pcl_msg->points[i].x, pcl_msg->points[i].y, pcl_msg->points[i].z, 1);
                if (pt_mat.empty()) {
                    RCLCPP_WARN(get_logger(), "pt_mat is empty. Skipping this point.");
                    continue;
                }
                cv::Mat dst_mat = radar_to_livox_ * pt_mat;

                // Populate the radar point with transformed coordinates and Doppler data
                radar_point_raw.x = dst_mat.at<double>(0, 0);
                radar_point_raw.y = dst_mat.at<double>(1, 0);
                radar_point_raw.z = dst_mat.at<double>(2, 0);
                radar_point_raw.intensity = pcl_msg->channels[2].values[i];
                radar_point_raw.doppler = pcl_msg->channels[0].values[i];
                radar_cloud_raw->points.push_back(radar_point_raw);
            }
        }

        // Publish the raw radar point cloud data
        sensor_msgs::msg::PointCloud2 pc2_raw_msg;
        pcl::toROSMsg(*radar_cloud_raw, pc2_raw_msg);
        pc2_raw_msg.header.stamp = pcl_msg->header.stamp;
        pc2_raw_msg.header.frame_id = "base_link";
        raw_pc2_publisher_->publish(pc2_raw_msg);

        // Estimate ego velocity based on the radar data
        Eigen::Vector3d v_radar, sigma_v_radar;
        sensor_msgs::msg::PointCloud2 inlier_radar_msg, outlier_radar_msg;

        if (initialization_) {
            q_previous_ = q_current_;
            initialization_ = false;
        }

        // Calculate rotation between previous and current orientations
        q_rotation_ = q_previous_.inverse() * q_current_;
        tf2::Matrix3x3 q_rotation_matrix(q_rotation_.normalize());
        double roll, pitch, yaw;
        q_rotation_matrix.getRPY(roll, pitch, yaw);

        // Estimate ego velocity using the radar ego-velocity estimator
        if (estimator_->estimate(pc2_raw_msg, pitch, roll, yaw, holonomic_vehicle_, v_radar, sigma_v_radar, inlier_radar_msg, outlier_radar_msg)) {
            // Publish the estimated twist with covariance
            geometry_msgs::msg::TwistWithCovarianceStamped twist_msg;
            twist_msg.header.stamp = pc2_raw_msg.header.stamp;
            twist_msg.twist.twist.linear.x = v_radar.x();
            twist_msg.twist.twist.linear.y = v_radar.y();
            twist_msg.twist.twist.linear.z = v_radar.z();
            twist_publisher_->publish(twist_msg);
        } else {
            RCLCPP_WARN(this->get_logger(), "Velocity estimation failed.");
        }

        // Process inlier radar point cloud
        pcl::PointCloud<pcl::PointXYZI>::Ptr radar_cloud_inlier(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::PointCloud<pcl::PointXYZI>::Ptr radar_cloud_raw_(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::fromROSMsg(inlier_radar_msg, *radar_cloud_inlier);
        pcl::fromROSMsg(pc2_raw_msg, *radar_cloud_raw_);

        // Choose source cloud based on dynamic object removal flag
        pcl::PointCloud<pcl::PointXYZI>::ConstPtr source_cloud;
        if (enable_dynamic_object_removal_) {
            source_cloud = radar_cloud_inlier;
        } else {
            source_cloud = radar_cloud_raw_;
        }

        if (source_cloud->empty()) {
            RCLCPP_WARN(this->get_logger(), "Source cloud is empty. Skipping.");
            return;
        }

        // Apply distance and height filtering to the source cloud
        pcl::PointCloud<pcl::PointXYZI>::ConstPtr filtered_cloud = distanceFilter(source_cloud);
        sensor_msgs::msg::PointCloud2 filtered_cloud_msg;
        pcl::toROSMsg(*filtered_cloud, filtered_cloud_msg);
        filtered_cloud_msg.header.stamp = pc2_raw_msg.header.stamp;
        filtered_cloud_msg.header.frame_id = "base_link";
        radar_filtered_publisher_->publish(filtered_cloud_msg);

        // Update previous orientation
        q_previous_ = q_current_;
    }

    // Distance and height filter for point clouds
    pcl::PointCloud<pcl::PointXYZI>::ConstPtr distanceFilter(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& cloud) const {
        pcl::PointCloud<pcl::PointXYZI>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZI>);
        filtered->reserve(cloud->size());

        std::copy_if(cloud->begin(), cloud->end(), std::back_inserter(filtered->points), [&](const pcl::PointXYZI& p) {
            double distance = p.getVector3fMap().norm();
            double z = p.z;
            return distance > distance_near_thresh_ && distance < distance_far_thresh_ && z < z_high_thresh_ && z > z_low_thresh_;
        });

        filtered->width = filtered->size();
        filtered->height = 1;
        filtered->is_dense = false;
        filtered->header = cloud->header;

        return filtered;
    }

    // Member variables
    std::string imu_topic_;
    std::string radar_topic_;

    rclcpp::Subscription<sensor_msgs::msg::PointCloud>::SharedPtr radar_subscriber_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_subscriber_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr radar_filtered_publisher_;
    rclcpp::Publisher<geometry_msgs::msg::TwistWithCovarianceStamped>::SharedPtr twist_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr inlier_pc2_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr outlier_pc2_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr raw_pc2_publisher_;

    cv::Mat radar_to_livox_, thermal_to_rgb_, radar_to_thermal_, rgb_to_livox_, livox_to_rgb_, change_radar_frame_;
    std::shared_ptr<rio::RadarEgoVel> estimator_;

    tf2::Quaternion q_previous_;
    tf2::Quaternion q_current_;
    tf2::Quaternion q_rotation_;

    bool initialization_ = true;
    bool enable_dynamic_object_removal_;
    bool holonomic_vehicle_;
    double distance_near_thresh_;
    double distance_far_thresh_;
    double z_low_thresh_;
    double z_high_thresh_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<RadarProcessor>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}