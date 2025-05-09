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
#include <sensor_msgs/point_cloud2_iterator.hpp>


// RadarPclProcessor class handles point cloud processing, transformation between sensor frames,
// and ego-velocity estimation using radar data.
class radarPclProcessor : public rclcpp::Node {
public:
    radarPclProcessor() : Node("radar_pcl_processor") {
        
        declareParams();
        // Retrieve parameters and set up communication channels.
        getParams();
        setupSubscribersAndPublishers();
        initializeTransformation();
    }

private:


    void declareParams()
    {
        // topics & flags
        this->declare_parameter<std::string>("imu_topic", "/arco/idmind_imu/imu");
        this->declare_parameter<std::string>("radar_topic", "/arco/radar/PointCloudDetection");
        this->declare_parameter<bool>("holonomic_vehicle", true);
        this->declare_parameter<bool>("ground_vehicle", true);
        this->declare_parameter<bool>("enable_dynamic_object_removal", true);

        // frame poses [x, y, z, roll, pitch, yaw]
        this->declare_parameter<std::vector<double>>(
        "imu_to_base_link", std::vector<double>{0.0, 0.0, 0.165, 0.0, 0.0, 0.0});
        this->declare_parameter<std::vector<double>>(
        "radar_to_base_link", std::vector<double>{0.45, 0.05, 0.5, 0.0, 0.0, 0.0});

        // channel indices
        this->declare_parameter<int64_t>("intensity_channel", 2);
        this->declare_parameter<int64_t>("doppler_channel", 0);

        // distance / height filters
        this->declare_parameter<double>("distance_near_thresh", 0.1);
        this->declare_parameter<double>("distance_far_thresh", 80.0);
        this->declare_parameter<double>("z_low_thresh", -40.0);
        this->declare_parameter<double>("z_high_thresh", 100.0);

        // ego‑velocity estimator params
        this->declare_parameter<double>("min_dist", 0.5);
        this->declare_parameter<double>("max_dist", 400.0);
        this->declare_parameter<double>("min_db", 5.0);
        this->declare_parameter<double>("elevation_thresh_deg", 50.0);
        this->declare_parameter<double>("azimuth_thresh_deg", 56.5);
        this->declare_parameter<double>("doppler_velocity_correction_factor", 1.0);
        this->declare_parameter<double>("thresh_zero_velocity", 0.05);
        this->declare_parameter<double>("allowed_outlier_percentage", 0.30);
        this->declare_parameter<double>("sigma_zero_velocity_x", 1.0e-3);
        this->declare_parameter<double>("sigma_zero_velocity_y", 3.2e-3);
        this->declare_parameter<double>("sigma_zero_velocity_z", 1.0e-2);
        this->declare_parameter<double>("sigma_offset_radar_x", 0.0);
        this->declare_parameter<double>("sigma_offset_radar_y", 0.0);
        this->declare_parameter<double>("sigma_offset_radar_z", 0.0);
        this->declare_parameter<double>("max_sigma_x", 0.2);
        this->declare_parameter<double>("max_sigma_y", 0.2);
        this->declare_parameter<double>("max_sigma_z", 0.2);
        this->declare_parameter<double>("max_r_cond", 0.2);
        this->declare_parameter<double>("outlier_prob", 0.05);
        this->declare_parameter<double>("success_prob", 0.995);
        this->declare_parameter<double>("N_ransac_points", 5.0);
        this->declare_parameter<double>("inlier_thresh", 0.5);
    }
    // Function to load parameters from the ROS2 parameter server.
    void getParams() {
        rio::RadarEgoVelocityEstimatorConfig config;

        // Retrieve ROS parameters
        this->get_parameter("imu_topic", imu_topic_);
        this->get_parameter("radar_topic", radar_topic_);
        this->get_parameter("enable_dynamic_object_removal", enable_dynamic_object_removal_);
        this->get_parameter("holonomic_vehicle", holonomic_vehicle_);
        this->get_parameter("ground_vehicle", ground_vehicle_);
        this->get_parameter("distance_near_thresh", distance_near_thresh_);
        this->get_parameter("distance_far_thresh", distance_far_thresh_);
        this->get_parameter("z_low_thresh", z_low_thresh_);
        this->get_parameter("z_high_thresh", z_high_thresh_);
        this->get_parameter("imu_to_base_link", imu_pose_);
        this->get_parameter("radar_to_base_link", radar_pose_);

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

        // --- Now print them out ---
        RCLCPP_INFO(get_logger(), "=== Loaded Parameters ===");
        RCLCPP_INFO(get_logger(), "imu_topic: %s", imu_topic_.c_str());
        RCLCPP_INFO(get_logger(), "radar_topic: %s", radar_topic_.c_str());
        RCLCPP_INFO(get_logger(), "enable_dynamic_object_removal: %s",
                    enable_dynamic_object_removal_ ? "true" : "false");
        RCLCPP_INFO(get_logger(), "holonomic_vehicle: %s",
                    holonomic_vehicle_ ? "true" : "false");

        RCLCPP_INFO(get_logger(), "ground_vehicle: %s",
                    ground_vehicle_ ? "true" : "false");

        auto printVec6 = [&](const std::string &name, const std::vector<double> &v) {
        if (v.size() == 6) {
            RCLCPP_INFO(get_logger(), "%s: [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f]",
                        name.c_str(), v[0], v[1], v[2], v[3], v[4], v[5]);
        } else {
            RCLCPP_WARN(get_logger(), "%s has %zu elements (expected 6)", name.c_str(), v.size());
        }
        };
        printVec6("imu_to_base_link", imu_pose_);
        printVec6("radar_to_base_link", radar_pose_);

        RCLCPP_INFO(get_logger(), "distance_near_thresh: %.3f", distance_near_thresh_);
        RCLCPP_INFO(get_logger(), "distance_far_thresh:  %.3f", distance_far_thresh_);
        RCLCPP_INFO(get_logger(), "z_low_thresh:          %.3f", z_low_thresh_);
        RCLCPP_INFO(get_logger(), "z_high_thresh:         %.3f", z_high_thresh_);

        RCLCPP_INFO(get_logger(), "min_dist:                        %.3f", config.min_dist);
        RCLCPP_INFO(get_logger(), "max_dist:                        %.3f", config.max_dist);
        RCLCPP_INFO(get_logger(), "min_db:                          %.3f", config.min_db);
        RCLCPP_INFO(get_logger(), "elevation_thresh_deg:            %.3f", config.elevation_thresh_deg);
        RCLCPP_INFO(get_logger(), "azimuth_thresh_deg:              %.3f", config.azimuth_thresh_deg);
        RCLCPP_INFO(get_logger(), "doppler_velocity_correction_factor: %.3f", config.doppler_velocity_correction_factor);
        RCLCPP_INFO(get_logger(), "thresh_zero_velocity:            %.3f", config.thresh_zero_velocity);
        RCLCPP_INFO(get_logger(), "allowed_outlier_percentage:      %.3f", config.allowed_outlier_percentage);
        RCLCPP_INFO(get_logger(), "sigma_zero_velocity_x:           %.6f", config.sigma_zero_velocity_x);
        RCLCPP_INFO(get_logger(), "sigma_zero_velocity_y:           %.6f", config.sigma_zero_velocity_y);
        RCLCPP_INFO(get_logger(), "sigma_zero_velocity_z:           %.6f", config.sigma_zero_velocity_z);
        RCLCPP_INFO(get_logger(), "sigma_offset_radar_x:            %.6f", config.sigma_offset_radar_x);
        RCLCPP_INFO(get_logger(), "sigma_offset_radar_y:            %.6f", config.sigma_offset_radar_y);
        RCLCPP_INFO(get_logger(), "sigma_offset_radar_z:            %.6f", config.sigma_offset_radar_z);
        RCLCPP_INFO(get_logger(), "max_sigma_x:                     %.3f", config.max_sigma_x);
        RCLCPP_INFO(get_logger(), "max_sigma_y:                     %.3f", config.max_sigma_y);
        RCLCPP_INFO(get_logger(), "max_sigma_z:                     %.3f", config.max_sigma_z);
        RCLCPP_INFO(get_logger(), "max_r_cond:                      %.3f", config.max_r_cond);
        RCLCPP_INFO(get_logger(), "use_cholesky_instead_of_bdcsvd:  %s",
                    config.use_cholesky_instead_of_bdcsvd ? "true" : "false");
        RCLCPP_INFO(get_logger(), "use_ransac:                      %s",
                    config.use_ransac ? "true" : "false");
        RCLCPP_INFO(get_logger(), "outlier_prob:                    %.3f", config.outlier_prob);
        RCLCPP_INFO(get_logger(), "success_prob:                    %.3f", config.success_prob);
        RCLCPP_INFO(get_logger(), "N_ransac_points:                 %.1f", config.N_ransac_points);
        RCLCPP_INFO(get_logger(), "inlier_thresh:                   %.3f", config.inlier_thresh);
    }

    // Set up ROS2 subscribers and publishers for IMU and radar point clouds.
    void setupSubscribersAndPublishers() {

        rclcpp::SensorDataQoS qos; // Use a QoS profile compatible with sensor data

        // Subscribe to IMU data
        imu_subscriber_ = create_subscription<sensor_msgs::msg::Imu>(
            imu_topic_, qos, [this](const sensor_msgs::msg::Imu::SharedPtr msg) {
                imuCallback(msg);
            });

        // Subscribe to radar point cloud data
        radar_subscriber_ = create_subscription<sensor_msgs::msg::PointCloud2>(
            radar_topic_, qos, [this](const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
                cloudCallback(msg);
            });

        // Publishers for processed data
        twist_publisher_ = create_publisher<geometry_msgs::msg::TwistWithCovarianceStamped>("/Ego_Vel_Twist", 5);
        radar_filtered_publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>("/filtered_pointcloud", 10);
        inlier_pc2_publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>("/inlier_pointcloud", 5);
        outlier_pc2_publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>("/outlier_pointcloud", 5);
        raw_pc2_publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>("/raw_pointcloud", 10);
    }


    void initializeTransformation() {
        
        if (imu_pose_.size() != 6 || radar_pose_.size() != 6) {
            throw std::runtime_error("Expected 6-element [x y z roll pitch yaw] vectors");
        }

        // 2) helper lambda to build a 4×4 from [x,y,z, roll,pitch,yaw]
        auto makeTransform = [](const std::vector<double>& v){
            double x = v[0], y = v[1], z = v[2];
            double roll  = v[3], pitch = v[4], yaw   = v[5];

            // Rotation around X (roll)
            cv::Mat Rx = (cv::Mat_<double>(3,3) << 
                1,           0,            0,
                0,  cos(roll), -sin(roll),
                0,  sin(roll),  cos(roll)
            );
            // Rotation around Y (pitch)
            cv::Mat Ry = (cv::Mat_<double>(3,3) << 
                cos(pitch), 0, sin(pitch),
                        0, 1,          0,
                -sin(pitch), 0, cos(pitch)
            );
            // Rotation around Z (yaw)
            cv::Mat Rz = (cv::Mat_<double>(3,3) << 
                cos(yaw), -sin(yaw), 0,
                sin(yaw),  cos(yaw), 0,
                    0,         0, 1
            );

            // Compose in Z·Y·X order (i.e. roll, then pitch, then yaw)
            cv::Mat R = Rz * Ry * Rx;
            // Build 4×4
            cv::Mat T = cv::Mat::eye(4,4,CV_64F);
            R.copyTo(T(cv::Range(0,3), cv::Range(0,3)));
            T.at<double>(0,3) = x;
            T.at<double>(1,3) = y;
            T.at<double>(2,3) = z;
            return T;
        };

        cv::Mat T_imu_baselink   = makeTransform(imu_pose_);
        cv::Mat T_radar_baselink = makeTransform(radar_pose_);

        // 4) compute radar -> imu
        radar_to_imu_ = T_imu_baselink.inv() * T_radar_baselink;

        // RCLCPP_INFO(get_logger(), "radar_to_imu:\n%s",
        //             cv::format(radar_to_imu_, cv::Formatter::FMT_DEFAULT).c_str());
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
    void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr pcl_msg) {

        // Create iterators for the fields we need.
        sensor_msgs::PointCloud2ConstIterator<float> iter_x(*pcl_msg, "x");
        sensor_msgs::PointCloud2ConstIterator<float> iter_y(*pcl_msg, "y");
        sensor_msgs::PointCloud2ConstIterator<float> iter_z(*pcl_msg, "z");
        sensor_msgs::PointCloud2ConstIterator<float> iter_v(*pcl_msg, "v");
        sensor_msgs::PointCloud2ConstIterator<int8_t> iter_rcs(*pcl_msg, "RCS");

        // Number of points in the cloud
        size_t num_points = pcl_msg->width * pcl_msg->height;

        // Create a PCL cloud to hold the transformed points.
        pcl::PointCloud<RadarPointCloudType>::Ptr radar_cloud_raw(new pcl::PointCloud<RadarPointCloudType>);

        // Iterate through all points using the iterators.
        for (size_t i = 0; i < num_points; ++i, ++iter_x, ++iter_y, ++iter_z, ++iter_v, ++iter_rcs) {
            // Check for invalid data.
            if (std::isnan(*iter_x) || std::isinf(*iter_y) || std::isinf(*iter_z))
                continue;

            // Create a homogeneous coordinate vector for the point.
            cv::Mat pt_mat = (cv::Mat_<double>(4, 1) << *iter_x, *iter_y, *iter_z, 1.0);
            if (pt_mat.empty()) {
                RCLCPP_WARN(get_logger(), "pt_mat is empty. Skipping this point.");
                continue;
            }

            // RCLCPP_INFO(get_logger(),
            // "[%zu] RAW → x=%.3f, y=%.3f, z=%.3f",
            // i, *iter_x, *iter_y, *iter_z);

            cv::Mat dst_mat = radar_to_imu_ * pt_mat;

            // Populate a new RadarPointCloudType with the transformed coordinates.
            RadarPointCloudType radar_point;
            radar_point.x = dst_mat.at<double>(0, 0);
            radar_point.y = dst_mat.at<double>(1, 0);
            radar_point.z = dst_mat.at<double>(2, 0);
            radar_point.intensity = static_cast<float>(*iter_rcs);
            radar_point.doppler = *iter_v;

            // RCLCPP_INFO(get_logger(),
            // "[%zu] TRANS → x=%.3f, y=%.3f, z=%.3f, v=%.3f, RCS=%.3f",
            // i, radar_point.x, radar_point.y, radar_point.z, radar_point.doppler, radar_point.intensity);

            radar_cloud_raw->points.push_back(radar_point);
        }

        //  // 6) Build and log the transform matrix
        //  cv::String s = cv::format(radar_to_imu_, cv::Formatter::FMT_DEFAULT);
        //  RCLCPP_INFO(get_logger(), "radar_to_imu:\n%s", s.c_str());

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
        if (estimator_->estimate(pc2_raw_msg, pitch, roll, yaw, holonomic_vehicle_, ground_vehicle_, v_radar, sigma_v_radar, inlier_radar_msg, outlier_radar_msg)) {
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
    std::string imu_topic_, radar_topic_;
    std::vector<double> imu_pose_, radar_pose_;

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr radar_subscriber_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_subscriber_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr radar_filtered_publisher_;
    rclcpp::Publisher<geometry_msgs::msg::TwistWithCovarianceStamped>::SharedPtr twist_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr inlier_pc2_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr outlier_pc2_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr raw_pc2_publisher_;

    cv::Mat radar_to_livox_, thermal_to_rgb_, radar_to_thermal_, rgb_to_livox_, livox_to_rgb_, change_radar_frame_;
    cv::Mat radar_to_imu_;
    std::shared_ptr<rio::RadarEgoVel> estimator_;

    tf2::Quaternion q_previous_;
    tf2::Quaternion q_current_;
    tf2::Quaternion q_rotation_;

    bool initialization_ = true;
    bool enable_dynamic_object_removal_;
    bool holonomic_vehicle_;
    bool ground_vehicle_;
    double distance_near_thresh_;
    double distance_far_thresh_;
    double z_low_thresh_;
    double z_high_thresh_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<radarPclProcessor>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}