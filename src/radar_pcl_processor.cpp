#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <std_msgs/msg/string.hpp>
#include <geometry_msgs/msg/twist_with_covariance_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include "Eigen/Dense"
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/core.hpp>
#include "rio_utils/radar_point_cloud.hpp"
#include "RadarEgoVel.hpp"
#include <chrono>
#include <vector>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/LinearMath/Transform.h>
  
class RadarProcessor : public rclcpp::Node {
public:

    RadarProcessor() : Node("Radar_pcl_processor") {
        getParams();
        setupSubscribersAndPublishers();
        initializeTransformation();
    }

private:

    void getParams() {

        rio::RadarEgoVelocityEstimatorConfig config;
        this->get_parameter("imu_topic", imu_topic);
        this->get_parameter("radar_topic", radar_topic);
        this->get_parameter("enable_dynamic_object_removal", enable_dynamic_object_removal);
        this->get_parameter("holonomic_vehicle", holonomic_vehicle);
        this->get_parameter("distance_near_thresh", distance_near_thresh);
        this->get_parameter("distance_far_thresh", distance_far_thresh);
        this->get_parameter("z_low_thresh", z_low_thresh);
        this->get_parameter("z_high_thresh", z_high_thresh);
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
        estimator = std::make_shared<rio::RadarEgoVel>(config);

    }

    void setupSubscribersAndPublishers() {


        // Usar los tópicos en las suscripciones
        imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
            imu_topic, 10, [this](const sensor_msgs::msg::Imu::SharedPtr msg) {
                imuCallback(msg);
        });

        radar_sub_ = create_subscription<sensor_msgs::msg::PointCloud>(
            radar_topic, 10, [this](const sensor_msgs::msg::PointCloud::SharedPtr msg) {
                cloudCallback(msg);
        });

        pub_twist = create_publisher<geometry_msgs::msg::TwistWithCovarianceStamped>("/Ego_Vel_Twist", 5);
        radar_filtered_ = create_publisher<sensor_msgs::msg::PointCloud2>("/filtered_pointcloud", 10);
        pub_inlier_pc2 = create_publisher<sensor_msgs::msg::PointCloud2>("/inlier_pointcloud", 5);
        pub_outlier_pc2 = create_publisher<sensor_msgs::msg::PointCloud2>("/outlier_pointcloud", 5);
        pc2_raw_pub = create_publisher<sensor_msgs::msg::PointCloud2>("/raw_pointcloud", 10);
        
    }

    void initializeTransformation() {

        livox_to_rgb = (cv::Mat_<double>(4,4) << 
        -0.006878330000, -0.999969000000, 0.003857230000, 0.029164500000,  
        -7.737180000000E-05, -0.003856790000, -0.999993000000, 0.045695200000,
        0.999976000000, -0.006878580000, -5.084110000000E-05, -0.19018000000,
        0,  0,  0,  1);
        rgb_to_livox = livox_to_rgb.inv();
        thermal_to_rgb = (cv::Mat_<double>(4,4) <<
        0.9999526089706319, 0.008963747151337641, -0.003798822163962599, 0.18106962419014,  
        -0.008945181135788245, 0.9999481006917174, 0.004876439015823288, -0.04546324090016857,
        0.00384233617405678, -0.004842226763999368, 0.999980894463835, 0.08046453079998771,
        0,0,0,1);
        radar_to_thermal = (cv::Mat_<double>(4,4) <<
        0.999665,    0.00925436,  -0.0241851,  -0.0248342,
        -0.00826999, 0.999146,    0.0404891,   0.0958317,
        0.0245392,   -0.0402755,  0.998887,    0.0268037,
        0,  0,  0,  1);
        change_radarframe = (cv::Mat_<double>(4,4) <<
        0, -1, 0, 0,
        0, 0, -1, 0,
        1, 0, 0, 0,
        0,  0,  0,  1);

        radar_to_livox = rgb_to_livox * thermal_to_rgb * radar_to_thermal * change_radarframe;

    }

    void imuCallback(const sensor_msgs::msg::Imu::SharedPtr imu_msg) {
        
        Eigen::Quaterniond q_ahrs(imu_msg->orientation.w,
                                imu_msg->orientation.x,
                                imu_msg->orientation.y,
                                imu_msg->orientation.z);
        Eigen::Quaterniond q_r = 
            Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitZ()) * 
            Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitY()) * 
            Eigen::AngleAxisd(0.00000, Eigen::Vector3d::UnitX());
        Eigen::Quaterniond q_rr = 
            Eigen::AngleAxisd(0.00000, Eigen::Vector3d::UnitZ()) * 
            Eigen::AngleAxisd(0.00000, Eigen::Vector3d::UnitY()) * 
            Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX());
        Eigen::Quaterniond q_out =  q_r * q_ahrs * q_rr;

        q_actual = tf2::Quaternion(q_out.x(), q_out.y(), q_out.z(), q_out.w());
    }

    void cloudCallback(const sensor_msgs::msg::PointCloud::SharedPtr pcl_msg) {
        
        // radarpoint_raw -> (x,y,z,intensity,doppler)
        RadarPointCloudType radarpoint_raw;
        pcl::PointCloud<RadarPointCloudType>::Ptr radarcloud_raw(new pcl::PointCloud<RadarPointCloudType>);

        // Radar_to_Livox
        for (size_t i = 0; i < pcl_msg->points.size(); ++i) {
            if (pcl_msg->channels[2].values[i] > 0.0) {
                if (std::isnan(pcl_msg->points[i].x) || std::isinf(pcl_msg->points[i].y) || std::isinf(pcl_msg->points[i].z))
                    continue;

                cv::Mat ptMat, dstMat;
                ptMat = (cv::Mat_<double>(4, 1) << pcl_msg->points[i].x, pcl_msg->points[i].y, pcl_msg->points[i].z, 1);
                if (ptMat.empty()) {
                    RCLCPP_WARN(get_logger(), "ptMat is empty. Skipping this point.");
                    continue;
                }
                dstMat = radar_to_livox * ptMat;
                radarpoint_raw.x = dstMat.at<double>(0, 0);
                radarpoint_raw.y = dstMat.at<double>(1, 0);
                radarpoint_raw.z = dstMat.at<double>(2, 0);
                radarpoint_raw.intensity = pcl_msg->channels[2].values[i];
                radarpoint_raw.doppler = pcl_msg->channels[0].values[i];
                radarcloud_raw->points.push_back(radarpoint_raw);
            }
        }

        // Publish radarpoint_raw
        sensor_msgs::msg::PointCloud2 pc2_raw_msg;
        pcl::toROSMsg(*radarcloud_raw, pc2_raw_msg);
        pc2_raw_msg.header.stamp = pcl_msg->header.stamp;
        pc2_raw_msg.header.frame_id = "base_link";
        pc2_raw_pub->publish(pc2_raw_msg);

        // Ego Velocity Estimation
        Eigen::Vector3d v_r, sigma_v_r;
        sensor_msgs::msg::PointCloud2 inlier_radar_msg, outlier_radar_msg;

        if (Initialization) {
            q_prev = q_actual;
            Initialization = false;
        }

        q_rot =q_prev.inverse()*q_actual;
        tf2::Matrix3x3 q_rot_matrix(q_rot.normalize());
        double roll, pitch, yaw;
        q_rot_matrix.getRPY(roll, pitch, yaw);

        if (estimator->estimate(pc2_raw_msg, pitch, roll, v_r, sigma_v_r, inlier_radar_msg, outlier_radar_msg)) {
            geometry_msgs::msg::TwistWithCovarianceStamped twist;
            twist.header.stamp = pc2_raw_msg.header.stamp;
            twist.twist.twist.linear.x = v_r.x();
            twist.twist.twist.linear.y = v_r.y();
            twist.twist.twist.linear.z = v_r.z();
            pub_twist->publish(twist);

        } else {
            RCLCPP_WARN(this->get_logger(), "Velocity estimation failed.");
        }

        pcl::PointCloud<pcl::PointXYZI>::Ptr radarcloud_inlier(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::PointCloud<pcl::PointXYZI>::Ptr radarcloud_raw_(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::fromROSMsg(inlier_radar_msg, *radarcloud_inlier);
        pcl::fromROSMsg(pc2_raw_msg, *radarcloud_raw_);

        pcl::PointCloud<pcl::PointXYZI>::ConstPtr src_cloud;
        if(enable_dynamic_object_removal){
            src_cloud = radarcloud_inlier;
        }else{
            src_cloud = radarcloud_raw_;
        }

        if (src_cloud->empty()) {
            RCLCPP_WARN(this->get_logger(), "Source cloud is empty. Skipping.");
            return;
        }

        pcl::PointCloud<pcl::PointXYZI>::ConstPtr filtered = distance_filter(src_cloud);
        sensor_msgs::msg::PointCloud2 cloud_msg;
        pcl::toROSMsg(*filtered, cloud_msg);
        cloud_msg.header.stamp = pc2_raw_msg.header.stamp;
        cloud_msg.header.frame_id = "base_link";
        radar_filtered_->publish(cloud_msg);
        q_prev = q_actual;
    }

    pcl::PointCloud<pcl::PointXYZI>::ConstPtr distance_filter(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& cloud) const {
        pcl::PointCloud<pcl::PointXYZI>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZI>);
        filtered->reserve(cloud->size());
        std::copy_if(cloud->begin(), cloud->end(), std::back_inserter(filtered->points), [&](const pcl::PointXYZI& p) {
            double d = p.getVector3fMap().norm();
            double z = p.z;
            return d > distance_near_thresh && d < distance_far_thresh && z < z_high_thresh && z > z_low_thresh;
        });
        filtered->width = filtered->size();
        filtered->height = 1;
        filtered->is_dense = false;
        filtered->header = cloud->header;
        return filtered;
    }

    std::string imu_topic;
    std::string radar_topic;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud>::SharedPtr radar_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr radar_filtered_;
    rclcpp::Publisher<geometry_msgs::msg::TwistWithCovarianceStamped>::SharedPtr pub_twist;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_inlier_pc2;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_outlier_pc2;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pc2_raw_pub;
    cv::Mat radar_to_livox, thermal_to_rgb, radar_to_thermal, rgb_to_livox, livox_to_rgb, change_radarframe;
    std::shared_ptr<rio::RadarEgoVel> estimator;
    
    tf2::Quaternion q_prev;
    tf2::Quaternion q_actual;
    tf2::Quaternion q_rot;
    
    bool Initialization = true;
    bool enable_dynamic_object_removal;
    bool holonomic_vehicle;
    double distance_near_thresh;
    double distance_far_thresh;
    double z_low_thresh;
    double z_high_thresh;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<RadarProcessor>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
