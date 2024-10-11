#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "tf2_ros/static_transform_broadcaster.h"

class BaselinkTf : public rclcpp::Node {
public:
    BaselinkTf() : Node("baselink_tf") {
        
        this->declare_parameter<std::string>("topic_name", "/odometry");
        std::string topic_name = this->get_parameter("topic_name").as_string();

        // Static Transform Broadcaster
        static_tf_broadcaster_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(this);

        odom_subscriber_ = this->create_subscription<nav_msgs::msg::Odometry>(
            topic_name, 10, std::bind(&BaselinkTf::odomCallback, this, std::placeholders::_1)
        );

        // Initial Static Transform------------------------------------------
        static_transform_.header.frame_id = "map";     // Parent frame
        static_transform_.child_frame_id = "base_link"; // Child frame

        static_transform_.transform.translation.x = 0.0;
        static_transform_.transform.translation.y = 0.0;
        static_transform_.transform.translation.z = 0.0;
        static_transform_.transform.rotation.x = 0.0;
        static_transform_.transform.rotation.y = 0.0;
        static_transform_.transform.rotation.z = 0.0;
        static_transform_.transform.rotation.w = 1.0;

        //Publish------------------------------------------------------------
        static_tf_broadcaster_->sendTransform(static_transform_);
    }

private:
    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        // Dynamic Transform--------------------------------------------------
        // Translation--------------------------------------------------------
        static_transform_.transform.translation.x = msg->pose.pose.position.x;
        static_transform_.transform.translation.y = msg->pose.pose.position.y;
        static_transform_.transform.translation.z = msg->pose.pose.position.z;

        // Rotation-----------------------------------------------------------
        static_transform_.transform.rotation = msg->pose.pose.orientation;

        //Publish-------------------------------------------------------------
        static_tf_broadcaster_->sendTransform(static_transform_);
    }

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_subscriber_;
    std::shared_ptr<tf2_ros::StaticTransformBroadcaster> static_tf_broadcaster_;
    geometry_msgs::msg::TransformStamped static_transform_;
};

int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<BaselinkTf>());
    rclcpp::shutdown();
    return 0;
}