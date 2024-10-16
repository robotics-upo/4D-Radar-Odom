#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include <fstream>

class RecordNode : public rclcpp::Node {
public:
    RecordNode() : Node("pose_logger_node") {
        odom_subscription_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/odometry", 10, std::bind(&RecordNode::odom_callback, this, std::placeholders::_1));
        

        odom_file_ = std::ofstream("odometry.txt");
    }

private:
    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        save_odom_to_file(odom_file_, msg);

    }

    void save_odom_to_file(std::ofstream& file, const nav_msgs::msg::Odometry::SharedPtr msg) {
        std::ostringstream timestamp;
        timestamp << std::fixed << std::setprecision(9)
                  << msg->header.stamp.sec + msg->header.stamp.nanosec / 1e9;

        std::string timestamp_str = timestamp.str();
        
        file << timestamp_str << " "
             << msg->pose.pose.position.x << " "
             << msg->pose.pose.position.y << " "
             << msg->pose.pose.position.z << " "
             << msg->pose.pose.orientation.x << " "
             << msg->pose.pose.orientation.y << " "
             << msg->pose.pose.orientation.z << " "
             << msg->pose.pose.orientation.w << std::endl;
        timestamp_stream.str("");
    }  

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_subscription_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr pose_subscription_;
    std::ofstream odom_file_;
    std::ostringstream timestamp_stream;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<RecordNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
