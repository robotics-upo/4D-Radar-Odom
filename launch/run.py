import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    
    package_dir = get_package_share_directory('radar_odom')
    config_uav = os.path.join(package_dir, 'config', 'config_uav.yaml')
    config_agv = os.path.join(package_dir, 'config', 'config_agv.yaml')

    uav_radar_pcl_processor = Node(
        package='radar_odom',
        executable='radar_pcl_processor',
        output='screen',
        name='uav_radar_pcl_processor',
        remappings = [
            ('/filtered_pointcloud', 'uav/filtered_pointcloud'),
            ('/Ego_Vel_Twist', 'uav/Ego_Vel_Twist'),
            ('/inlier_pointcloud', 'uav/inlier_pointcloud'),
            ('/outlier_pointcloud', 'uav/outlier_pointcloud'),
            ('/raw_pointcloud', 'uav/raw_pointcloud')
        ],
        parameters=[config_uav]
    )

    agv_radar_pcl_processor = Node(
        package='radar_odom',
        executable='radar_pcl_processor',
        output='screen',
        name='agv_radar_pcl_processor',
        remappings = [
            ('/filtered_pointcloud', 'agv/filtered_pointcloud'),
            ('/Ego_Vel_Twist', 'agv/Ego_Vel_Twist'),
            ('/inlier_pointcloud', 'agv/inlier_pointcloud'),
            ('/outlier_pointcloud', 'agv/outlier_pointcloud'),
            ('/raw_pointcloud', 'agv/raw_pointcloud')
        ],
        parameters=[config_agv]
    )

    agv_optimizer = Node(
        package='radar_odom',
        executable='optimizer',
        output='screen',
        name='agv_optimizer',
        remappings = [
            ('/filtered_points', 'agv/filtered_pointcloud'),
            ('/Ego_Vel_Twist', 'agv/Ego_Vel_Twist'),
            ('/vectornav/imu', '/arco/idmind_imu/imu'),
            ('/odometry', 'agv/radar_odometry'),
            ('/keyframe_cloud', 'agv/keyframe_cloud'),
        ]
    )

    uav_optimizer = Node(
        package='radar_odom',
        executable='optimizer',
        output='screen',
        name='uav_optimizer',
        remappings = [
            ('/filtered_points', 'uav/filtered_pointcloud'),
            ('/Ego_Vel_Twist', 'uav/Ego_Vel_Twist'),
            ('/vectornav/imu', '/dji_sdk/imu'),
            ('/odometry', 'uav/radar_odometry'),
            ('/keyframe_cloud', 'uav/keyframe_cloud'),
        ]
    )

    agv_baselink_tf = Node(
        package='radar_odom',
        executable='baselink_tf',
        name='agv_baselink_tf',
        parameters=[{'topic_name': 'agv/odometry'}]
    )

    uav_baselink_tf = Node(
        package='radar_odom',
        executable='baselink_tf',
        name='uav_baselink_tf',
        parameters=[{'topic_name': 'uav/odometry'}]
    )

    # record = Node(
    #     package='radar_odom',
    #     executable='record',
    #     name='record'
    # )

    nodes_to_execute = [uav_radar_pcl_processor, agv_radar_pcl_processor]
    #nodes_to_execute = [uav_radar_pcl_processor, agv_radar_pcl_processor, agv_optimizer, uav_optimizer]

    return LaunchDescription(nodes_to_execute)
