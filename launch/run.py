import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    
    package_dir = get_package_share_directory('radar_odom')
    config_radar = os.path.join(package_dir, 'config', 'radar_pcl_processor.yaml')
    config_graph = os.path.join(package_dir, 'config', 'graph_slam.yaml')

    radar_pcl_processor = Node(
        package='radar_odom',
        executable='radar_pcl_processor',
        output='screen',
        name='radar_pcl_processor',
        parameters=[config_radar]
    )

    optimizer = Node(
        package='radar_odom',
        executable='optimizer',
        output='screen',
        name='graph_slam',
        parameters=[config_graph]
    )

    baselink_tf = Node(
        package='radar_odom',
        executable='baselink_tf',
        name='baselink_tf',
        parameters=[{'topic_name': '/odometry'}]
    )

    record = Node(
        package='radar_odom',
        executable='record',
        name='record'
    )

    nodes_to_execute = [radar_pcl_processor,optimizer,record,baselink_tf]


    return LaunchDescription(nodes_to_execute)
