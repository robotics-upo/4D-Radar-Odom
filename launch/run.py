import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    
    package_dir = get_package_share_directory('radar_odom')
    config = os.path.join(package_dir, 'config', 'config.yaml')

    radar_pcl_processor = Node(
        package='radar_odom',
        executable='radar_pcl_processor',
        output='screen',
        name='radar_pcl_processor',
        parameters=[config]
    )

    optimizer = Node(
        package='radar_odom',
        executable='optimizer',
        output='screen',
        name='optimizer'
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
