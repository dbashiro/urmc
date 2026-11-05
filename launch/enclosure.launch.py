"""ros2 launch robothub enclosure.launch.py"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    params = os.path.join(get_package_share_directory('robothub'), 'config', 'robothub.yaml')
    return LaunchDescription([
        DeclareLaunchArgument('params', default_value=params),
        Node(package='robothub', executable='enclosure_mission', name='enclosure_mission', output='screen',
             parameters=[LaunchConfiguration('params')]),
    ])
