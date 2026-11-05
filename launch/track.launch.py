"""ros2 launch robothub track.launch.py mission:=straight|curve|barrier|intersection|follow"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    params = os.path.join(get_package_share_directory('robothub'), 'config', 'robothub.yaml')
    return LaunchDescription([
        DeclareLaunchArgument('mission', default_value='straight'),
        DeclareLaunchArgument('params', default_value=params),
        Node(package='robothub', executable='track_mission', name='track_mission', output='screen',
             parameters=[LaunchConfiguration('params'), {'mission': LaunchConfiguration('mission')}]),
    ])
