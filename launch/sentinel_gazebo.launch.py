"""Gazebo Sim (Harmonic) + SentinelBot Mk II in the TurtleBot3 world.

    ros2 launch sentinel_patrol sentinel_gazebo.launch.py
    ros2 launch sentinel_patrol sentinel_gazebo.launch.py world:=/path/to/other.world x_pose:=-2.0 y_pose:=-0.5

Then, in another terminal:
    ros2 launch sentinel_patrol patrol.launch.py holonomic:=true sensor_mode:=per_sensor
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    pkg = get_package_share_directory('sentinel_patrol')
    tb3 = get_package_share_directory('turtlebot3_gazebo')
    ros_gz_sim = get_package_share_directory('ros_gz_sim')

    world = LaunchConfiguration('world')
    x_pose = LaunchConfiguration('x_pose')
    y_pose = LaunchConfiguration('y_pose')
    gui = LaunchConfiguration('gui')

    model_sdf = os.path.join(pkg, 'models', 'sentinel_mk2', 'model.sdf')
    bridge_yaml = os.path.join(pkg, 'config', 'sentinel_mk2_bridge.yaml')
    resource_path = os.pathsep.join([
        os.path.join(pkg, 'models'),
        os.path.join(tb3, 'models'),
        os.environ.get('GZ_SIM_RESOURCE_PATH', ''),
    ])

    return LaunchDescription([
        DeclareLaunchArgument('world', default_value=os.path.join(pkg, 'worlds', 'sentinel_world.sdf'),
                              description='TurtleBot3 arena + ApplyLinkWrench system for the holonomic base'),
        DeclareLaunchArgument('x_pose', default_value='-2.0'),
        DeclareLaunchArgument('y_pose', default_value='-0.5'),
        DeclareLaunchArgument('gui', default_value='true'),
        SetEnvironmentVariable('GZ_SIM_RESOURCE_PATH', resource_path),

        # server and GUI as separate processes (a shared process creates every rendering sensor
        # twice and crashes the gpu_lidar render thread)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(ros_gz_sim, 'launch', 'gz_sim.launch.py')),
            launch_arguments={'gz_args': ['-r -s -v2 ', world], 'on_exit_shutdown': 'true'}.items(),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(ros_gz_sim, 'launch', 'gz_sim.launch.py')),
            launch_arguments={'gz_args': '-g -v2 ', 'on_exit_shutdown': 'true'}.items(),
            condition=IfCondition(gui),
        ),

        Node(
            package='ros_gz_sim', executable='create', name='spawn_sentinel_mk2', output='screen',
            arguments=['-file', model_sdf, '-name', 'sentinel_mk2',
                       '-x', x_pose, '-y', y_pose, '-z', '0.01'],
        ),

        Node(
            package='ros_gz_bridge', executable='parameter_bridge', name='sentinel_bridge', output='screen',
            parameters=[{'config_file': bridge_yaml, 'use_sim_time': True}],
        ),

        # resultant-wrench base controller (stands in for the four mecanum wheels)
        Node(
            package='sentinel_patrol', executable='mecanum_base_sim', name='mecanum_base_sim', output='screen',
            parameters=[{'use_sim_time': True}],
        ),
    ])
