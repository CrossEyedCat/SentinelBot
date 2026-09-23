"""Launch the FSM node, the /scan -> Range adapter, the map builder and (optionally) RViz.

    ros2 launch sentinel_patrol patrol.launch.py
    ros2 launch sentinel_patrol patrol.launch.py auto_start:=true
    ros2 launch sentinel_patrol patrol.launch.py holonomic:=true      # mecanum platform
    ros2 launch sentinel_patrol patrol.launch.py rviz:=true           # map window, click to drive

Gazebo + the robot are launched separately (see README).
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    pkg = get_package_share_directory('sentinel_patrol')
    params = os.path.join(pkg, 'config', 'patrol_params.yaml')
    rviz_config = os.path.join(pkg, 'rviz', 'sentinel_map.rviz')
    auto_start = LaunchConfiguration('auto_start')
    holonomic = LaunchConfiguration('holonomic')
    use_sim_time = LaunchConfiguration('use_sim_time')
    cmd_vel_stamped = LaunchConfiguration('cmd_vel_stamped')
    sensor_mode = LaunchConfiguration('sensor_mode')
    build_map = LaunchConfiguration('map')
    publish_tf = LaunchConfiguration('publish_tf')

    return LaunchDescription([
        DeclareLaunchArgument('map', default_value='true',
                              description='Build and publish the occupancy map on /map'),
        DeclareLaunchArgument('publish_tf', default_value='true',
                              description='Let the mapper broadcast odom->base_link and the sensor '
                                          'frames. Set false for a robot that already publishes TF'),
        DeclareLaunchArgument('plan', default_value='true',
                              description='Run the costmap + A* planner: goals are routed around '
                                          'obstacles instead of driven at directly'),
        DeclareLaunchArgument('rviz', default_value='false',
                              description='Open RViz2 with the map, the scan and the "2D Goal Pose" '
                                          'tool: clicking a free cell drives the robot there'),
        DeclareLaunchArgument('sensor_mode', default_value='sectors',
                              description='scan_to_range mode: "sectors" (slice one /scan, stock TurtleBot3) '
                                          'or "per_sensor" (Mk II model with one LaserScan per range sensor)'),
        DeclareLaunchArgument('cmd_vel_stamped', default_value='false',
                              description='Publish geometry_msgs/TwistStamped on /cmd_vel '
                                          '(ROS2 Jazzy ros_gz bridge) instead of Twist (Humble)'),
        DeclareLaunchArgument('auto_start', default_value='false',
                              description='Leave IDLE without waiting for /patrol_cmd start'),
        DeclareLaunchArgument('holonomic', default_value='false',
                              description='Enable strafing (mecanum platform)'),
        DeclareLaunchArgument('pilot', default_value='false',
                              description='Steer NAVIGATING with the imitation-trained policy '
                                          '(config/pilot_mlp.npz) instead of the carrot law'),
        DeclareLaunchArgument('pilot_weights', default_value='',
                              description='Policy file for pilot:=true. Empty: config/pilot_mlp.npz'),
        DeclareLaunchArgument('use_sim_time', default_value='true'),

        Node(
            package='sentinel_patrol',
            executable='scan_to_range',
            name='scan_to_range',
            output='screen',
            parameters=[params, {'use_sim_time': use_sim_time, 'mode': sensor_mode}],
        ),
        Node(
            package='sentinel_patrol',
            executable='patrol_fsm',
            name='patrol_fsm',
            output='screen',
            parameters=[params, {
                'auto_start': auto_start,
                'holonomic': holonomic,
                'cmd_vel_stamped': cmd_vel_stamped,
                'use_sim_time': use_sim_time,
                'pilot': LaunchConfiguration('pilot'),
                'pilot_weights': LaunchConfiguration('pilot_weights'),
            }],
        ),
        Node(
            package='sentinel_patrol',
            executable='occupancy_mapper',
            name='occupancy_mapper',
            output='screen',
            condition=IfCondition(build_map),
            parameters=[params, {'use_sim_time': use_sim_time, 'publish_tf': publish_tf}],
        ),
        Node(
            package='sentinel_patrol',
            executable='path_planner',
            name='path_planner',
            output='screen',
            condition=IfCondition(LaunchConfiguration('plan')),
            parameters=[params, {'use_sim_time': use_sim_time}],
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            condition=IfCondition(LaunchConfiguration('rviz')),
            arguments=['-d', rviz_config],
            parameters=[{'use_sim_time': use_sim_time}],
        ),
    ])
