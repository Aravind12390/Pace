#!/usr/bin/env python3
"""
Launches the Task 2 sensor-fusion / analytics pipeline on top of the
Task 1 simulation (corridor.launch.py). Run corridor.launch.py separately
(or use the 'include' below) to actually have /odom, /imu, /scan/points
publishing before this pipeline has anything to consume.

Pipeline:
    /odom, /imu  (ground truth, from gz-sim)
        -> noise_injector           -> /odom_noisy, /imu_noisy, /noise/gyro_bias
        -> robot_localization ekf   -> /odometry/filtered  (running pose covariance)
        -> covariance_logger        -> /ekf/pose_covariance_planar (+ CSV log)

    /scan/points (from gz-sim 3D LiDAR)
        -> degeneracy_monitor       -> /scan_matching/information_matrix
                                     -> /scan_matching/degeneracy
"""

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('ddr_pkg')
    ekf_yaml = os.path.join(pkg_share, 'config', 'ekf.yaml')
    corridor_launch = os.path.join(pkg_share, 'launch', 'corridor.launch.py')

    use_sim_time = LaunchConfiguration('use_sim_time')
    launch_sim = LaunchConfiguration('launch_sim')

    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time', default_value='true',
        description='Use simulation (Gazebo) clock'
    )
    declare_launch_sim = DeclareLaunchArgument(
        'launch_sim', default_value='true',
        description='If true, also bring up corridor.launch.py (world + robot + bridge). '
                     'Set false if you already have that running separately.'
    )

    sim_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(corridor_launch),
        condition=IfCondition(launch_sim),
    )

    noise_injector_node = Node(
        package='ddr_pkg',
        executable='noise_injector',
        name='noise_injector',
        output='screen',
        parameters=[{
            'patch_x_min': 10.0,
            'patch_x_max': 15.0,
            'slip_scale': 0.70,
            'sigma_v': 0.02,
            'sigma_g': 0.0025,
            'sigma_b': 1.0e-4,
            'use_sim_time': use_sim_time,
        }],
    )

    ekf_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[ekf_yaml, {'use_sim_time': use_sim_time}],
    )

    covariance_logger_node = Node(
        package='ddr_pkg',
        executable='covariance_logger',
        name='covariance_logger',
        output='screen',
        parameters=[{
            'log_to_csv': True,
            'csv_path': os.path.expanduser('~/deliverables/ekf_pose_covariance_log.csv'),
            'use_sim_time': use_sim_time,
        }],
    )

    degeneracy_monitor_node = Node(
        package='ddr_pkg',
        executable='degeneracy_monitor',
        name='degeneracy_monitor',
        output='screen',
        parameters=[{
            'z_band_min': -0.05,
            'z_band_max': 0.05,
            'max_points': 2000,
            'corr_max_dist': 0.5,
            'normal_k': 8,
            'icp_iterations': 5,
            'degeneracy_eigenvalue_threshold': 5.0,
            'corridor_axis': 'x',
            'use_sim_time': use_sim_time,
        }],
    )

    ghost_obstacle_mapper_node = Node(
        package='ddr_pkg',
        executable='ghost_obstacle_mapper',
        name='ghost_obstacle_mapper',
        output='screen',
        parameters=[{
            'resolution': 0.05,
            'origin_x': -2.0,
            'origin_y': -3.0,
            'width_m': 24.0,
            'height_m': 6.0,
            'submap_window_sec': 5.0,
            'angular_bins': 720,
            'free_margin': 0.10,
            'occupied_vote_threshold': 3,
            'free_vote_threshold': 3,
            'world_frame': 'odom',
            'sensor_frame': 'lidar_link',
            'use_sim_time': use_sim_time,
        }],
    )

    diagnostics_node = Node(
        package='ddr_pkg',
        executable='diagnostics_node',
        name='diagnostics_node',
        output='screen',
        parameters=[{
            'corridor_axis': 'x',
            'lambda_min_floor': 30.0,
            'spike_window': 10,
            'spike_drop_ratio': 0.35,
            'csv_path': os.path.expanduser('~/deliverables/diagnostics_log.csv'),
            'warnings_log_path': os.path.expanduser('~/deliverables/degeneracy_warnings.log'),
            'use_sim_time': use_sim_time,
        }],
    )

    trajectory_logger_node = Node(
        package='ddr_pkg',
        executable='trajectory_logger',
        name='trajectory_logger',
        output='screen',
        parameters=[{
            'gt_topic': '/odom',
            'unfused_topic': '/odom_noisy',
            'fused_topic': '/odometry/filtered',
            'log_dir': os.path.expanduser('~/deliverables'),
            'use_sim_time': use_sim_time,
        }],
    )

    return LaunchDescription([
        declare_use_sim_time,
        declare_launch_sim,
        sim_include,
        noise_injector_node,
        ekf_node,
        covariance_logger_node,
        degeneracy_monitor_node,
        ghost_obstacle_mapper_node,
        diagnostics_node,
        trajectory_logger_node,
    ])
