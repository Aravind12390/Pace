#!/usr/bin/env python3
import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration

from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg_share = get_package_share_directory('ddr_pkg')
    ros_gz_sim_share = get_package_share_directory('ros_gz_sim')

    world_file = os.path.join(pkg_share, 'worlds', 'corridor.world')
    xacro_file = os.path.join(pkg_share, 'urdf', 'ddr_robot.urdf.xacro')

    use_sim_time = LaunchConfiguration('use_sim_time')
    headless = LaunchConfiguration('headless')
    robot_x = LaunchConfiguration('robot_x')
    robot_y = LaunchConfiguration('robot_y')

    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time', default_value='true',
        description='Use simulation (Gazebo) clock'
    )
    declare_headless = DeclareLaunchArgument(
        'headless', default_value='false',
        description='Run gz-sim server only (no GUI) if true'
    )
    declare_robot_x = DeclareLaunchArgument(
        'robot_x', default_value='2.0',
        description='Spawn X position (near one end of the 20 m corridor)'
    )
    declare_robot_y = DeclareLaunchArgument(
        'robot_y', default_value='0.0',
        description='Spawn Y position (centered between the walls)'
    )

    gz_sim_headless = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim_share, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': '-r -s ' + world_file}.items(),
        condition=IfCondition(headless),
    )

    gz_sim_gui = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim_share, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': '-r ' + world_file}.items(),
        condition=UnlessCondition(headless),
    )

    # ---------------- robot_state_publisher ----------------
    robot_description = ParameterValue(
        Command(['xacro ', xacro_file]), value_type=str
    )

    rsp_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': use_sim_time,
        }],
    )

    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        name='spawn_ddr_robot',
        output='screen',
        arguments=[
            '-topic', 'robot_description',
            '-name', 'ddr_robot',
            '-x', robot_x, '-y', robot_y, '-z', '0.15',
        ],
        parameters=[{'use_sim_time': use_sim_time}],
    )

    bridge_node = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='ros_gz_bridge',
        output='screen',
        arguments=[
            '/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
            '/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry',
            '/tf@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V',
            '/imu@sensor_msgs/msg/Imu[gz.msgs.IMU',
            '/scan/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked',
            '/joint_states@sensor_msgs/msg/JointState[gz.msgs.Model',
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
        ],
        remappings=[],
        parameters=[{'use_sim_time': use_sim_time}],
    )

    # ---------------- wall shifter (1.5 m sideways every 15 s) ----------------
    wall_shifter_node = Node(
        package='ddr_pkg',
        executable='wall_shifter',
        name='wall_shifter',
        output='screen',
        parameters=[{
            'world_name': 'corridor_world',
            'model_name': 'shifting_wall',
            'period_sec': 15.0,
            'shift_amount': 1.5,
            'x': 10.0,
            'z': 1.5,
            'start_y': 0.75,
            'use_sim_time': use_sim_time,
        }],
    )

    return LaunchDescription([
        declare_use_sim_time,
        declare_headless,
        declare_robot_x,
        declare_robot_y,
        gz_sim_headless,
        gz_sim_gui,
        rsp_node,
        spawn_robot,
        bridge_node,
        wall_shifter_node,
    ])
