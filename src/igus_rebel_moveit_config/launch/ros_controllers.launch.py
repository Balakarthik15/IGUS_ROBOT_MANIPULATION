from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration

import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    
    robot_trajectory_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["rebel_arm_trajectory_controller", "--controller-manager", "/controller_manager"],
        output="both",
    )

    robot_position_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["rebel_arm_position_controller", "--controller-manager", "/controller_manager"],
        output="both",
    )
    
    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"],
        parameters=[{'use_sim_time': True}],
        output="both",
    )

    gripper_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="gripper_bridge",
        arguments=[
            "/gripper/left/cmd_pos@std_msgs/msg/Float64]gz.msgs.Double",
            "/gripper/right/cmd_pos@std_msgs/msg/Float64]gz.msgs.Double",
            "/gripper/joint_states@sensor_msgs/msg/JointState[gz.msgs.Model",
        ],
        remappings=[("/gripper/joint_states", "/joint_states")],
        parameters=[{'use_sim_time': True}],
        output="both",
    )
    
    return LaunchDescription([
        robot_trajectory_controller_spawner,
        # robot_position_controller_spawner,
        joint_state_broadcaster_spawner,
        gripper_bridge,
    ])