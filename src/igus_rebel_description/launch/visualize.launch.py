import os
from os import pathsep
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
 
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
    TimerAction,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.substitutions import (
    Command, LaunchConfiguration, PathJoinSubstitution, PythonExpression
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
 
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():
    # Specifying package name and URDF path 
    package_name = "igus_rebel_description"
    urdf_path = 'urdf/igus_rebel_robot2.urdf.xacro'
    
    # Get full path 
    pkg_share = get_package_share_directory(package_name)
    urdf_model_path = os.path.join(pkg_share, urdf_path)

    # FIXED INDENTATION HERE
    # Also wrapped in ParameterValue to ensure ROS 2 handles the Command substitution string properly
    robot_description = ParameterValue(
        Command(['xacro', ' ', urdf_model_path]),
        value_type=str
    )

    world_path =os.path.join(pkg_share,'worlds','rebel_world.sdf')

    # Launch Gazebo world

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': f'-r {world_path}'}.items()
    )


    # Spawn Node to inject robot description into the running world instance
    # Z-axis is set to 0.80m (0.75m table + 0.05m plate height) so it rests exactly on top
    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-topic', 'robot_description',
            '-name', 'igus_rebel',
            '-z', '0.80'
        ],
        output='screen'
    ) 


    robot_state_publisher_node = Node(
        package = "robot_state_publisher",
        executable = "robot_state_publisher",
        parameters=[{"robot_description": robot_description},

                    {"use_sim_time": True}
                    ]
    )

    joint_state_publisher_gui_node = Node(
        package = 'joint_state_publisher_gui',
        executable="joint_state_publisher_gui"
    )
    
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen'
    )

    controller_manager = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[
            {"robot_description": robot_description,
             "use_sim_time": True},
            os.path.join(
                get_package_share_directory("rebel_ros2_controllers"),
                "config",
                "ros2_controllers.yaml",
            ),
        ],
    )

    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        output = "screen",
        arguments=[
            "joint_state_broadcaster",
            "--controller-manager",
            "/controller_manager",
        ],
    )

    arm_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        output = "screen",
        arguments=["arm_controller", "--controller-manager", "/controller_manager"],
    )

    gz_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        output="screen",
        arguments=[
            "/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock"
        ],
    )


    return LaunchDescription([
        robot_state_publisher_node,
        joint_state_publisher_gui_node,
        rviz_node,
        gazebo,
        spawn_robot,
        joint_state_broadcaster_spawner,
        arm_controller_spawner,
        gz_bridge
    ])