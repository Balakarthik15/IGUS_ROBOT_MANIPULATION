from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration

import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    
    # 1. FIX: Automatically inject Gazebo environment paths inside Python
    description_share = get_package_share_directory('igus_rebel_description')
    worlds_dir = os.path.join(description_share, 'worlds')
    
    # Update the environment path so Gazebo can discover 'rebel_world.sdf'
    current_gz_path = os.environ.get('GZ_SIM_RESOURCE_PATH', '')
    os.environ['GZ_SIM_RESOURCE_PATH'] = f"{current_gz_path}:{worlds_dir}:{description_share}/share"

    use_sim_time_arg = DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation (Gazebo) clock if true')

    # Include the world launch from Gazebo using your package-relative target
    gz_sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={
            'gz_args': '-r rebel_world.sdf'
        }.items()
    )

    # 2. FIX: Synchronized package name to match your active 'rebel_movieit' package
    moveit_pkg_name = 'igus_rebel_moveit_config'
    
    gz_bridge_config_file = os.path.join(
        get_package_share_directory(moveit_pkg_name), 'config', 'gz_bridge.yaml'
    )

    gz_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name="param_bridge",
        output='screen',
        arguments=['--ros-args', '-p', ['config_file:=', gz_bridge_config_file]],
        parameters=[
            {
                'use_sim_time': LaunchConfiguration('use_sim_time'),
            }
        ],
    )

    # Spawn the robot model in Gazebo (Placed at z=0.80 to rest safely on the table!)
    spawn_node = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-topic', 'robot_description',
            '-name', 'igus_rebel',
            '-z', '0.80'
        ],
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time'),
        }],
        output='screen'
    ) 

    # Include the ROS controllers launch file
    ros_controllers_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory(moveit_pkg_name), 'launch', 'ros_controllers.launch.py')
        ),
    )
    
    return LaunchDescription([
        use_sim_time_arg,
        gz_bridge,
        gz_sim_launch,
        spawn_node,
        TimerAction(period=10.0, actions=[ros_controllers_launch]),
    ])