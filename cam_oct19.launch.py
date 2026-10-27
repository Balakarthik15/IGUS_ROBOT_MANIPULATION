""" Static transform publisher acquired via MoveIt 2 hand-eye calibration """
""" EYE-TO-HAND: base_link -> camera_color_optical_frame """
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    nodes = [
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            output="log",
            arguments=[
                "--frame-id",
                "base_link",
                "--child-frame-id",
                "camera_color_optical_frame",
                "--x",
                "0.438016",
                "--y",
                "0.0305018",
                "--z",
                "0.0577739",
                "--qx",
                "0.327816",
                "--qy",
                "0.377734",
                "--qz",
                "-0.118717",
                "--qw",
                "0.857765",
                # "--roll",
                # "0.916915",
                # "--pitch",
                # "0.606724",
                # "--yaw",
                # "-0.581608",
            ],
        ),
    ]
    return LaunchDescription(nodes)
