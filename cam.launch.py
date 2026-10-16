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
                "0.541878",
                "--y",
                "0.000304774",
                "--z",
                "0.82384",
                "--qx",
                "-0.151309",
                "--qy",
                "0.987202",
                "--qz",
                "0.044975",
                "--qw",
                "0.0226784",
                # "--roll",
                # "0.0958549",
                # "--pitch",
                # "3.11042",
                # "--yaw",
                # "-0.305668",
            ],
        ),
    ]
    return LaunchDescription(nodes)
