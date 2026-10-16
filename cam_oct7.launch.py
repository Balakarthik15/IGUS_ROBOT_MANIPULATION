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
                "0.46975",
                "--y",
                "0.0111156",
                "--z",
                "0.723209",
                "--qx",
                "-0.0131357",
                "--qy",
                "0.998228",
                "--qz",
                "-0.00625241",
                "--qw",
                "0.0576949",
                # "--roll",
                # "3.13055",
                # "--pitch",
                # "0.115607",
                # "--yaw",
                # "3.11592",
            ],
        ),
    ]
    return LaunchDescription(nodes)
