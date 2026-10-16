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
                "0.53686",
                "--y",
                "-0.0639207",
                "--z",
                "0.284045",
                "--qx",
                "0.414049",
                "--qy",
                "-0.0225345",
                "--qz",
                "-0.0155845",
                "--qw",
                "0.909842",
                # "--roll",
                # "0.853876",
                # "--pitch",
                # "-0.0539373",
                # "--yaw",
                # "-0.00971239",
            ],
        ),
    ]
    return LaunchDescription(nodes)
