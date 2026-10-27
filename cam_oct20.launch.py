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
                "0.151651",
                "--y",
                "-0.0103138",
                "--z",
                "0.679775",
                "--qx",
                "0.0106101",
                "--qy",
                "0.97301",
                "--qz",
                "0.0825868",
                "--qw",
                "0.215215",
                # "--roll",
                # "0.172971",
                # "--pitch",
                # "2.70752",
                # "--yaw",
                # "-0.0164243",
            ],
        ),
    ]
    return LaunchDescription(nodes)
