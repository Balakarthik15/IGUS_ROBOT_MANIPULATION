#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from ros2_aruco_interfaces.msg import ArucoMarkers
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped

class ArucoTfBridge(Node):
    def __init__(self):
        super().__init__('aruco_tf_bridge')
        self.br = TransformBroadcaster(self)
        self.sub = self.create_subscription(
            ArucoMarkers, '/aruco_markers', self.callback, 10)

    def callback(self, msg):
        if len(msg.marker_ids) == 0:
            return  # no marker detected this frame — nothing to broadcast

        t = TransformStamped()
        t.header.stamp = msg.header.stamp
        t.header.frame_id = msg.header.frame_id      # camera_color_optical_frame
        t.child_frame_id = 'aruco_marker_frame'
        t.transform.translation.x = msg.poses[0].position.x
        t.transform.translation.y = msg.poses[0].position.y
        t.transform.translation.z = msg.poses[0].position.z
        t.transform.rotation = msg.poses[0].orientation
        self.br.sendTransform(t)

def main():
    rclpy.init()
    node = ArucoTfBridge()
    rclpy.spin(node)

if __name__ == '__main__':
    main()