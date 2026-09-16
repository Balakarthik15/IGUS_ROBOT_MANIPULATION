import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class TestNode(Node):
    def __init__(self):
        super().__init__('test_node')
        self.bridge = CvBridge()
        self.sub = self.create_subscription(
            Image, '/camera/camera/color/image_raw', self.callback, 10)

    def callback(self, msg):
        print("callback fired")
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        cv2.imshow("Test", frame)
        cv2.waitKey(1)

def main():
    rclpy.init()
    node = TestNode()
    rclpy.spin(node)

if __name__ == '__main__':
    main()