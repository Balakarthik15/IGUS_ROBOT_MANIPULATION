#!/usr/bin/env python3
"""Digital open/close for the simulated gripper, like the real digital output.

Subscribes to /gripper/close (std_msgs/Bool):
  true  -> both fingers close (CLOSE_POS)
  false -> both fingers open  (0.0)
and sends the targets to the Gazebo finger controllers.

Run with:  python3 sim_gripper_io.py --ros-args -p use_sim_time:=true
Test with: ros2 topic pub --once -w 1 /gripper/close std_msgs/msg/Bool "{data: true}"
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float64

OPEN_POS = 0.0
CLOSE_POS = 0.04   # 31.8 mm block is touched at about 0.034, so this squeezes it


class SimGripperIO(Node):
    def __init__(self):
        super().__init__("sim_gripper_io")
        self.left = self.create_publisher(Float64, "/gripper/left/cmd_pos", 10)
        self.right = self.create_publisher(Float64, "/gripper/right/cmd_pos", 10)
        self.create_subscription(Bool, "/gripper/close", self.on_cmd, 10)
        self.target = OPEN_POS
        # resend the target regularly so a lost message or a late bridge does not matter
        self.create_timer(0.1, self.send)
        self.get_logger().info("sim gripper ready, publish true/false on /gripper/close")

    def on_cmd(self, msg):
        self.target = CLOSE_POS if msg.data else OPEN_POS
        self.get_logger().info("close" if msg.data else "open")
        self.send()

    def send(self):
        m = Float64()
        m.data = self.target
        self.left.publish(m)
        self.right.publish(m)


def main():
    rclpy.init()
    node = SimGripperIO()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
