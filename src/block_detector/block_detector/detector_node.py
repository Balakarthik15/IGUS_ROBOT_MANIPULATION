

import math
import os

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from ultralytics import YOLO

from rclpy.time import Time
from geometry_msgs.msg import PointStamped
from tf2_ros import Buffer, TransformListener, TransformException
from tf2_geometry_msgs import do_transform_point

from std_msgs.msg import String

from geometry_msgs.msg import PointStamped, PoseStamped
from visualization_msgs.msg import Marker

MARKER_RGB = {'red': (0.9, 0.1, 0.1), 'blue': (0.1, 0.3, 0.9),
              'yellow': (0.95, 0.85, 0.1), 'green': (0.1, 0.7, 0.2)}

class DetectorNode(Node):
    def __init__(self):
        super().__init__('block_detector')

        self.declare_parameter('model_path', os.path.expanduser('~/IGUS_ROBOT_MANIPULATION/YOLO_Detection/Result_runs/yolo11n_obb_sim/weights/best.pt'))
        self.declare_parameter('conf', 0.5)
        self.declare_parameter('imgsz', 1280)
        self.declare_parameter('rate', 1.0)


        model_path= self.get_parameter('model_path').value
        self.conf = self.get_parameter('conf').value
        self.imgsz = self.get_parameter('imgsz').value
        rate = self.get_parameter('rate').value


        self.model = YOLO(model_path)
        self.get_logger().info(f'model loaded: {model_path}, classes {self.model.names}')


        self.bridge = CvBridge()
        self.latest_image = None
        self.latest_depth = None
        self.K = None


        self.image_sub = self.create_subscription(
            Image, '/camera/camera/color/image_raw',
            self.image_callback,
            qos_profile_sensor_data
        )
        
        self.depth_sub = self.create_subscription(
            Image, '/camera/camera/aligned_depth_to_color/image_raw',
            self.depth_callback, qos_profile_sensor_data)

        self.info_sub = self.create_subscription(
            CameraInfo, '/camera/camera/color/camera_info',
            self.info_callback, qos_profile_sensor_data)

        self.timer = self.create_timer(1.0 / rate, self.detect)

        #tf
        self.declare_parameter('target_frame', 'base_link')
        self.target_frame = self.get_parameter('target_frame').value

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.declare_parameter('block_height', 0.019)
        self.block_height = self.get_parameter('block_height').value

        self.declare_parameter('max_time_diff', 0.5)
        self.max_time_diff = self.get_parameter('max_time_diff').value

        #target color
        self.target_color = None
        self.target_sub = self.create_subscription(
            String, '/detector/target_color', self.target_callback, 10)
        self.status_pub = self.create_publisher(String, '/detector/status', 10)

        #marker

        self.declare_parameter('block_length', 0.0638)
        self.declare_parameter('block_width', 0.0318)
        self.block_length = self.get_parameter('block_length').value
        self.block_width = self.get_parameter('block_width').value

        self.pose_pub = self.create_publisher(PoseStamped, '/detector/target_pose', 10)
        self.marker_pub = self.create_publisher(Marker, '/detector/target_marker', 10)

    def image_callback(self, msg):
        
        self.latest_image = msg

    def depth_callback(self, msg):
        self.latest_depth = msg

    def info_callback(self, msg):
        if self.K is None:
            self.K = np.array(msg.k).reshape(3, 3)
            self.get_logger().info(
                f'camera_info: fx {self.K[0, 0]:.1f}, fy {self.K[1, 1]:.1f}, '
                f'cx {self.K[0, 2]:.1f}, cy {self.K[1, 2]:.1f}')


    def depth_at(self, depth, u, v, half=3):
            """Median depth in metres in a small square around pixel (u, v). None if no valid pixel."""
            h, w = depth.shape
            u, v = int(round(u)), int(round(v))
            patch = depth[max(v - half, 0):min(v + half + 1, h),
                        max(u - half, 0):min(u + half + 1, w)].astype(np.float32)
            if depth.dtype == np.uint16:
                patch = patch / 1000.0
            valid = patch[np.isfinite(patch) & (patch > 0)]
            if valid.size == 0:
                return None
            return float(np.median(valid))

    def pixel_to_point(self, u, v, z):
        """Pixel (u, v) with depth z (m) -> point (X, Y, Z) in the camera optical frame (m)."""
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        return x, y, z

    def to_base(self, u, v, d, tf, camera_frame):
        """Pixel (u, v) with depth d -> point (x, y, z) in the target frame."""
        X, Y, Z = self.pixel_to_point(u, v, d)
        p = PointStamped()
        p.header.frame_id = camera_frame
        p.point.x, p.point.y, p.point.z = float(X), float(Y), float(Z)
        q = do_transform_point(p, tf).point
        return q.x, q.y, q.z


    def target_callback(self, msg):
        color = msg.data.strip().lower()
        known = list(self.model.names.values())
        if color not in known:
            self.get_logger().warning(f'unknown colour "{msg.data}", known: {known}')
            return
        self.target_color = color
        self.get_logger().info(f'target colour set to {color}')


    def publish_target(self, det):
        pose = PoseStamped()
        pose.header.frame_id = self.target_frame
        pose.header.stamp = self.latest_image.header.stamp
        pose.pose.position.x = det['x']
        pose.pose.position.y = det['y']
        pose.pose.position.z = det['z']
        pose.pose.orientation.z = math.sin(det['yaw'] / 2.0)
        pose.pose.orientation.w = math.cos(det['yaw'] / 2.0)
        self.pose_pub.publish(pose)

        m = Marker()
        m.header = pose.header
        m.ns = 'target_block'
        m.id = 0
        m.type = Marker.CUBE
        m.action = Marker.ADD
        m.pose = pose.pose
        m.scale.x = self.block_length
        m.scale.y = self.block_width
        m.scale.z = self.block_height
        r, g, b = MARKER_RGB[det['name']]
        m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, 0.8
        self.marker_pub.publish(m)

    def clear_marker(self):
        m = Marker()
        m.header.frame_id = self.target_frame
        m.ns = 'target_block'
        m.id = 0
        m.action = Marker.DELETE
        self.marker_pub.publish(m)


    def detect(self):
        if self.latest_image is None or self.latest_depth is None or self.K is None:
            return

        image = self.bridge.imgmsg_to_cv2(self.latest_image, desired_encoding='bgr8')
        depth = self.bridge.imgmsg_to_cv2(self.latest_depth, desired_encoding='passthrough')

        t_img = self.latest_image.header.stamp.sec + self.latest_image.header.stamp.nanosec * 1e-9
        t_dep = self.latest_depth.header.stamp.sec + self.latest_depth.header.stamp.nanosec * 1e-9

        if abs(t_img - t_dep) > self.max_time_diff:
            self.get_logger().warning(
                f'colour and depth {abs(t_img - t_dep) * 1000:.0f} ms apart, skipping')
            return

        result = self.model.predict(image, imgsz=self.imgsz, conf=self.conf, verbose=False)[0]
        obb = result.obb

        lines = []
        detections = []

        camera_frame = self.latest_image.header.frame_id
        try:
            tf = self.tf_buffer.lookup_transform(self.target_frame, camera_frame, Time())

        except TransformException as e:
            self.get_logger().warning(f'no transform {camera_frame} -> {self.target_frame}: {e}')
            return

        


        for cls_id, conf, box, corners in zip(obb.cls.cpu().numpy(),
                                              obb.conf.cpu().numpy(),
                                              obb.xywhr.cpu().numpy(),
                                              obb.xyxyxyxy.cpu().numpy()):
            cx, cy = box[0], box[1]
            name = result.names[int(cls_id)]
            d = self.depth_at(depth, cx, cy)
            if d is None:
                lines.append(f'{name} {conf:.2f} at ({cx:.0f}, {cy:.0f}) px, no depth')
                continue

            # centre of the top face -> grasp point in the middle of the block
            x, y, z_top = self.to_base(cx, cy, d, tf, camera_frame)
            z_grasp = z_top - self.block_height / 2.0

            # long edge of the box in the image
            e1 = np.linalg.norm(corners[1] - corners[0])
            e2 = np.linalg.norm(corners[2] - corners[1])
            a, b = (corners[0], corners[1]) if e1 >= e2 else (corners[1], corners[2])

            # both corners in base_link -> direction of the long side
            ax, ay, _ = self.to_base(a[0], a[1], d, tf, camera_frame)
            bx, by, _ = self.to_base(b[0], b[1], d, tf, camera_frame)
            yaw = math.atan2(by - ay, bx - ax)
            if yaw > math.pi / 2:
                yaw -= math.pi
            elif yaw <= -math.pi / 2:
                yaw += math.pi
            long_side = math.hypot(bx - ax, by - ay)

        
            detections.append({'name': name, 'conf': float(conf),
                               'x': x, 'y': y, 'z': z_grasp, 'yaw': yaw})


        self.get_logger().info(f'{len(detections)} blocks: '
                               + ', '.join(f"{d['name']} {d['conf']:.2f}" for d in detections))

        if self.target_color is None:
            return

        matches = [d for d in detections if d['name'] == self.target_color]
        status = String()

        if not matches:
            status.data = 'not_found'
            self.status_pub.publish(status)
            self.clear_marker()
            self.get_logger().info(f'target {self.target_color}: not found')
            return

        if len(matches) > 1:
            status.data = 'multiple'
            self.get_logger().warning(
                f'{len(matches)} {self.target_color} blocks found, using the most confident one')
        else:
            status.data = 'found'

        best = max(matches, key=lambda d: d['conf'])
        self.publish_target(best)
        self.status_pub.publish(status)
        self.get_logger().info(
            f"target {best['name']} ({status.data}): grasp ({best['x']:+.3f}, {best['y']:+.3f}, "
            f"{best['z']:+.4f}) m, yaw {math.degrees(best['yaw']):+.0f} deg")



def main():
    rclpy.init()
    node = DetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

