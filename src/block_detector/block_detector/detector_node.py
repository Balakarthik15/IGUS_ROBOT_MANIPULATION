"""
Block detector (detection only): YOLO-OBB on the colour image -> pixel centre + image angle.

    ros2 run <your_package> detector_node          (or: python3 detector_node.py)
    ros2 run <your_package> detector_node --ros-args -p show_window:=false -p conf:=0.6

Publishes
  /detector/detections   std_msgs/String, JSON:
      {"stamp": sec, "frame": "...", "width": 1280, "height": 720,
       "blocks": [{"id": 0, "name": "red", "conf": 0.93, "u": 612.4, "v": 401.8,
                   "angle_img": -87.5, "long_px": 54.1, "short_px": 27.3,
                   "corners": [[u, v], [u, v], [u, v], [u, v]]}, ...]}
      angle_img = direction of the LONG side in the image, degrees, from +u (right) towards
      +v (down), in (-90, 90]. id = number shown on the image (sorted left to right).
  /detector/debug_image  the same drawing as the window (for rqt_image_view)

Window (show_window:=true): boxes, id, class, confidence, pixel and angle on the live image.
Keys in the window: s = save a snapshot, p = pause / resume.
No depth, no TF: the conversion to robot coordinates is in homography_calib.py.
"""
import json
import math
import os
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from ultralytics import YOLO

DRAW_BGR = {'red': (40, 40, 230), 'green': (40, 170, 40), 'yellow': (0, 200, 240),
            'white': (240, 240, 240), 'lightblue': (230, 200, 90)}


def long_side_angle(corners):
    """corners (4, 2) -> angle of the long side in degrees, (-90, 90], plus long and short length"""
    e1 = corners[1] - corners[0]
    e2 = corners[2] - corners[1]
    l1, l2 = np.linalg.norm(e1), np.linalg.norm(e2)
    e = e1 if l1 >= l2 else e2
    ang = math.degrees(math.atan2(e[1], e[0]))
    while ang > 90:
        ang -= 180
    while ang <= -90:
        ang += 180
    return ang, max(l1, l2), min(l1, l2)


class DetectorNode(Node):
    def __init__(self):
        super().__init__('block_detector')
        self.declare_parameter('model_path', os.path.expanduser('~/IGUS_ROBOT_MANIPULATION/models/best.pt'))
        self.declare_parameter('conf', 0.5)
        self.declare_parameter('imgsz', 1280)
        self.declare_parameter('rate', 2.0)
        self.declare_parameter('image_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('show_window', True)
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.conf, self.imgsz, self.show = g('conf'), g('imgsz'), g('show_window')

        self.model = YOLO(g('model_path'))
        self.get_logger().info(f"model loaded: {g('model_path')}, classes {self.model.names}")

        self.bridge = CvBridge()
        self.latest_image = None
        self.last_view = None
        self.paused = False
        self.warned_size = False

        self.create_subscription(Image, g('image_topic'), self.image_callback, qos_profile_sensor_data)
        self.det_pub = self.create_publisher(String, '/detector/detections', 10)
        self.debug_pub = self.create_publisher(Image, '/detector/debug_image', 1)
        self.create_timer(1.0 / g('rate'), self.detect)
        if self.show:
            cv2.namedWindow('detections', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('detections', 1280, 720)
            self.create_timer(0.05, self.gui)            # keeps the window responsive

    def image_callback(self, msg):
        self.latest_image = msg

    # ---------------------------------------------------------------- window
    def gui(self):
        if self.last_view is not None:
            view = self.last_view
            if self.paused:
                view = view.copy()
                cv2.putText(view, 'PAUSED (p)', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.imshow('detections', view)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('p'):
            self.paused = not self.paused
        elif k == ord('s') and self.last_view is not None:
            name = time.strftime('detections_%Y%m%d_%H%M%S.png')
            cv2.imwrite(name, self.last_view)
            self.get_logger().info(f'saved {name}')

    # ------------------------------------------------------------- detection
    def detect(self):
        if self.latest_image is None or self.paused:
            return
        msg = self.latest_image
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        h, w = image.shape[:2]
        if (w, h) != (1280, 720) and not self.warned_size:
            self.get_logger().error(f'image is {w}x{h}, the calibration was made at 1280x720!')
            self.warned_size = True

        result = self.model.predict(image, imgsz=self.imgsz, conf=self.conf, verbose=False)[0]
        obb = result.obb
        blocks = []
        if obb is not None:
            for cls_id, conf, box, corners in zip(obb.cls.cpu().numpy(), obb.conf.cpu().numpy(),
                                                  obb.xywhr.cpu().numpy(), obb.xyxyxyxy.cpu().numpy()):
                ang, lo, sh = long_side_angle(corners)
                blocks.append({'name': result.names[int(cls_id)], 'conf': round(float(conf), 3),
                               'u': round(float(box[0]), 1), 'v': round(float(box[1]), 1),
                               'angle_img': round(ang, 1), 'long_px': round(float(lo), 1),
                               'short_px': round(float(sh), 1),
                               'corners': [[round(float(a), 1), round(float(b), 1)] for a, b in corners]})
        blocks.sort(key=lambda b: b['u'])
        for i, b in enumerate(blocks):
            b['id'] = i

        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        out = String()
        out.data = json.dumps({'stamp': stamp, 'frame': msg.header.frame_id, 'width': w, 'height': h,
                               'blocks': blocks})
        self.det_pub.publish(out)

        # ---- drawing
        view = image.copy()
        for b in blocks:
            col = DRAW_BGR.get(b['name'], (255, 0, 255))
            pts = np.array(b['corners'], np.int32)
            cv2.polylines(view, [pts], True, (0, 0, 0), 4)
            cv2.polylines(view, [pts], True, col, 2)
            c = (int(round(b['u'])), int(round(b['v'])))
            cv2.circle(view, c, 3, (0, 0, 255), -1)
            a = math.radians(b['angle_img'])
            tip = (int(c[0] + 25 * math.cos(a)), int(c[1] + 25 * math.sin(a)))
            cv2.line(view, c, tip, (0, 0, 255), 1)                       # long-side direction
            x0, y0 = int(pts[:, 0].max()) + 6, int(pts[:, 1].min()) + 4
            for k, t in enumerate([f"#{b['id']} {b['name']} {b['conf']:.2f}",
                                   f"px {b['u']:.1f}, {b['v']:.1f}",
                                   f"angle {b['angle_img']:+.1f} deg"]):
                cv2.putText(view, t, (x0, y0 + 16 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
                cv2.putText(view, t, (x0, y0 + 16 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
        cv2.putText(view, f'{len(blocks)} blocks   s = snapshot, p = pause', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        self.last_view = view
        dbg = self.bridge.cv2_to_imgmsg(view, encoding='bgr8')
        dbg.header = msg.header
        self.debug_pub.publish(dbg)
        self.get_logger().info(f'{len(blocks)} blocks: ' + ', '.join(
            f"#{b['id']} {b['name']} {b['conf']:.2f} ({b['u']:.0f},{b['v']:.0f}) {b['angle_img']:+.0f}deg"
            for b in blocks), throttle_duration_sec=2.0)


def main():
    rclpy.init()
    node = DetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
