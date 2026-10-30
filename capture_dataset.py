#!/usr/bin/env python3
"""
Save camera images for the YOLO dataset. Press Enter to save, q to quit.

    python3 capture_dataset.py
    python3 capture_dataset.py --session day1_morning
    python3 capture_dataset.py --topic /camera/camera/color/image_raw --out dataset

Every run makes its own folder:  dataset/<session>/img_0001.png, img_0002.png, ...
plus log.csv with the time of each image.
Images are saved exactly as the camera sends them (full size, no drawing on them).
The live window shows a counter and a box around the board area only on screen.
"""
import argparse
import csv
import os
import threading
import time
from datetime import datetime

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

BOARD_U = (380, 750)   # board area in the full 1280 x 720 image, only drawn for orientation


def to_bgr(msg):
    """sensor_msgs/Image -> OpenCV BGR array, without cv_bridge."""
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    if msg.encoding in ("rgb8", "bgr8"):
        img = buf.reshape(msg.height, msg.step)[:, :msg.width * 3].reshape(msg.height, msg.width, 3)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if msg.encoding == "rgb8" else img.copy()
    if msg.encoding in ("rgba8", "bgra8"):
        img = buf.reshape(msg.height, msg.step)[:, :msg.width * 4].reshape(msg.height, msg.width, 4)
        code = cv2.COLOR_RGBA2BGR if msg.encoding == "rgba8" else cv2.COLOR_BGRA2BGR
        return cv2.cvtColor(img, code)
    raise ValueError("unsupported encoding %s" % msg.encoding)


class Grabber(Node):
    def __init__(self, topic):
        super().__init__("capture_dataset")
        self.lock = threading.Lock()
        self.frame, self.stamp = None, None
        self.create_subscription(Image, topic, self.cb, qos_profile_sensor_data)

    def cb(self, msg):
        try:
            img = to_bgr(msg)
        except ValueError as exc:
            self.get_logger().error(str(exc))
            return
        with self.lock:
            self.frame = img
            self.stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def latest(self):
        with self.lock:
            return (None, None) if self.frame is None else (self.frame.copy(), self.stamp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default="/camera/camera/color/image_raw")
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--session", default=datetime.now().strftime("session_%Y%m%d_%H%M"))
    a = ap.parse_args()

    folder = os.path.join(a.out, a.session)
    os.makedirs(folder, exist_ok=True)
    existing = [f for f in os.listdir(folder) if f.startswith("img_") and f.endswith(".png")]
    count = len(existing)
    log_path = os.path.join(folder, "log.csv")
    new_log = not os.path.exists(log_path)
    log = open(log_path, "a", newline="")
    writer = csv.writer(log)
    if new_log:
        writer.writerow(["file", "saved_at", "image_stamp", "width", "height"])

    rclpy.init()
    node = Grabber(a.topic)
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    print("saving to %s   (Enter = save, q = quit)" % folder)

    flash_until = 0.0
    cv2.namedWindow("capture", cv2.WINDOW_NORMAL)
    while True:
        img, stamp = node.latest()
        if img is None:
            blank = np.zeros((200, 600, 3), np.uint8)
            cv2.putText(blank, "waiting for %s" % a.topic, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.imshow("capture", blank)
        else:
            show = img.copy()
            h = show.shape[0]
            cv2.rectangle(show, (BOARD_U[0], 0), (BOARD_U[1], h - 1), (255, 255, 0), 1)
            txt = "%s   saved: %d   Enter=save  q=quit" % (a.session, count)
            cv2.putText(show, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            if time.time() < flash_until:
                cv2.putText(show, "SAVED", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            cv2.imshow("capture", show)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        if key in (13, 10) and img is not None:      # Enter
            count += 1
            name = "img_%04d.png" % count
            cv2.imwrite(os.path.join(folder, name), img)
            writer.writerow([name, datetime.now().isoformat(timespec="seconds"), "%.3f" % stamp,
                             img.shape[1], img.shape[0]])
            log.flush()
            print("saved %s" % name)
            flash_until = time.time() + 0.4

    log.close()
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()
    print("%d images in %s" % (count, folder))


if __name__ == "__main__":
    main()
