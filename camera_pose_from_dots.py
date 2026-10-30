#!/usr/bin/env python3
"""
Camera pose (base_link -> camera_color_optical_frame) from your 11 screw-tip dots.

Your current calibration is a 2D homography (pixel -> table x, y). It has no 3D camera pose
inside. This script computes one from the same 11 dots with OpenCV solvePnP, using the real
camera intrinsics, so you also get a TF for the depth-based path and for RViz.

Run with the camera driver running (reads camera_info itself):
    python3 camera_pose_from_dots.py
or give the intrinsics by hand (from: ros2 topic echo --once /camera/camera/color/camera_info):
    python3 camera_pose_from_dots.py --k FX FY CX CY

It prints:
  - camera position and orientation in base_link (x y z, quaternion, rpy)
  - reprojection error per dot (pixels) and the position error on the table (mm)
  - a ready static_transform_publisher command
If the RealSense driver already publishes camera_link -> camera_color_optical_frame (it does by
default), a second parent for the optical frame would break the TF tree. In that case the
script (with --tf-lookup) also gives you base_link -> camera_link, which is the one to publish.
"""
import argparse
import math

import cv2
import numpy as np

TIP_L = 0.215       # link6 origin -> screw tip (m), fitted from the dots
HOVER = 0.001       # tip was about 1 mm above the table

# dot: centre pixel u, v | link6 x, y, z | 3rd column of the link6 rotation (tool z axis)
DOTS = [
    (1, 522.0, 401.0, 0.432, 0.021, 0.175, (0.001, -0.001, -1.000)),
    (2, 651.0, 348.0, 0.286, -0.053, 0.178, (0.002, -0.001, -1.000)),
    (3, 710.0, 460.5, 0.207, 0.076, 0.172, (0.003, -0.001, -1.000)),
    (4, 611.0, 533.5, 0.315, 0.169, 0.173, (0.012, 0.001, -1.000)),
    (5, 507.0, 534.5, 0.437, 0.180, 0.176, (0.010, 0.001, -1.000)),
    (6, 462.0, 446.5, 0.496, 0.081, 0.174, (0.015, 0.002, -1.000)),
    (7, 468.0, 302.5, 0.502, -0.091, 0.178, (0.016, 0.002, -1.000)),
    (8, 534.0, 248.5, 0.426, -0.161, 0.172, (0.031, -0.001, -1.000)),
    (9, 621.0, 223.0, 0.324, -0.198, 0.174, (0.034, -0.004, -0.999)),
    (10, 685.0, 185.5, 0.251, -0.248, 0.172, (0.037, -0.006, -0.999)),
    (11, 504.0, 158.5, 0.470, -0.265, 0.174, (0.031, -0.004, -1.000)),
]


def dot_points():
    uv = np.array([[d[1], d[2]] for d in DOTS])
    pts = []
    for d in DOTS:
        t = np.array(d[3:6])
        zc = np.array(d[6])
        tip = t + TIP_L * zc
        pts.append([tip[0], tip[1], tip[2] - HOVER])
    pts = np.array(pts)
    pts[:, 2] = np.median(pts[:, 2])          # the dots lie on one flat table: use one height
    return uv, pts


def mat_to_quat(r):
    t = np.trace(r)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        return np.array([(r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s, 0.25 * s])
    i = int(np.argmax(np.diag(r)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(r[i, i] - r[j, j] - r[k, k] + 1.0) * 2
    q = np.zeros(4)
    q[i] = 0.25 * s
    q[3] = (r[k, j] - r[j, k]) / s
    q[j] = (r[j, i] + r[i, j]) / s
    q[k] = (r[k, i] + r[i, k]) / s
    return q / np.linalg.norm(q)


def quat_to_mat(q):
    x, y, z, w = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def rpy(r):
    return (math.atan2(r[2, 1], r[2, 2]), math.asin(-max(-1, min(1, r[2, 0]))), math.atan2(r[1, 0], r[0, 0]))


def solve(uv, pts, k, dist):
    ok, rvec, tvec = cv2.solvePnP(pts, uv, k, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise RuntimeError("solvePnP failed")
    rvec, tvec = cv2.solvePnPRefineLM(pts, uv, k, dist, rvec, tvec)
    r_cb = cv2.Rodrigues(rvec)[0]                 # base -> camera
    r_bc = r_cb.T                                 # camera orientation in base
    t_bc = (-r_cb.T @ tvec).ravel()               # camera position in base
    proj = cv2.projectPoints(pts, rvec, tvec, k, dist)[0].reshape(-1, 2)
    px_err = np.linalg.norm(proj - uv, axis=1)
    # table error: ray of each measured pixel cut with the table plane, compared to the dot
    und = cv2.undistortPoints(uv.reshape(-1, 1, 2), k, dist).reshape(-1, 2)
    zt = pts[0, 2]
    tab = []
    for (a, b) in und:
        ray = r_bc @ np.array([a, b, 1.0])
        s = (zt - t_bc[2]) / ray[2]
        tab.append(t_bc + s * ray)
    mm_err = np.linalg.norm(np.array(tab)[:, :2] - pts[:, :2], axis=1) * 1000
    return r_bc, t_bc, px_err, mm_err


def get_ros(args):
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import CameraInfo
    rclpy.init()
    node = Node("camera_pose_from_dots")
    got = {}
    node.create_subscription(CameraInfo, args.info_topic, lambda m: got.setdefault("info", m), 10)
    t_link = None
    buf = None
    if args.tf_lookup:
        from tf2_ros import Buffer, TransformListener
        buf = Buffer()
        TransformListener(buf, node)
    import time
    t0 = time.time()
    while "info" not in got and time.time() - t0 < 10:
        rclpy.spin_once(node, timeout_sec=0.2)
    if "info" not in got:
        raise RuntimeError("no camera_info on %s" % args.info_topic)
    if buf is not None:
        from rclpy.time import Time
        t0 = time.time()
        while time.time() - t0 < 5 and t_link is None:
            rclpy.spin_once(node, timeout_sec=0.2)
            try:
                tf = buf.lookup_transform(args.camera_link, args.optical_frame, Time())
                tr, q = tf.transform.translation, tf.transform.rotation
                t_link = (np.array([tr.x, tr.y, tr.z]), quat_to_mat([q.x, q.y, q.z, q.w]))
            except Exception:  # noqa: BLE001
                pass
    node.destroy_node()
    rclpy.shutdown()
    info = got["info"]
    return np.array(info.k).reshape(3, 3), np.array(info.d, float) if len(info.d) else np.zeros(5), \
        (info.width, info.height), t_link


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=float, nargs=4, metavar=("FX", "FY", "CX", "CY"))
    ap.add_argument("--info-topic", default="/camera/camera/color/camera_info")
    ap.add_argument("--tf-lookup", action="store_true",
                    help="also read camera_link -> optical frame from TF and print base_link -> camera_link")
    ap.add_argument("--camera-link", default="camera_link")
    ap.add_argument("--optical-frame", default="camera_color_optical_frame")
    a = ap.parse_args()

    t_link = None
    if a.k:
        fx, fy, cx, cy = a.k
        k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
        dist = np.zeros(5)
        size = None
    else:
        k, dist, size, t_link = get_ros(a)
    if size and tuple(size) != (1280, 720):
        print("WARNING: camera image is %dx%d, the dots were read at 1280x720." % size)

    uv, pts = dot_points()
    r_bc, t_bc, px_err, mm_err = solve(uv, pts, k, dist)
    q = mat_to_quat(r_bc)
    ro, pi, ya = rpy(r_bc)
    print("intrinsics: fx %.1f fy %.1f cx %.1f cy %.1f, dist %s" % (k[0, 0], k[1, 1], k[0, 2], k[1, 2],
                                                                    np.round(dist, 4).tolist()))
    print("table height used (base_link z): %.4f m" % pts[0, 2])
    print("\nbase_link -> %s" % a.optical_frame)
    print("  translation x y z (m)  : %.4f %.4f %.4f" % tuple(t_bc))
    print("  quaternion x y z w     : %.5f %.5f %.5f %.5f" % tuple(q))
    print("  rpy (deg)              : %.2f %.2f %.2f" % tuple(math.degrees(v) for v in (ro, pi, ya)))
    print("  camera height above table: %.3f m" % (t_bc[2] - pts[0, 2]))
    print("\ndot   reprojection (px)   on table (mm)")
    for d, e1, e2 in zip(DOTS, px_err, mm_err):
        print("%3d   %8.2f            %6.2f" % (d[0], e1, e2))
    print("RMS   %8.2f            %6.2f" % (np.sqrt(np.mean(px_err ** 2)), np.sqrt(np.mean(mm_err ** 2))))
    print("\nstatic transform (only if NOTHING else publishes %s):" % a.optical_frame)
    print("  ros2 run tf2_ros static_transform_publisher --x %.4f --y %.4f --z %.4f "
          "--qx %.5f --qy %.5f --qz %.5f --qw %.5f --frame-id base_link --child-frame-id %s"
          % (*t_bc, *q, a.optical_frame))
    if t_link is not None:
        t_lo, r_lo = t_link                         # camera_link -> optical
        r_bl = r_bc @ r_lo.T
        t_bl = t_bc - r_bl @ t_lo
        ql = mat_to_quat(r_bl)
        print("\nbase_link -> %s (publish THIS when the RealSense driver publishes the optical frame):" % a.camera_link)
        print("  ros2 run tf2_ros static_transform_publisher --x %.4f --y %.4f --z %.4f "
              "--qx %.5f --qy %.5f --qz %.5f --qw %.5f --frame-id base_link --child-frame-id %s"
              % (*t_bl, *ql, a.camera_link))
    elif not a.k:
        print("\n(add --tf-lookup to also get base_link -> camera_link)")


if __name__ == "__main__":
    main()
