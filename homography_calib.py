#!/usr/bin/env python3
"""
Homography calibration: camera pixel -> robot base_link x, y. One place for all values.

Other scripts import it:
    from homography_calib import pixel_to_base, image_angle_to_base_yaw, BLOCK_TOP_H

Command line:
    python3 homography_calib.py query U V            pixel -> x, y (block top and table level)
    python3 homography_calib.py query U V --h 0      for a point on the table itself
    python3 homography_calib.py fit points.csv       new calibration from dots
         CSV columns: dot,u,v,x,y   (u v = dot centre pixel, x y = tip position in base_link, m)
         prints the error per dot and the new H to paste below
    python3 homography_calib.py --selftest

Valid only while: camera not moved, image 1280 x 720, objects lying flat on the table.
"""
import argparse
import csv
import math
import sys

import numpy as np

# ---------------- current calibration (screw tip, 28 Oct 2026, 11 dots, about 1 mm)
H = np.array([
    [-0.0011955698109245357, -7.365592435330288e-05, 1.0954614426821216],
    [-9.03704614697729e-05, 0.0012075322247273575, -0.41498965262663823],
    [9.684829331241521e-06, 3.8082753139176714e-05, 1.0],
])
IMAGE_SIZE = (1280, 720)
NADIR_XY = np.array([0.294, -0.0185])   # table point straight below the camera (camera_pose_from_dots.py)
CAM_HEIGHT = 0.762                      # camera above the table (m)
TABLE_Z = -0.042                        # table top in base_link (m)

# ---------------- scene
BOARD = 0.003                           # brown board under the blocks
BLOCK_H = 0.019
BLOCK_TOP_H = BOARD + BLOCK_H           # block top above the table
CAL_AREA = ((0.20, 0.51), (-0.27, 0.19))  # x range, y range covered by the dots (m)


def pixel_to_table(u, v):
    """pixel -> x, y of the table point seen at that pixel"""
    p = H @ np.array([u, v, 1.0])
    return p[:2] / p[2]


def pixel_to_base(u, v, h=BLOCK_TOP_H):
    """pixel -> x, y in base_link of a point h metres above the table (parallax corrected).
    With the gripper vertical, this is also the link6 x, y to put the gripper centre there."""
    xy = pixel_to_table(u, v)
    return NADIR_XY + (xy - NADIR_XY) * (1.0 - h / CAM_HEIGHT)


def image_angle_to_base_yaw(u, v, angle_img_deg, h=BLOCK_TOP_H, step_px=20.0):
    """direction in the image (deg, from +u towards +v) at pixel (u, v) -> direction in base_link (rad)"""
    a = math.radians(angle_img_deg)
    p0 = pixel_to_base(u, v, h)
    p1 = pixel_to_base(u + step_px * math.cos(a), v + step_px * math.sin(a), h)
    return math.atan2(p1[1] - p0[1], p1[0] - p0[0])


def fold90(rad):
    """a line has no direction: bring an angle into (-90, 90] deg"""
    while rad > math.pi / 2:
        rad -= math.pi
    while rad <= -math.pi / 2:
        rad += math.pi
    return rad


def in_area(x, y, margin=0.02):
    (x0, x1), (y0, y1) = CAL_AREA
    return x0 - margin <= x <= x1 + margin and y0 - margin <= y <= y1 + margin


# ---------------- fitting a new calibration
def fit_homography(pix, xy):
    pix, xy = np.asarray(pix, float), np.asarray(xy, float)

    def norm(p):
        c = p.mean(axis=0)
        s = math.sqrt(2.0) / np.mean(np.linalg.norm(p - c, axis=1))
        return np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1.0]])
    tp, tq = norm(pix), norm(xy)
    p = (tp @ np.c_[pix, np.ones(len(pix))].T).T
    q = (tq @ np.c_[xy, np.ones(len(xy))].T).T
    rows = []
    for (x, y, _), (a, b, _) in zip(p, q):
        rows.append([-x, -y, -1, 0, 0, 0, a * x, a * y, a])
        rows.append([0, 0, 0, -x, -y, -1, b * x, b * y, b])
    hn = np.linalg.svd(np.array(rows))[2][-1].reshape(3, 3)
    h = np.linalg.inv(tq) @ hn @ tp
    return h / h[2, 2]


def apply(h, u, v):
    p = h @ np.array([u, v, 1.0])
    return p[:2] / p[2]


def run_fit(path):
    ids, pix, xy = [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            ids.append(row["dot"])
            pix.append([float(row["u"]), float(row["v"])])
            xy.append([float(row["x"]), float(row["y"])])
    pix, xy = np.array(pix), np.array(xy)
    if len(ids) < 6:
        print("need at least 6 dots")
        return
    h = fit_homography(pix, xy)
    err = np.linalg.norm(np.array([apply(h, *p) for p in pix]) - xy, axis=1) * 1000
    loo = [np.linalg.norm(apply(fit_homography(np.delete(pix, k, 0), np.delete(xy, k, 0)), *pix[k]) - xy[k]) * 1000
           for k in range(len(ids))]
    print("dot   fit error   leave-one-out (mm)")
    for i, e, l in zip(ids, err, loo):
        print("%4s   %6.2f      %6.2f" % (i, e, l))
    print("RMS   %6.2f      %6.2f" % (np.sqrt(np.mean(err ** 2)), np.sqrt(np.mean(np.square(loo)))))
    print("\npaste into homography_calib.py:\nH = np.array([")
    for r in h:
        print("    [%r, %r, %r]," % tuple(float(x) for x in r))
    print("])")


def selftest():
    # the 11 calibration dots must come back within a few mm (pixels, tip x y from 28 Oct)
    dots = [(522, 401, .4322, .0208), (651, 348, .2864, -.0532), (710, 460.5, .2076, .0758),
            (611, 533.5, .3176, .1692), (507, 534.5, .4392, .1802), (462, 446.5, .4992, .0814),
            (468, 302.5, .5054, -.0906), (534, 248.5, .4327, -.1612), (621, 223, .3313, -.1989),
            (685, 185.5, .2590, -.2493), (504, 158.5, .4767, -.2659)]
    e = [np.linalg.norm(pixel_to_table(u, v) - [x, y]) * 1000 for u, v, x, y in dots]
    print("calibration dots: mean %.2f mm, max %.2f mm" % (np.mean(e), np.max(e)))
    yaw = image_angle_to_base_yaw(640, 360, 90.0)
    print("image 'down' (+v) is base direction %.1f deg (expected about +90)" % math.degrees(yaw))
    ok = np.max(e) < 2.5 and abs(math.degrees(yaw) - 90) < 10
    print("SELFTEST", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    q = sub.add_parser("query")
    q.add_argument("u", type=float)
    q.add_argument("v", type=float)
    q.add_argument("--h", type=float, default=BLOCK_TOP_H)
    f = sub.add_parser("fit")
    f.add_argument("csv")
    a = ap.parse_args()
    if a.cmd == "query":
        x, y = pixel_to_base(a.u, a.v, a.h)
        tx, ty = pixel_to_table(a.u, a.v)
        print("pixel %.1f %.1f" % (a.u, a.v))
        print("at height %.3f m: x %.4f  y %.4f  (= link6 x y for a vertical gripper)" % (a.h, x, y))
        print("on the table    : x %.4f  y %.4f" % (tx, ty))
        if not in_area(x, y):
            print("WARNING: outside the calibrated area")
    elif a.cmd == "fit":
        run_fit(a.csv)
    else:
        ap.print_help()
