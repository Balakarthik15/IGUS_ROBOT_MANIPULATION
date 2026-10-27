#!/usr/bin/env python3
"""Sim dataset generator for YOLO OBB block detection (igus ReBeL thesis setup).

Every scene has exactly 4 blocks in the pick area:
    k main blocks (studded 2x4, one per colour, k = 1..4)
  + 4-k distractors (2x4 without studs, 2x2 or 2x6, any colour)
Only the main blocks get a label line. Distractors stay unlabelled, so YOLO
learns to treat them as background.

Classes:  0 red, 1 blue, 2 yellow, 3 green

How it works per scene:
  1. pick the blocks, sample non-overlapping poses inside the pick area that
     the camera can fully see
  2. move them with ONE gz set_pose_vector call (unused blocks go to parking)
  3. optionally randomize the sun light
  4. wait until the blocks settle, take the next camera image
  5. project the top corners of each main block into the image -> OBB label

Run (sim running, arm in a pose that does not cover the pick area):
  python3 gen_sim_dataset.py --ros-args -p use_sim_time:=true -p num_scenes:=1500 -p out_dir:=$HOME/yolo_blocks

Offline geometry check (no ROS needed):
  python3 gen_sim_dataset.py --selftest
"""
import math
import os
import random
import subprocess
import sys

import numpy as np

# ------------------------------------------------------------------ setup --
WORLD_NAME = "rebel_lab_world"
SPAWN_XYZ = (0.0, 0.0, 0.80)          # create node: robot spawned at z = 0.80
TABLE_TOP_WORLD_Z = 0.75               # lab_table top in the Gazebo world
TABLE_TOP_Z = TABLE_TOP_WORLD_Z - SPAWN_XYZ[2]   # in base_link = -0.05

# Pick area in base_link (metres): the lab table strip next to the robot
# base plate, where the 4 main blocks stand in the world file.
# Table x -0.6..0.6, plate x -0.35..0.35, so the strip is x 0.35..0.6.
# Positions the camera cannot fully see are rejected automatically.
PICK_X = (0.36, 0.59)
PICK_Y = (-0.39, 0.39)
MIN_GAP = 0.008                         # free space between blocks (m)
IMG_MARGIN_PX = 8                       # block must be this far inside the image

# Block geometry (must match make_blocks.py)
PITCH = 0.01595
BODY_H = 0.019
WIDTH = 2 * PITCH
LENGTH = {"2x4": 4 * PITCH, "2x4_nostud": 4 * PITCH, "2x2": 2 * PITCH, "2x6": 6 * PITCH}

CLASSES = ["red", "blue", "yellow", "green"]          # class id = index
COLORS = CLASSES
DISTRACTOR_TYPES = ["2x4_nostud", "2x2", "2x6"]
SAME_COLOUR_PROB = 0.5    # chance a distractor copies the colour of a main block in the scene

# Gazebo entity names from rebel_world.sdf
def main_name(color):
    return f"block_{color}"

def distractor_name(dtype, color):
    return f"distractor_{dtype}_{color}"

# Parking spots (world frame), far from the table and out of camera view
def parking_pose(name):
    all_names = [main_name(c) for c in COLORS] + \
                [distractor_name(t, c) for t in DISTRACTOR_TYPES for c in COLORS]
    i = all_names.index(name)
    return (3.0 + 0.3 * (i // 4), -0.45 + 0.3 * (i % 4), 0.0, 0.0)

# Camera topics (sim bridge uses the real D455 names)
IMAGE_TOPIC = "/camera/camera/color/image_raw"
INFO_TOPIC = "/camera/camera/color/camera_info"
CAMERA_FRAME = "camera_color_optical_frame"
BASE_FRAME = "base_link"
SETTLE_S = 0.4            # sim seconds to wait after moving the blocks


# --------------------------------------------------------------- geometry --
def block_corners(x, y, yaw, length, z):
    """4 corners (base frame) of the block's top face, counter-clockwise."""
    c, s = math.cos(yaw), math.sin(yaw)
    hl, hw = length / 2.0, WIDTH / 2.0
    pts = []
    for dx, dy in ((hl, hw), (-hl, hw), (-hl, -hw), (hl, -hw)):
        pts.append((x + c * dx - s * dy, y + s * dx + c * dy, z))
    return np.array(pts)


def rects_overlap(a, b, gap):
    """Separating axis test for two rectangles given as 4x2 corner arrays, with a gap."""
    for poly in (a, b):
        for i in range(4):
            p1, p2 = poly[i], poly[(i + 1) % 4]
            axis = np.array([-(p2[1] - p1[1]), p2[0] - p1[0]])
            axis /= np.linalg.norm(axis)
            pa, pb = a @ axis, b @ axis
            if pa.max() + gap < pb.min() or pb.max() + gap < pa.min():
                return False
    return True


def project(points_base, R, t, K):
    """base-frame points -> pixel coords using camera pose (R, t: base -> optical) and K."""
    p = (R @ points_base.T).T + t
    uv = np.empty((len(p), 2))
    uv[:, 0] = K[0, 0] * p[:, 0] / p[:, 2] + K[0, 2]
    uv[:, 1] = K[1, 1] * p[:, 1] / p[:, 2] + K[1, 2]
    return uv, p[:, 2]


def quat_to_R(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


# ---------------------------------------------------------- scene sampling --
def sample_blocks(rng):
    """Choose which 4 blocks are in the scene. Returns list of (name, type, colour, is_main)."""
    k = rng.randint(1, 4)
    mains = rng.sample(COLORS, k)
    blocks = [(main_name(c), "2x4", c, True) for c in mains]
    used = set()
    while len(blocks) < 4:
        dtype = rng.choice(DISTRACTOR_TYPES)
        if rng.random() < SAME_COLOUR_PROB:
            color = rng.choice(mains)
        else:
            color = rng.choice(COLORS)
        if (dtype, color) in used:
            continue
        used.add((dtype, color))
        blocks.append((distractor_name(dtype, color), dtype, color, False))
    return blocks


def sample_poses(rng, blocks, R, t, K, width, height, tries=400):
    """Random non-overlapping, fully visible poses. Returns list of (x, y, yaw) or None."""
    for _ in range(tries):
        placed, rects = [], []
        ok = True
        for _name, btype, _c, _main in blocks:
            L = LENGTH[btype]
            good = False
            for _ in range(60):
                x = rng.uniform(*PICK_X)
                y = rng.uniform(*PICK_Y)
                yaw = rng.uniform(0.0, math.pi)
                top = block_corners(x, y, yaw, L, TABLE_TOP_Z + BODY_H)
                foot = top[:, :2]
                if foot[:, 0].min() < PICK_X[0] or foot[:, 0].max() > PICK_X[1]:
                    continue
                if foot[:, 1].min() < PICK_Y[0] or foot[:, 1].max() > PICK_Y[1]:
                    continue
                uv, z = project(top, R, t, K)
                if (z <= 0).any():
                    continue
                m = IMG_MARGIN_PX
                if uv[:, 0].min() < m or uv[:, 0].max() > width - m or \
                   uv[:, 1].min() < m or uv[:, 1].max() > height - m:
                    continue
                if any(rects_overlap(foot, r, MIN_GAP) for r in rects):
                    continue
                placed.append((x, y, yaw))
                rects.append(foot)
                good = True
                break
            if not good:
                ok = False
                break
        if ok:
            return placed
    return None


def label_lines(blocks, poses, R, t, K, width, height):
    lines = []
    for (name, btype, color, is_main), (x, y, yaw) in zip(blocks, poses):
        if not is_main:
            continue
        top = block_corners(x, y, yaw, LENGTH[btype], TABLE_TOP_Z + BODY_H)
        uv, _ = project(top, R, t, K)
        coords = []
        for u, v in uv:
            coords += [min(max(u / width, 0.0), 1.0), min(max(v / height, 0.0), 1.0)]
        lines.append(f"{CLASSES.index(color)} " + " ".join(f"{c:.6f}" for c in coords))
    return lines


# --------------------------------------------------------------- gz calls --
def pose_text(name, x, y, z, yaw):
    return (f'pose {{ name: "{name}" position {{ x: {x:.5f} y: {y:.5f} z: {z:.5f} }} '
            f'orientation {{ x: 0 y: 0 z: {math.sin(yaw / 2):.6f} w: {math.cos(yaw / 2):.6f} }} }}')


LAST_GZ_OUTPUT = ""


def gz_service(service, reqtype, req, timeout_ms=5000, attempts=3):
    """Call a gz service, retry if Gazebo is slow. Returns True when Gazebo answers data: true."""
    global LAST_GZ_OUTPUT
    import time
    cmd = ["gz", "service", "-s", service, "--reqtype", reqtype,
           "--reptype", "gz.msgs.Boolean", "--timeout", str(timeout_ms), "--req", req]
    for i in range(attempts):
        r = subprocess.run(cmd, capture_output=True, text=True)
        LAST_GZ_OUTPUT = (r.stdout + r.stderr).strip()
        if r.returncode == 0 and "data: true" in r.stdout:
            return True
        time.sleep(1.0)
    return False


def move_blocks(blocks, poses):
    """Put scene blocks on the table, park all others. One set_pose_vector call."""
    on_table = {b[0]: p for b, p in zip(blocks, poses)}
    all_names = [main_name(c) for c in COLORS] + \
                [distractor_name(t, c) for t in DISTRACTOR_TYPES for c in COLORS]
    parts = []
    for name in all_names:
        if name in on_table:
            x, y, yaw = on_table[name]
            wx, wy = x + SPAWN_XYZ[0], y + SPAWN_XYZ[1]
            parts.append(pose_text(name, wx, wy, TABLE_TOP_WORLD_Z + 0.0005, yaw))
        else:
            px, py, pz, pyaw = parking_pose(name)
            parts.append(pose_text(name, px, py, pz, pyaw))
    return gz_service(f"/world/{WORLD_NAME}/set_pose_vector", "gz.msgs.Pose_V", " ".join(parts))


def randomize_light(rng):
    # every field must be set: unset fields in gz.msgs.Light are 0, and intensity 0 turns the sun off
    r = rng.uniform(0.7, 1.0)
    intensity = rng.uniform(0.8, 1.2)
    dx, dy = rng.uniform(-0.6, 0.6), rng.uniform(-0.6, 0.6)
    req = (f'name: "sun" type: DIRECTIONAL cast_shadows: true intensity: {intensity:.3f} '
           f'diffuse {{ r: {r:.3f} g: {r:.3f} b: {r:.3f} a: 1 }} '
           f'specular {{ r: 0.2 g: 0.2 b: 0.2 a: 1 }} '
           f'direction {{ x: {dx:.3f} y: {dy:.3f} z: -0.9 }}')
    return gz_service(f"/world/{WORLD_NAME}/light_config", "gz.msgs.Light", req)


# ------------------------------------------------- waiting for the camera --
LIGHT_WAIT_S = 0.6        # wall seconds after a light change before the "before" image
CHANGE_THRESH = 20.0      # mean grey difference at the new block spots that counts as "changed"
SETTLE_AFTER_CHANGE_S = 0.5
CHANGE_TIMEOUT_S = 10.0


def scene_mask(blocks, poses, R, t, K, W, H):
    """Image mask covering the new block positions (top face and footprint)."""
    import cv2
    mask = np.zeros((H, W), np.uint8)
    for (_n, btype, _c, _m), (x, y, yaw) in zip(blocks, poses):
        L = LENGTH[btype]
        pts = np.vstack([block_corners(x, y, yaw, L, TABLE_TOP_Z + BODY_H),
                         block_corners(x, y, yaw, L, TABLE_TOP_Z)])
        uv, _ = project(pts, R, t, K)
        hull = cv2.convexHull(uv.round().astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 255)
    return mask > 0


def _gray(node, msg):
    import cv2
    return cv2.cvtColor(node.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8"), cv2.COLOR_BGR2GRAY).astype(np.float32)


def wait_frames(node, seconds):
    """Spin for some wall time, return the latest image (grey)."""
    import time
    import rclpy
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.05)
    return _gray(node, node.latest)


def wait_for_change(node, before, mask, log):
    """Wait until the image at the new block spots differs from 'before', let it settle,
    and return the settled color image. None on timeout."""
    import time
    import rclpy
    start = time.monotonic()
    changed_at = None
    last_stamp = None
    while time.monotonic() - start < CHANGE_TIMEOUT_S:
        rclpy.spin_once(node, timeout_sec=0.05)
        msg = node.latest
        stamp = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        if stamp == last_stamp:
            continue
        last_stamp = stamp
        diff = float(np.abs(_gray(node, msg) - before)[mask].mean())
        if changed_at is None:
            if diff > CHANGE_THRESH:
                changed_at = time.monotonic()
        elif time.monotonic() - changed_at >= SETTLE_AFTER_CHANGE_S:
            if diff > CHANGE_THRESH:
                return node.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            changed_at = None      # went back, not settled yet
    return None


# ------------------------------------------------------------ ROS 2 node --
def run_ros():
    import cv2
    import rclpy
    from rclpy.node import Node
    from rclpy.time import Time
    from rclpy.duration import Duration
    from sensor_msgs.msg import Image, CameraInfo
    from cv_bridge import CvBridge
    import tf2_ros

    class Generator(Node):
        def __init__(self):
            super().__init__("sim_dataset_generator")
            self.declare_parameter("num_scenes", 1500)
            self.declare_parameter("out_dir", os.path.expanduser("~/yolo_blocks"))
            self.declare_parameter("val_fraction", 0.2)
            self.declare_parameter("seed", 0)
            self.declare_parameter("randomize_light", True)
            self.declare_parameter("debug_every", 10)
            self.bridge = CvBridge()
            self.latest = None
            self.K = None
            self.create_subscription(Image, IMAGE_TOPIC, self.on_image, 2)
            self.create_subscription(CameraInfo, INFO_TOPIC, self.on_info, 2)
            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        def on_image(self, msg):
            self.latest = msg

        def on_info(self, msg):
            self.K = np.array(msg.k).reshape(3, 3)
            self.size = (msg.width, msg.height)

    rclpy.init(args=sys.argv)
    node = Generator()
    p = lambda n: node.get_parameter(n).value
    rng = random.Random(p("seed"))
    out = p("out_dir")
    for sub in ("images/train", "images/val", "labels/train", "labels/val", "debug"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)

    log = node.get_logger()
    if not node.get_parameter("use_sim_time").value:
        log.error("run with --ros-args -p use_sim_time:=true, the image stamps are sim time")
        rclpy.shutdown()
        return
    log.info("waiting for camera_info, image and TF ...")
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.1)
        if node.K is None or node.latest is None:
            continue
        try:
            tf = node.tf_buffer.lookup_transform(CAMERA_FRAME, BASE_FRAME, Time())
            break
        except Exception:
            continue
    q, tr = tf.transform.rotation, tf.transform.translation
    R = quat_to_R(q.x, q.y, q.z, q.w)
    t = np.array([tr.x, tr.y, tr.z])
    K = node.K
    W, H = node.size
    log.info(f"camera ready, image {W}x{H}, fx {K[0,0]:.1f}, camera {t.round(3)}")

    # make sure every block from the world file really exists in Gazebo
    all_names = [main_name(c) for c in COLORS] + \
                [distractor_name(dt, c) for dt in DISTRACTOR_TYPES for c in COLORS]
    r = subprocess.run(["gz", "model", "--list"], capture_output=True, text=True)
    if r.returncode == 0:
        present = {ln.strip().lstrip("-").strip() for ln in r.stdout.splitlines()}
        missing = [n for n in all_names if n not in present]
        if missing:
            log.error(f"these blocks are not loaded in Gazebo: {missing}. "
                      "Check the Gazebo startup log for model:// errors and the models folder.")
            rclpy.shutdown()
            return
        log.info("all 16 blocks found in Gazebo")
    else:
        log.warning("could not run 'gz model --list', skipping the block check")

    light_ok = p("randomize_light")
    n_total, n_done, n_fail, n_retry = p("num_scenes"), 0, 0, 0

    # continue an earlier run in the same out_dir: count the scenes already saved
    import glob
    existing = glob.glob(os.path.join(out, "images", "*", "scene_*.jpg"))
    if existing:
        n_done = max(int(os.path.basename(f)[6:11]) for f in existing) + 1
        rng = random.Random(p("seed") * 100003 + n_done)   # new random scenes, not repeats
        log.info(f"found {len(existing)} scenes in {out}, continuing from scene {n_done}")
    with open(os.path.join(out, "data.yaml"), "w") as f:
        f.write(f"path: {out}\ntrain: images/train\nval: images/val\nnames:\n")
        for i, c in enumerate(CLASSES):
            f.write(f"  {i}: {c}\n")

    while rclpy.ok() and n_done < n_total:
        blocks = sample_blocks(rng)
        poses = sample_poses(rng, blocks, R, t, K, W, H)
        if poses is None:
            n_fail += 1
            if n_fail > 50:
                log.error("cannot place blocks, pick area too small or not visible")
                break
            continue
        # 1. change the light first and let it show up, so the "before" image already has it
        if light_ok and not randomize_light(rng):
            log.warning("light_config failed, light randomization switched off")
            light_ok = False
        before = wait_frames(node, LIGHT_WAIT_S)

        # 2. move the blocks
        if not move_blocks(blocks, poses):
            log.error(f"set_pose_vector failed 3 times. Gazebo answered: '{LAST_GZ_OUTPUT[:300]}'. "
                      f"Check that the sim is still running and the world is '{WORLD_NAME}'. "
                      "Start the script again with the same out_dir to continue.")
            break

        # 3. wait until the camera really shows the new scene (pixels at the new block
        #    spots changed), then wait a bit more for the blocks to settle
        mask = scene_mask(blocks, poses, R, t, K, W, H)
        img = wait_for_change(node, before, mask, log)
        if img is None:
            n_retry += 1
            log.warning(f"new scene did not appear in the camera within {CHANGE_TIMEOUT_S} s, retrying")
            if n_retry > 5:
                log.error("the camera image does not follow the block moves, stopping")
                break
            continue
        n_retry = 0
        split = "val" if rng.random() < p("val_fraction") else "train"
        stem = f"scene_{n_done:05d}"
        cv2.imwrite(os.path.join(out, "images", split, stem + ".jpg"), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        lines = label_lines(blocks, poses, R, t, K, W, H)
        with open(os.path.join(out, "labels", split, stem + ".txt"), "w") as f:
            f.write("\n".join(lines) + "\n")

        if p("debug_every") > 0 and n_done % p("debug_every") == 0:
            dbg = img.copy()
            for (name, btype, color, is_main), (x, y, yaw) in zip(blocks, poses):
                top = block_corners(x, y, yaw, LENGTH[btype], TABLE_TOP_Z + BODY_H)
                uv, _ = project(top, R, t, K)
                pts = uv.round().astype(np.int32).reshape(-1, 1, 2)
                col = (0, 255, 0) if is_main else (255, 0, 255)
                cv2.polylines(dbg, [pts], True, col, 2 if is_main else 1)
                txt = color if is_main else f"d {btype} {color}"
                cv2.putText(dbg, txt, tuple(pts[0, 0]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
            cv2.imwrite(os.path.join(out, "debug", stem + ".jpg"), dbg, [cv2.IMWRITE_JPEG_QUALITY, 80])

        n_done += 1
        if n_done % 50 == 0:
            log.info(f"{n_done}/{n_total} scenes")

    log.info(f"done, {n_done} scenes in {out}")
    node.destroy_node()
    rclpy.shutdown()


# --------------------------------------------------------- offline check --
def camera_from_urdf():
    """Camera pose base->optical and K computed from the URDF + sim sensor, for the self test."""
    def rpy(r, p, y):
        cr, sr, cp, sp, cy, sy = map(lambda a: a, (math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)))
        return np.array([[cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                         [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                         [-sp, cp*sr, cp*cr]])
    def T(xyz, r):
        M = np.eye(4); M[:3, :3] = r; M[:3, 3] = xyz; return M
    chain = T((0.31, -0.65, 0.3725), np.eye(3)) @ T((0, 0.435, 0.3975), rpy(0, 0, -1.5708)) \
        @ T((-0.395, 0, -0.06), np.eye(3)) @ T((0, 0, -0.03), rpy(math.pi, 0, -math.pi / 2))
    inv = np.linalg.inv(chain)            # base -> optical
    W, H, hfov = 1280, 720, 1.518
    f = (W / 2) / math.tan(hfov / 2)
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])
    return inv[:3, :3], inv[:3, 3], K, W, H


def selftest():
    R, t, K, W, H = camera_from_urdf()
    cam = -R.T @ t
    print(f"camera in base_link: {cam.round(3)}, height above table {cam[2]-TABLE_TOP_Z:.3f} m, fx {K[0,0]:.1f}")
    # visible part of the pick strip
    xs = np.linspace(*PICK_X, 47); ys = np.linspace(*PICK_Y, 157)
    vis = np.zeros((len(ys), len(xs)), bool)
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            uv, z = project(np.array([[x, y, TABLE_TOP_Z + BODY_H]]), R, t, K)
            vis[j, i] = z[0] > 0 and 0 <= uv[0, 0] <= W and 0 <= uv[0, 1] <= H
    rows = np.where(vis.all(axis=1))[0]
    print(f"pick strip x {PICK_X}, y {PICK_Y}")
    print(f"fully visible y range: {ys[rows[0]]:.3f} .. {ys[rows[-1]]:.3f}" if len(rows) else "pick strip NOT visible")
    # pixel size at table
    a, _ = project(np.array([[0.45, 0.1, TABLE_TOP_Z + BODY_H], [0.4819, 0.1, TABLE_TOP_Z + BODY_H]]), R, t, K)
    print(f"block width 31.8 mm is about {np.linalg.norm(a[0]-a[1]):.1f} px, one stud (9.4 mm) about {np.linalg.norm(a[0]-a[1])*9.4/31.8:.1f} px")
    # sampling
    rng = random.Random(1)
    ok, k_hist, same = 0, [0]*5, 0
    examples = []
    for _ in range(300):
        blocks = sample_blocks(rng)
        assert len(blocks) == 4 and len({b[0] for b in blocks}) == 4
        poses = sample_poses(rng, blocks, R, t, K, W, H)
        if poses is None:
            continue
        ok += 1
        mains = [b for b in blocks if b[3]]
        k_hist[len(mains)] += 1
        if any((not b[3]) and b[2] in {m[2] for m in mains} for b in blocks):
            same += 1
        lines = label_lines(blocks, poses, R, t, K, W, H)
        assert len(lines) == len(mains)
        for ln in lines:
            v = ln.split(); assert len(v) == 9 and all(0 <= float(c) <= 1 for c in v[1:])
        if len(examples) < 1:
            examples.append((blocks, poses, lines))
    print(f"scenes placed: {ok}/300, main-block count 1..4: {k_hist[1:]}, scenes with a same-colour distractor: {same}")
    blocks, poses, lines = examples[0]
    print("example scene:")
    for b, pz in zip(blocks, poses):
        print(f"  {b[0]:28s} main={b[3]!s:5s} x={pz[0]:.3f} y={pz[1]:.3f} yaw={math.degrees(pz[2]):.0f}")
    print("  labels:"); [print("   ", l) for l in lines]
    print("  set_pose_vector request starts with:", pose_text(blocks[0][0], 0.4, 0.1, 0.75, 0.3)[:90], "...")
    return R, t, K, W, H, examples


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        run_ros()
