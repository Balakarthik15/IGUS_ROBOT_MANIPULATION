#!/usr/bin/env python3
"""Sim dataset generator for YOLO OBB block detection (igus ReBeL thesis setup, lab table v2).

Classes (5 main 2x4 studded blocks):  0 red, 1 green, 2 yellow, 3 white, 4 lightblue

Scenes follow the real lab images: at most 5 blocks on the board, all studs up,
block centres at least 140 mm apart (room for the open gripper), random yaw.
Distractors come from blocks_manifest.json (2x2, 2x6, 4x4, lime, blue, orange,
step pieces, blocks without studs, similar shades ...). Only the main blocks get a
label line. Distractors stay unlabelled, so YOLO learns to treat them as background.

Scene types (shares are parameters):
  many         50 %  3..5 main blocks, distractors up to 5 blocks in total
  few          30 %  1..2 main blocks, distractors up to 5 blocks in total
  distractors  10 %  1..5 distractors only, empty label file
  lab           5 %  all 5 main blocks on the lab dice spots, colours shuffled
  empty         5 %  empty board, empty label file

How it works per scene:
  1. pick the blocks, sample non-overlapping poses on the cardboard that
     the camera can fully see
  2. move them with ONE gz set_pose_vector call (unused blocks go to parking)
  3. optionally randomize the sun light
  4. wait until the camera shows the new scene, let it settle, save the image
  5. project the top corners of each main block into the image -> OBB label

Run (sim running, arm in a pose that does not cover the pick area):
  python3 gen_sim_dataset.py --ros-args -p use_sim_time:=true -p num_scenes:=1500 \
      -p out_dir:=$HOME/yolo_blocks_v2 -p manifest:=<your package>/models/blocks_manifest.json

Offline geometry check (no ROS needed):
  python3 gen_sim_dataset.py --selftest [path/to/blocks_manifest.json]
"""
import json
import math
import os
import random
import subprocess
import sys

import numpy as np

# ------------------------------------------------------------------ setup --
WORLD_NAME = "rebel_lab_world"
SPAWN_XYZ = (0.0, 0.0, 0.79)            # create node: robot spawned at z = 0.79 (on the 40 mm plate)
SURFACE_WORLD_Z = 0.753                 # top of the cardboard in the Gazebo world (table 0.75 + 3 mm)
SURFACE_Z = SURFACE_WORLD_Z - SPAWN_XYZ[2]   # in base_link = -0.037

# Pick area on the board, in base_link (metres). Block footprints stay inside it.
# x from the robot plate edge (170 mm) to 25 mm before the table edge.
# y covers where the blocks stand in the lab camera images.
PICK_X = (0.170, 0.545)
PICK_Y = (-0.125, 0.200)
MAX_BLOCKS = 5                          # never more than 5 blocks on the board
MIN_CENTER_DIST = 0.140                 # gripper needs 140 mm between block centres
EDGE_MARGIN = 0.005                     # keep blocks this far inside the cardboard edge (m)
MIN_GAP = 0.008                         # free space between blocks (m)
IMG_MARGIN_PX = 8                       # block must be this far inside the image

# Lab dice spots (centre x, y in base_link), long side along y, as in the world file.
# Columns measured in the lab, rows and middle spot taken from the lab camera images.
# In lab scenes the 5 colours are shuffled over these spots.
LAB_SPOTS = [(0.444, -0.087), (0.271, -0.087), (0.367, 0.042), (0.444, 0.151), (0.271, 0.151)]
LAB_YAW = math.pi / 2
LAB_JITTER_XY = 0.006                   # +- shift around the lab spot (m)
LAB_JITTER_YAW = math.radians(6)        # +- turn around the lab yaw

BODY_H = 0.019                          # body height without studs, the labelled top face
CLASSES = ["red", "green", "yellow", "white", "lightblue"]   # class id = index
SAME_COLOUR_PROB = 0.5    # chance a distractor has the colour of a main block in the scene
SCENE_TYPES = ["many", "few", "distractors", "lab", "empty"]
SCENE_SHARES = [0.50, 0.30, 0.10, 0.05, 0.05]

# Camera topics (sim bridge uses the real D455 names)
IMAGE_TOPIC = "/camera/camera/color/image_raw"
INFO_TOPIC = "/camera/camera/color/camera_info"
CAMERA_FRAME = "camera_color_optical_frame"
BASE_FRAME = "base_link"


# ---------------------------------------------------------------- blocks --
def entity_name(model, is_main, color):
    """Gazebo entity name as used in rebel_world.sdf."""
    if is_main:
        return f"block_{color}"
    return "d_" + model[len("block_"):]


def load_manifest(path):
    """Read blocks_manifest.json from make_blocks.py. Returns (mains, distractors), lists of dicts."""
    with open(path) as f:
        data = json.load(f)
    mains, dists = [], []
    for b in data["blocks"]:
        if b["kind"] == "upside":          # only studs-up blocks are used
            continue
        blk = {"name": entity_name(b["model"], b["is_main"], b["color"]),
               "model": b["model"], "kind": b["kind"], "color": b["color"],
               "is_main": b["is_main"], "in_lab": b["in_lab"],
               "length": b["length"], "width": b["width"], "height": b["height"]}
        (mains if b["is_main"] else dists).append(blk)
    found = sorted(m["color"] for m in mains)
    if found != sorted(CLASSES):
        raise SystemExit(f"manifest main colours {found} do not match CLASSES {CLASSES}")
    mains.sort(key=lambda m: CLASSES.index(m["color"]))
    return mains, dists


def all_entities(mains, dists):
    return [b["name"] for b in mains + dists]


def parking_pose(i):
    """Parking spot number i (world frame), far from the table and out of camera view."""
    return (3.0 + 0.3 * (i // 6), -0.75 + 0.25 * (i % 6), 0.0, 0.0)


# --------------------------------------------------------------- geometry --
def block_corners(x, y, yaw, length, width, z):
    """4 corners (base frame) of a block rectangle at height z, counter-clockwise."""
    c, s = math.cos(yaw), math.sin(yaw)
    hl, hw = length / 2.0, width / 2.0
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


def top_face(blk, pose):
    x, y, yaw = pose
    return block_corners(x, y, yaw, blk["length"], blk["width"], SURFACE_Z + min(BODY_H, blk["height"]))


# ---------------------------------------------------------- scene sampling --
def pick_scene_type(rng, shares):
    return rng.choices(SCENE_TYPES, weights=shares)[0]


def sample_blocks(rng, scene, mains, dists):
    """Choose the blocks of one scene. Returns a list of block dicts, mains first."""
    if scene == "empty":
        return []
    if scene == "lab":
        return list(mains)
    if scene == "distractors":
        k, n_d = 0, rng.randint(1, MAX_BLOCKS)
    else:
        k = rng.randint(3, 5) if scene == "many" else rng.randint(1, 2)
        n_d = rng.randint(0, MAX_BLOCKS - k)
    chosen = rng.sample(mains, k)
    scene_colours = {m["color"] for m in chosen}
    same = [d for d in dists if d["color"] in scene_colours]
    picked = []
    while len(picked) < n_d:
        pool = same if (same and rng.random() < SAME_COLOUR_PROB) else dists
        d = rng.choice(pool)
        if d not in picked:
            picked.append(d)
    return chosen + picked


def pose_ok(blk, pose, placed, rects, min_dist, R, t, K, width, height):
    """Footprint inside the pick area, far enough from the other blocks, top face fully
    in the image. Returns the footprint or None."""
    top = top_face(blk, pose)
    foot = top[:, :2]
    m = EDGE_MARGIN
    if foot[:, 0].min() < PICK_X[0] + m or foot[:, 0].max() > PICK_X[1] - m:
        return None
    if foot[:, 1].min() < PICK_Y[0] + m or foot[:, 1].max() > PICK_Y[1] - m:
        return None
    if any(math.hypot(pose[0] - q[0], pose[1] - q[1]) < min_dist for q in placed):
        return None
    if any(rects_overlap(foot, r, MIN_GAP) for r in rects):
        return None
    uv, z = project(top, R, t, K)
    if (z <= 0).any():
        return None
    pm = IMG_MARGIN_PX
    if uv[:, 0].min() < pm or uv[:, 0].max() > width - pm or \
       uv[:, 1].min() < pm or uv[:, 1].max() > height - pm:
        return None
    return foot


def sample_poses(rng, blocks, scene, R, t, K, width, height, min_dist=MIN_CENTER_DIST, tries=300):
    """Poses (x, y, yaw) for all blocks, or None if they do not fit."""
    if scene == "lab":
        spots = rng.sample(LAB_SPOTS, len(blocks))     # shuffle colours over the spots
    for _ in range(tries):
        placed, rects = [], []
        ok = True
        for i, blk in enumerate(blocks):
            good = False
            for _ in range(100):
                if scene == "lab":
                    sx, sy = spots[i]
                    pose = (sx + rng.uniform(-LAB_JITTER_XY, LAB_JITTER_XY),
                            sy + rng.uniform(-LAB_JITTER_XY, LAB_JITTER_XY),
                            LAB_YAW + rng.uniform(-LAB_JITTER_YAW, LAB_JITTER_YAW))
                    md = 0.0                                   # the lab spots are fixed
                else:
                    pose = (rng.uniform(*PICK_X), rng.uniform(*PICK_Y), rng.uniform(0.0, math.pi))
                    md = min_dist
                foot = pose_ok(blk, pose, placed, rects, md, R, t, K, width, height)
                if foot is None:
                    continue
                placed.append(pose)
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
    for blk, pose in zip(blocks, poses):
        if not blk["is_main"]:
            continue
        uv, _ = project(top_face(blk, pose), R, t, K)
        coords = []
        for u, v in uv:
            coords += [min(max(u / width, 0.0), 1.0), min(max(v / height, 0.0), 1.0)]
        lines.append(f"{CLASSES.index(blk['color'])} " + " ".join(f"{c:.6f}" for c in coords))
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
    for _ in range(attempts):
        r = subprocess.run(cmd, capture_output=True, text=True)
        LAST_GZ_OUTPUT = (r.stdout + r.stderr).strip()
        if r.returncode == 0 and "data: true" in r.stdout:
            return True
        time.sleep(1.0)
    return False


def move_blocks(blocks, poses, names):
    """Put scene blocks on the cardboard, park all others. One set_pose_vector call."""
    on_table = {b["name"]: p for b, p in zip(blocks, poses)}
    parts = []
    for i, name in enumerate(names):
        if name in on_table:
            x, y, yaw = on_table[name]
            parts.append(pose_text(name, x + SPAWN_XYZ[0], y + SPAWN_XYZ[1],
                                   SURFACE_WORLD_Z + 0.0005, yaw))
        else:
            px, py, pz, pyaw = parking_pose(i)
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
    for blk, (x, y, yaw) in zip(blocks, poses):
        pts = np.vstack([top_face(blk, (x, y, yaw)),
                         block_corners(x, y, yaw, blk["length"], blk["width"], SURFACE_Z)])
        uv, _ = project(pts, R, t, K)
        hull = cv2.convexHull(uv.round().astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 255)
    return mask > 0


def _gray(node, msg):
    import cv2
    return cv2.cvtColor(node.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8"),
                        cv2.COLOR_BGR2GRAY).astype(np.float32)


def wait_frames(node, seconds):
    """Spin for some wall time, return the latest image (grey)."""
    import time
    import rclpy
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.05)
    return _gray(node, node.latest)


def wait_for_change(node, before, mask):
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


def draw_debug(img, blocks, poses, R, t, K):
    import cv2
    dbg = img.copy()
    for blk, pose in zip(blocks, poses):
        uv, _ = project(top_face(blk, pose), R, t, K)
        pts = uv.round().astype(np.int32).reshape(-1, 1, 2)
        col = (0, 255, 0) if blk["is_main"] else (255, 0, 255)
        cv2.polylines(dbg, [pts], True, col, 2 if blk["is_main"] else 1)
        txt = blk["color"] if blk["is_main"] else blk["name"]
        cv2.putText(dbg, txt, tuple(pts[0, 0]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
    return dbg


# ------------------------------------------------------------ ROS 2 node --
def run_ros():
    import cv2
    import glob
    import rclpy
    from rclpy.node import Node
    from rclpy.time import Time
    from sensor_msgs.msg import Image, CameraInfo
    from cv_bridge import CvBridge
    import tf2_ros

    class Generator(Node):
        def __init__(self):
            super().__init__("sim_dataset_generator")
            self.declare_parameter("num_scenes", 1500)
            self.declare_parameter("out_dir", os.path.expanduser("~/yolo_blocks_v2"))
            self.declare_parameter("manifest", "blocks_manifest.json")
            self.declare_parameter("val_fraction", 0.2)
            self.declare_parameter("seed", 0)
            self.declare_parameter("randomize_light", True)
            self.declare_parameter("debug_every", 1)          # 1 = debug image for every scene
            # shares of the scene types many, few, distractors, lab, empty (normalised)
            self.declare_parameter("scene_shares", SCENE_SHARES)
            self.declare_parameter("min_center_dist", MIN_CENTER_DIST)
            self.declare_parameter("lab_only_distractors", False)  # True = only blocks you also have in the lab
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
    log = node.get_logger()

    manifest = os.path.expanduser(p("manifest"))
    if not os.path.isfile(manifest):
        log.error(f"manifest not found: {manifest}. Pass -p manifest:=<package>/models/blocks_manifest.json")
        rclpy.shutdown()
        return
    mains, dists = load_manifest(manifest)
    names = all_entities(mains, dists)          # every block in the world, for moving and parking
    if p("lab_only_distractors"):
        dists = [d for d in dists if d["in_lab"]]
    log.info(f"{len(mains)} main blocks, {len(dists)} distractors in use")

    rng = random.Random(p("seed"))
    out = os.path.expanduser(p("out_dir"))
    for sub in ("images/train", "images/val", "labels/train", "labels/val", "debug"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)

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
    r = subprocess.run(["gz", "model", "--list"], capture_output=True, text=True)
    if r.returncode == 0:
        present = {ln.strip().lstrip("-").strip() for ln in r.stdout.splitlines()}
        missing = [n for n in names if n not in present]
        if missing:
            log.error(f"these blocks are not loaded in Gazebo: {missing}. "
                      "Check the Gazebo startup log for model:// errors and the models folder.")
            rclpy.shutdown()
            return
        log.info(f"all {len(names)} blocks found in Gazebo")
    else:
        log.warning("could not run 'gz model --list', skipping the block check")

    shares = [float(v) for v in p("scene_shares")]
    if len(shares) != len(SCENE_TYPES):
        log.error(f"scene_shares needs {len(SCENE_TYPES)} numbers for {SCENE_TYPES}")
        rclpy.shutdown()
        return
    min_dist = p("min_center_dist")
    # can the camera see the lab spots? (lab scenes are impossible otherwise)
    hidden = [s for s in LAB_SPOTS if pose_ok(mains[0], (*s, LAB_YAW), [], [], 0.0, R, t, K, W, H) is None]
    if hidden and shares[3] > 0:
        log.warning(f"lab spots {hidden} are not fully visible in the camera, lab scenes switched off")
        shares[3] = 0.0
    log.info("scene shares " + ", ".join(f"{n} {v:.2f}" for n, v in zip(SCENE_TYPES, shares)))
    prev_blocks, prev_poses = [], []       # last scene, its spots must change in the image too

    light_ok = p("randomize_light")
    n_total, n_done, n_fail, n_retry = p("num_scenes"), 0, 0, 0

    # continue an earlier run in the same out_dir: count the scenes already saved
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
        scene = pick_scene_type(rng, shares)
        blocks = sample_blocks(rng, scene, mains, dists)
        poses = sample_poses(rng, blocks, scene, R, t, K, W, H, min_dist)
        if poses is None:
            n_fail += 1
            if n_fail > 50:
                log.error("cannot place blocks, pick area too small or not visible")
                break
            continue
        n_fail = 0
        # 1. change the light first and let it show up, so the "before" image already has it
        if light_ok and not randomize_light(rng):
            log.warning("light_config failed, light randomization switched off")
            light_ok = False
        before = wait_frames(node, LIGHT_WAIT_S)

        # 2. move the blocks
        if not move_blocks(blocks, poses, names):
            log.error(f"set_pose_vector failed 3 times. Gazebo answered: '{LAST_GZ_OUTPUT[:300]}'. "
                      f"Check that the sim is still running and the world is '{WORLD_NAME}'. "
                      "Start the script again with the same out_dir to continue.")
            break

        # 3. wait until the camera really shows the new scene (pixels at the old and the
        #    new block spots changed), then let the blocks settle
        mask = scene_mask(prev_blocks + blocks, prev_poses + poses, R, t, K, W, H)
        if mask.any():
            img = wait_for_change(node, before, mask)
        else:                                  # empty board after an empty board: nothing to see change
            wait_frames(node, SETTLE_AFTER_CHANGE_S + 0.5)
            img = node.bridge.imgmsg_to_cv2(node.latest, desired_encoding="bgr8")
        if img is None:
            n_retry += 1
            log.warning(f"new scene did not appear in the camera within {CHANGE_TIMEOUT_S} s, retrying")
            if n_retry > 5:
                log.error("the camera image does not follow the block moves, stopping")
                break
            continue
        n_retry = 0
        prev_blocks, prev_poses = blocks, poses
        split = "val" if rng.random() < p("val_fraction") else "train"
        stem = f"scene_{n_done:05d}"
        cv2.imwrite(os.path.join(out, "images", split, stem + ".jpg"), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        lines = label_lines(blocks, poses, R, t, K, W, H)
        with open(os.path.join(out, "labels", split, stem + ".txt"), "w") as f:
            f.write("".join(ln + "\n" for ln in lines))      # empty file = no main block

        if p("debug_every") > 0 and n_done % p("debug_every") == 0:
            cv2.imwrite(os.path.join(out, "debug", stem + ".jpg"),
                        draw_debug(img, blocks, poses, R, t, K), [cv2.IMWRITE_JPEG_QUALITY, 80])

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
        cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
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


def selftest(manifest_path):
    mains, dists = load_manifest(manifest_path)
    print(f"manifest: {len(mains)} main, {len(dists)} distractors "
          f"({sum(d['in_lab'] for d in dists)} also in the lab)")
    R, t, K, W, H = camera_from_urdf()
    cam = -R.T @ t
    print(f"camera in base_link: {cam.round(3)}, height above cardboard {cam[2]-SURFACE_Z:.3f} m, fx {K[0,0]:.1f}")
    corners = np.array([[x, y, SURFACE_Z] for x in PICK_X for y in PICK_Y])
    uv, _ = project(corners, R, t, K)
    print("cardboard corners in the image (px):", [tuple(c) for c in uv.round().astype(int)])
    xs = np.linspace(*PICK_X, 60); ys = np.linspace(*PICK_Y, 50)
    vis = sum(1 for x in xs for y in ys
              if (lambda a: 0 <= a[0][0, 0] <= W and 0 <= a[0][0, 1] <= H and a[1][0] > 0)(
                  project(np.array([[x, y, SURFACE_Z + BODY_H]]), R, t, K)))
    print(f"visible share of the cardboard: {100 * vis / (len(xs) * len(ys)):.0f} %")
    hidden = [s for s in LAB_SPOTS if pose_ok(mains[0], (*s, LAB_YAW), [], [], 0.0, R, t, K, W, H) is None]
    print("lab spots fully visible:", "all" if not hidden else f"NOT {hidden}")
    a, _ = project(np.array([[0.35, 0.05, SURFACE_Z + BODY_H], [0.3819, 0.05, SURFACE_Z + BODY_H]]), R, t, K)
    print(f"block width 31.9 mm is about {np.linalg.norm(a[0]-a[1]):.1f} px")

    rng = random.Random(1)
    N = 2000
    count = {n: 0 for n in SCENE_TYPES}
    fails, k_hist, n_hist, dmin, examples = 0, [0] * 6, [0] * 6, 9.0, {}
    for _ in range(N):
        scene = pick_scene_type(rng, SCENE_SHARES)
        blocks = sample_blocks(rng, scene, mains, dists)
        assert len({b["name"] for b in blocks}) == len(blocks) <= MAX_BLOCKS
        assert all(b["kind"] != "upside" for b in blocks)
        poses = sample_poses(rng, blocks, scene, R, t, K, W, H)
        if poses is None:
            fails += 1
            continue
        count[scene] += 1
        k = sum(b["is_main"] for b in blocks)
        k_hist[k] += 1
        n_hist[len(blocks)] += 1
        if scene != "lab":
            for i in range(len(poses)):
                for j in range(i):
                    dmin = min(dmin, math.hypot(poses[i][0] - poses[j][0], poses[i][1] - poses[j][1]))
        lines = label_lines(blocks, poses, R, t, K, W, H)
        assert len(lines) == k
        for ln in lines:
            v = ln.split(); assert len(v) == 9 and all(0 <= float(c) <= 1 for c in v[1:])
        examples.setdefault(scene, (blocks, poses, lines))
    print(f"{N} scenes tried, {fails} did not fit (they are simply sampled again)")
    print("scene types:", ", ".join(f"{n} {100 * c / (N - fails):.0f} %" for n, c in count.items()))
    print(f"main blocks per scene 0..5: {k_hist}")
    print(f"blocks on the board 0..5:   {n_hist}")
    print(f"smallest centre distance in random scenes: {dmin * 1000:.0f} mm")
    for scene, (blocks, poses, lines) in examples.items():
        print(f"example {scene}:")
        for b, pz in zip(blocks, poses):
            print(f"  {b['name']:24s} main={b['is_main']!s:5s} x={pz[0]:.3f} y={pz[1]:.3f} yaw={math.degrees(pz[2]):.0f}")
    return R, t, K, W, H, mains, dists, examples


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        rest = [a for a in sys.argv[1:] if a != "--selftest"]
        selftest(rest[0] if rest else "blocks_manifest.json")
    else:
        run_ros()
