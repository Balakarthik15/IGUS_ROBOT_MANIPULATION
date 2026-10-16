#!/usr/bin/env python3
"""
Hand-eye consistency check.

Idea: the calibration board is clamped in the gripper, so the transform
link6 -> handeye_target must be CONSTANT, whatever pose the arm is in.
TF computes it through the calibrated camera pose:

    link6 <- base_link <- camera_color_optical_frame <- handeye_target
             (robot)      (your calibration)             (board detection)

If the calibration is good, the value barely changes between poses.
If it is bad, it drifts. The spread is your real error in mm and degrees.

Before you start:
  - source your ROS 2 workspace
  - the HandEye panel is open, you clicked Solve, and it publishes the camera TF
  - the board is visible and axes are drawn in the target_detection image

Run:
    python3 check_handeye_consistency.py

Then: move the arm to a NEW pose (8 or more, different tilts and positions),
wait until it is completely still, press Enter to record. Type q + Enter to finish.

Math self-test (no ROS needed):
    python3 check_handeye_consistency.py --selftest
"""
import sys
import numpy as np

REF_FRAME = "link6"            # frame the board is clamped to
BOARD_FRAME = "handeye_target"  # board frame published by the HandEye panel
MAX_AGE_S = 1.0                 # older than this = board probably not detected


# ---------------------------------------------------------------- math
def quat_to_mat(x, y, z, w):
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def mean_rotation(rots):
    """Average of rotation matrices, projected back onto a valid rotation."""
    u, _, vt = np.linalg.svd(np.mean(rots, axis=0))
    if np.linalg.det(u @ vt) < 0:
        u[:, -1] *= -1
    return u @ vt


def rot_angle_deg(ra, rb):
    c = (np.trace(ra.T @ rb) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def analyse(ts, rots):
    """Return mean translation, per-sample translation deviation (mm),
    per-sample rotation deviation (deg)."""
    ts = np.asarray(ts, dtype=float)
    t_mean = ts.mean(axis=0)
    dt_mm = np.linalg.norm(ts - t_mean, axis=1) * 1000.0
    r_mean = mean_rotation(np.asarray(rots))
    dr_deg = np.array([rot_angle_deg(r_mean, r) for r in rots])
    return t_mean, dt_mm, dr_deg


def rms(v):
    return float(np.sqrt(np.mean(np.square(v))))


def grade(t_rms_mm, r_rms_deg):
    if t_rms_mm <= 5 and r_rms_deg <= 2:
        return "GOOD"
    if t_rms_mm <= 10 and r_rms_deg <= 4:
        return "BORDERLINE"
    return "BAD"


def report(ts, rots):
    t_mean, dt, dr = analyse(ts, rots)
    print("\n================ RESULT ================")
    print("poses recorded         : %d" % len(ts))
    print("mean link6->board (m)  : x=%.4f  y=%.4f  z=%.4f" % tuple(t_mean))
    print("translation spread     : RMS %.1f mm | max %.1f mm" % (rms(dt), dt.max()))
    print("rotation spread        : RMS %.2f deg | max %.2f deg" % (rms(dr), dr.max()))
    print("per pose (mm / deg)    : " + "  ".join(
        "#%d:%.1f/%.1f" % (i + 1, a, b) for i, (a, b) in enumerate(zip(dt, dr))))
    print("verdict                : %s" % grade(rms(dt), rms(dr)))
    worst = int(np.argmax(dt))
    print("worst pose (translation): #%d" % (worst + 1))
    print("========================================")
    print("Rule of thumb: <=5 mm and <=2 deg good; <=10 mm and <=4 deg borderline.")


# ---------------------------------------------------------------- self-test
def _rotvec_to_mat(v):
    th = np.linalg.norm(v)
    if th < 1e-12:
        return np.eye(3)
    k = v / th
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * kx + (1 - np.cos(th)) * (kx @ kx)


def selftest():
    rng = np.random.default_rng(0)
    r_true = quat_to_mat(0.1, 0.2, 0.3, 0.9)
    t_true = np.array([0.02, -0.05, 0.11])

    # 1) quaternion conversion gives a proper rotation
    assert np.allclose(r_true @ r_true.T, np.eye(3), atol=1e-9)
    assert abs(np.linalg.det(r_true) - 1) < 1e-9

    # 2) zero noise -> zero spread
    ts = [t_true] * 6
    rots = [r_true] * 6
    _, dt, dr = analyse(ts, rots)
    assert dt.max() < 1e-6 and dr.max() < 1e-6

    # 3) known noise -> spread matches theory (3 axes, sigma each)
    n, s_t_mm, s_r_deg = 5000, 3.0, 1.0
    ts = [t_true + rng.normal(0, s_t_mm / 1000.0, 3) for _ in range(n)]
    rots = [r_true @ _rotvec_to_mat(np.radians(rng.normal(0, s_r_deg, 3)))
            for _ in range(n)]
    _, dt, dr = analyse(ts, rots)
    exp_t, exp_r = s_t_mm * np.sqrt(3), s_r_deg * np.sqrt(3)
    print("selftest translation RMS: got %.2f mm, expected %.2f mm" % (rms(dt), exp_t))
    print("selftest rotation RMS   : got %.2f deg, expected %.2f deg" % (rms(dr), exp_r))
    assert abs(rms(dt) - exp_t) / exp_t < 0.05
    assert abs(rms(dr) - exp_r) / exp_r < 0.05

    # 4) one bad pose is flagged as the worst one
    ts = [t_true + rng.normal(0, 0.001, 3) for _ in range(8)]
    ts[5] = ts[5] + np.array([0.04, 0, 0])
    rots = [r_true] * 8
    _, dt, _ = analyse(ts, rots)
    assert int(np.argmax(dt)) == 5

    print("selftest OK")


# ---------------------------------------------------------------- ROS part
def run_ros():
    import rclpy
    from rclpy.node import Node
    from rclpy.time import Time
    from tf2_ros import Buffer, TransformListener

    rclpy.init()
    node = Node("handeye_consistency_check")
    buf = Buffer()
    TransformListener(buf, node, spin_thread=True)

    ts, rots = [], []
    print(__doc__.split("Run:")[0])
    try:
        while True:
            cmd = input("[Enter] record pose #%d   [q] finish > " % (len(ts) + 1))
            if cmd.strip().lower() == "q":
                break
            try:
                tf = buf.lookup_transform(REF_FRAME, BOARD_FRAME, Time())
            except Exception as exc:  # noqa: BLE001
                print("  TF lookup failed: %s" % exc)
                continue

            stamp = Time.from_msg(tf.header.stamp)
            age = None
            if stamp.nanoseconds != 0:
                age = (node.get_clock().now() - stamp).nanoseconds * 1e-9
                if age > MAX_AGE_S:
                    print("  STALE (%.1f s old). Board not detected? Not recorded." % age)
                    continue

            t, q = tf.transform.translation, tf.transform.rotation
            ts.append([t.x, t.y, t.z])
            rots.append(quat_to_mat(q.x, q.y, q.z, q.w))
            print("  recorded: x=%.4f y=%.4f z=%.4f%s" % (
                t.x, t.y, t.z, "" if age is None else "  (age %.2f s)" % age))
    except (EOFError, KeyboardInterrupt):
        pass

    if len(ts) < 3:
        print("Need at least 3 poses (8 or more is better).")
    else:
        report(ts, rots)
    rclpy.shutdown()


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        run_ros()