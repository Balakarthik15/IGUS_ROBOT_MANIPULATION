#!/usr/bin/env python3
"""
Touch test for the hand-eye calibration. Needs a pin clamped in the gripper.

Why a pivot step: link6 is not at the pin tip. The tip sits some cm away, and
that offset turns with the wrist. So first we find where the tip is, relative
to link6, by touching one fixed dot from many wrist orientations.

Step 1 (once per pin):
    python3 touch_test.py pivot
  Touch ONE fixed dot with the pin tip, 8 or more times, with the pin tilted
  differently each time (change joints 4, 5 and 6). The script solves for the
  tip offset and saves it to pin_tip.json. Do not move the pin in the gripper
  after this step.

Step 2:
    python3 touch_test.py points [--sq 0.030]
  Tape the ChArUco board flat on the table. The HandEye panel must be running
  and publishing the camera pose, and the board must be detected.
    b        capture the board pose (arm out of the camera view)
    t A B    the pin tip is on the board corner that is A squares along the
             red axis and B squares along the green axis, counted from the
             corner where the drawn axes start. Press Enter when the arm is
             completely still.
    d        delete the last touch
    q        show the summary
  Error = camera prediction minus robot touch, in base_link axes, in mm.

Step 2b, go-to version (easier, same idea):
    python3 touch_test.py guide A B [--sq 0.030]
  The camera says where board corner A,B is. The script captures the board
  pose (keep the arm out of the camera view for the first 3 seconds), then
  shows live how far the pin tip is from that point, in base_link axes (mm):
      dx dy dz = where the camera says the corner is, minus where the tip is
  Jog the arm until dx and dy are near 0 with the tip about 20 mm above the
  paper, then lower the tip straight down. Look at the real gap between the
  tip and the corner (close-up photo or ruler). That real gap is your error.
  Ctrl-C to stop. Repeat for corners across the whole pick area.

Math self-test (no ROS needed):
    python3 touch_test.py --selftest
"""
import sys
import json
import time
import numpy as np

BASE, EEF, BOARD = "base_link", "link6", "handeye_target"
PIN_FILE = "pin_tip.json"


# ------------------------------------------------------------------ math
def quat_to_mat(x, y, z, w):
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def mean_rotation(rots):
    u, _, vt = np.linalg.svd(np.mean(rots, axis=0))
    if np.linalg.det(u @ vt) < 0:
        u[:, -1] *= -1
    return u @ vt


def rot_angle_deg(ra, rb):
    c = (np.trace(ra.T @ rb) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def max_pair_angle(rots):
    best = 0.0
    for i in range(len(rots)):
        for j in range(i + 1, len(rots)):
            best = max(best, rot_angle_deg(rots[i], rots[j]))
    return best


def solve_pivot(rots, ts):
    """Pin tip in the link6 frame, from touches of one fixed point.
    p_world = R_i * tip + t_i for every touch i."""
    a = np.vstack([np.hstack([r, -np.eye(3)]) for r in rots])
    b = -np.concatenate(ts)
    x = np.linalg.lstsq(a, b, rcond=None)[0]
    tip, fixed = x[:3], x[3:]
    res = np.array([np.linalg.norm(r @ tip + t - fixed) for r, t in zip(rots, ts)])
    return tip, fixed, res


def board_point(rb, tb, a, b, sq):
    return rb @ np.array([a * sq, b * sq, 0.0]) + tb


def touch_error(rb, tb, a, b, sq, tip, r_eef, t_eef):
    """camera prediction minus robot touch (m, base_link axes)"""
    return board_point(rb, tb, a, b, sq) - (r_eef @ tip + t_eef)


def summary(rows):
    err = np.array([r["err"] for r in rows]) * 1000.0
    horiz = np.linalg.norm(err[:, :2], axis=1)
    norm = np.linalg.norm(err, axis=1)
    mean = err.mean(axis=0)
    dev = np.linalg.norm(err - mean, axis=1)
    print("\n================ RESULT ================")
    print("board  corner(A,B)   dx      dy      dz     horiz   total   (mm)")
    for r, e, h, n in zip(rows, err, horiz, norm):
        print("  %-4d %3g,%-3g   %6.1f  %6.1f  %6.1f  %6.1f  %6.1f" % (
            r["board"], r["a"], r["b"], e[0], e[1], e[2], h, n))
    rms = lambda v: float(np.sqrt(np.mean(np.square(v))))
    print("touches                       : %d" % len(rows))
    print("mean error vector (mm)        : x %.1f  y %.1f  z %.1f" % tuple(mean))
    print("RMS error                     : total %.1f | horizontal %.1f | max total %.1f" % (
        rms(norm), rms(horiz), norm.max()))
    print("RMS after removing the mean   : %.1f mm" % rms(dev))
    print("========================================")
    print("Compare 'horizontal' with the error your gripper can take.")
    print("Same sign and size everywhere = a constant offset. Scattered = noise or a bad area.")


# ------------------------------------------------------------------ self-test
def _rot(axis, deg):
    axis = np.asarray(axis, float)
    axis /= np.linalg.norm(axis)
    th = np.radians(deg)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(th) * k + (1 - np.cos(th)) * (k @ k)


def selftest():
    rng = np.random.default_rng(3)
    tip_true = np.array([0.012, -0.020, 0.155])
    fixed_true = np.array([0.30, 0.10, 0.02])
    r0 = _rot([0, 1, 0], 180.0)  # tool pointing down

    # 1) pivot recovery with 0.5 mm touch noise, diverse tilts
    rots, ts = [], []
    for _ in range(10):
        r = r0 @ _rot(rng.normal(size=3), rng.uniform(15, 35))
        t = fixed_true - r @ tip_true + rng.normal(0, 0.0005, 3)
        rots.append(r)
        ts.append(t)
    tip, fixed, res = solve_pivot(rots, ts)
    print("pivot: tip error %.2f mm | fixed point error %.2f mm | residual RMS %.2f mm" % (
        np.linalg.norm(tip - tip_true) * 1e3, np.linalg.norm(fixed - fixed_true) * 1e3,
        np.sqrt(np.mean(res ** 2)) * 1e3))
    assert np.linalg.norm(tip - tip_true) < 0.003
    assert np.linalg.norm(fixed - fixed_true) < 0.003
    assert max_pair_angle(rots) > 40

    # 2) identical orientations cannot separate tip from fixed point -> must be flagged
    same = [r0] * 6
    assert max_pair_angle(same) < 1e-6

    # 3) board error: a 5 mm / -3 mm / 2 mm camera offset must be reported exactly
    rb = _rot([0.2, 0.1, 1.0], 40.0)
    tb = np.array([0.45, 0.05, 0.0])
    off = np.array([0.005, -0.003, 0.002])
    sq = 0.03
    for a, b in [(1, 1), (4, 1), (1, 6), (0, 0)]:
        touched = board_point(rb, tb, a, b, sq)           # true point on the table
        r_eef = r0 @ _rot(rng.normal(size=3), 20.0)
        t_eef = touched - r_eef @ tip_true                # arm that puts the tip there
        err = touch_error(rb, tb + off, a, b, sq, tip_true, r_eef, t_eef)
        assert np.allclose(err, off, atol=1e-9), err
    print("board error: reported camera offset matches (5.0, -3.0, 2.0) mm")

    rows = [{"board": 1, "a": 1, "b": 1, "err": off}, {"board": 1, "a": 4, "b": 1, "err": off * 1.2}]
    summary(rows)
    print("selftest OK")


# ------------------------------------------------------------------ ROS part
class Tf:
    def __init__(self):
        import rclpy
        from rclpy.node import Node
        from tf2_ros import Buffer, TransformListener
        rclpy.init()
        self.rclpy = rclpy
        self.node = Node("touch_test")
        self.buf = Buffer()
        TransformListener(self.buf, self.node, spin_thread=True)

    def get(self, child, max_age=1.0):
        from rclpy.time import Time
        tf = self.buf.lookup_transform(BASE, child, Time())
        stamp = Time.from_msg(tf.header.stamp)
        if stamp.nanoseconds != 0:
            age = (self.node.get_clock().now() - stamp).nanoseconds * 1e-9
            if age > max_age:
                raise RuntimeError("%s has not updated for %.1f s" % (child, age))
        t, q = tf.transform.translation, tf.transform.rotation
        return quat_to_mat(q.x, q.y, q.z, q.w), np.array([t.x, t.y, t.z])

    def still_eef(self, n=6, dt=0.1, tol=0.001):
        rs, ts = [], []
        for _ in range(n):
            r, t = self.get(EEF)
            rs.append(r)
            ts.append(t)
            time.sleep(dt)
        span = float(np.ptp(np.array(ts), axis=0).max())
        if span > tol:
            raise RuntimeError("arm is still moving (%.1f mm). Wait and press again." % (span * 1e3))
        return rs[-1], np.mean(ts, axis=0)

    def board_pose(self, n=20, dt=0.1):
        rs, ts = [], []
        for _ in range(n):
            r, t = self.get(BOARD)
            rs.append(r)
            ts.append(t)
            time.sleep(dt)
        ts = np.array(ts)
        rm = mean_rotation(np.array(rs))
        spread_mm = float(np.linalg.norm(ts - ts.mean(axis=0), axis=1).max() * 1e3)
        spread_deg = max(rot_angle_deg(rm, r) for r in rs)
        return rm, ts.mean(axis=0), spread_mm, spread_deg


def run_pivot():
    tf = Tf()
    rots, ts = [], []
    print(__doc__.split("Step 2")[0])
    while True:
        cmd = input("[Enter] record touch #%d  [d] delete last  [q] solve > " % (len(rots) + 1)).strip().lower()
        if cmd == "q":
            break
        if cmd == "d" and rots:
            rots.pop()
            ts.pop()
            print("  deleted, %d left" % len(rots))
            continue
        try:
            r, t = tf.still_eef()
        except Exception as exc:  # noqa: BLE001
            print("  %s" % exc)
            continue
        rots.append(r)
        ts.append(t)
        print("  recorded %d" % len(rots))
    if len(rots) < 6:
        print("Need at least 6 touches (8 or more is better).")
        return
    tip, fixed, res = solve_pivot(rots, ts)
    ang = max_pair_angle(rots)
    print("\npin tip in link6 frame (mm): x %.1f  y %.1f  z %.1f" % tuple(tip * 1e3))
    print("fixed dot in base_link (m)  : %.4f %.4f %.4f" % tuple(fixed))
    print("residual per touch (mm)     : " + " ".join("%.1f" % (v * 1e3) for v in res))
    print("residual RMS %.1f mm | max %.1f mm" % (np.sqrt(np.mean(res ** 2)) * 1e3, res.max() * 1e3))
    print("biggest wrist rotation between two touches: %.0f deg" % ang)
    if ang < 40:
        print("WARNING: orientations too similar. Tilt the pin more, then redo.")
    with open(PIN_FILE, "w") as f:
        json.dump({"tip_link6_m": tip.tolist(), "residual_rms_mm": float(np.sqrt(np.mean(res ** 2)) * 1e3)}, f)
    print("saved %s" % PIN_FILE)
    print("The residual RMS is the noise level of this whole test. Errors smaller than that cannot be seen.")


def run_points(sq):
    with open(PIN_FILE) as f:
        pin = json.load(f)
    tip = np.array(pin["tip_link6_m"])
    print("pin tip loaded (pivot residual %.1f mm)" % pin["residual_rms_mm"])
    tf = Tf()
    board, rows, rb, tb = 0, [], None, None
    print(__doc__.split("Step 2:")[1].split("Step 2b")[0])
    while True:
        cmd = input("[b] capture board  [t A B] touch  [d] delete last  [q] summary > ").strip().lower().split()
        if not cmd:
            continue
        if cmd[0] == "q":
            break
        if cmd[0] == "d" and rows:
            print("  deleted %s" % rows.pop())
            continue
        try:
            if cmd[0] == "b":
                rb, tb, s_mm, s_deg = tf.board_pose()
                board += 1
                print("  board #%d captured, noise %.1f mm / %.1f deg%s" % (
                    board, s_mm, s_deg, "  (large: board unsteady, capture again)" if s_mm > 3 or s_deg > 2 else ""))
            elif cmd[0] == "t" and len(cmd) == 3:
                if rb is None:
                    print("  capture the board first with b")
                    continue
                a, b = float(cmd[1]), float(cmd[2])
                r, t = tf.still_eef()
                err = touch_error(rb, tb, a, b, sq, tip, r, t)
                rows.append({"board": board, "a": a, "b": b, "err": err})
                print("  error mm: x %.1f  y %.1f  z %.1f   horizontal %.1f" % (
                    err[0] * 1e3, err[1] * 1e3, err[2] * 1e3, np.linalg.norm(err[:2]) * 1e3))
            else:
                print("  unknown command")
        except Exception as exc:  # noqa: BLE001
            print("  %s" % exc)
    if rows:
        summary(rows)
    else:
        print("no touches recorded")


def run_guide(a, b, sq):
    with open(PIN_FILE) as f:
        pin = json.load(f)
    tip = np.array(pin["tip_link6_m"])
    tf = Tf()
    print("Keep the arm OUT of the camera view. Capturing the board in 3 s ...")
    time.sleep(3.0)
    rb, tb, s_mm, s_deg = tf.board_pose()
    print("board captured, noise %.1f mm / %.1f deg%s" % (
        s_mm, s_deg, "  (large: board unsteady, restart)" if s_mm > 3 or s_deg > 2 else ""))
    target = board_point(rb, tb, a, b, sq)
    print("camera says corner (%g,%g) is at base_link (m): %.4f %.4f %.4f" % ((a, b) + tuple(target)))
    print("Now jog the arm. dx dy = how far the tip must still move. Ctrl-C to stop.\n")
    try:
        while True:
            try:
                r, t = tf.get(EEF)
            except Exception as exc:  # noqa: BLE001
                print("\r%s" % exc, end="")
                time.sleep(0.3)
                continue
            d = (target - (r @ tip + t)) * 1e3
            print("\rdx %7.1f  dy %7.1f  dz %7.1f   horizontal %6.1f mm   " % (
                d[0], d[1], d[2], np.linalg.norm(d[:2])), end="", flush=True)
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    elif len(sys.argv) >= 4 and sys.argv[1] == "guide":
        sq_val = 0.030
        if "--sq" in sys.argv:
            sq_val = float(sys.argv[sys.argv.index("--sq") + 1])
        run_guide(float(sys.argv[2]), float(sys.argv[3]), sq_val)
    elif len(sys.argv) > 1 and sys.argv[1] == "pivot":
        run_pivot()
    elif len(sys.argv) > 1 and sys.argv[1] == "points":
        sq_val = 0.030
        if "--sq" in sys.argv:
            sq_val = float(sys.argv[sys.argv.index("--sq") + 1])
        run_points(sq_val)
    else:
        print(__doc__)