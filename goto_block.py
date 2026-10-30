#!/usr/bin/env python3
"""
Move to a block given by its pixel (and optional yaw), or pick it. Robot moves SLOWLY.

Needs homography_calib.py in the same folder (all calibration values live there).

    python3 goto_block.py U V --dry          only print the targets
    python3 goto_block.py U V --go pre       move above the block
    python3 goto_block.py U V --go grasp     move down to grasp height (gripper untouched)
    python3 goto_block.py U V --go pick      pre-grasp, open, down, close, up
    (planning group defaults to igus_rebel_arm, change with --group)
    place_sequence() is used by pick_block.py --mode place

Options (defaults in brackets):
    --angle-img DEG     long-side angle of the block in the image (from the detector). If given,
                        the gripper yaw is computed from it. If not, --yaw-deg is used.
    --yaw-deg -4.2      fixed link6 yaw when no --angle-img
    --yaw-offset 90     deg between link6 x and the finger closing direction (GRIPPER_YAW_OFFSET)
    --grasp-z 0.129     link6 z when closing on the block (block on the 3 mm board)
    --pre-height 0.08   pre-grasp = grasp z + this (m)
    --speed 0.05        velocity and acceleration scaling (5 %), allowed 0.01 .. 0.2
    --grip-time 3.0     seconds the gripper output stays on
    --no-confirm        pick without pressing Enter before each step
For picking straight from the detector, use pick_block.py.
"""
import argparse
import math
import sys
import time

from homography_calib import pixel_to_base, image_angle_to_base_yaw, fold90, in_area

GRASP_Z = 0.129
# Angle between the link6 x axis and the direction in which the fingers close (deg).
# Measured 28 Oct: gripper closed on a red block (it pushes the block straight), link6 yaw -4.15 deg,
# detector angle -3.5..+4 deg at pixel (495, 529) -> offset 78..86, mean 82.
GRIPPER_YAW_OFFSET = -8
PRE_HEIGHT = 0.080
SPEED = 0.05
MAX_SPEED = 0.2
GRIP_TIME = 3.0
OUT_CLOSE, OUT_OPEN = 30, 31
BASE, EEF = "base_link", "link6"


def down_quat(yaw_deg):
    """link6 straight down, turned by yaw about base z (x y z w). Same as tf2_echo RPY (180, 0, yaw)."""
    a = math.radians(yaw_deg) / 2.0
    return (math.cos(a), math.sin(a), 0.0, 0.0)


def gripper_yaw_deg(u, v, angle_img_deg, offset_deg=0.0):
    """block long side in the image -> link6 yaw so that the fingers close across the short side.
    Checked on the robot: block long side along base +y (image up-down) worked with link6 yaw about 0."""
    long_yaw = image_angle_to_base_yaw(u, v, angle_img_deg)
    return math.degrees(fold90(long_yaw - math.pi / 2 + math.radians(offset_deg)))


class Robot:
    def __init__(self, group, speed, node=None):
        import rclpy
        from rclpy.action import ActionClient
        from rclpy.node import Node
        from moveit_msgs.action import MoveGroup
        self.rclpy = rclpy
        self.own = node is None
        if self.own:
            rclpy.init()
            node = Node("goto_block")
        self.node = node
        self.group, self.speed = group, speed
        self.io = None
        self.cli = ActionClient(self.node, MoveGroup, "/move_action")
        if not self.cli.wait_for_server(timeout_sec=10.0):
            raise RuntimeError("move_group action /move_action not found. Is MoveIt running?")

    def move(self, x, y, z, q):
        from moveit_msgs.action import MoveGroup
        from moveit_msgs.msg import Constraints, PositionConstraint, OrientationConstraint
        from shape_msgs.msg import SolidPrimitive
        from geometry_msgs.msg import Pose
        goal = MoveGroup.Goal()
        r = goal.request
        r.group_name = self.group
        r.num_planning_attempts = 5
        r.allowed_planning_time = 5.0
        r.max_velocity_scaling_factor = self.speed
        r.max_acceleration_scaling_factor = self.speed
        r.start_state.is_diff = True
        r.workspace_parameters.header.frame_id = BASE
        pc = PositionConstraint()
        pc.header.frame_id = BASE
        pc.link_name = EEF
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = float(x), float(y), float(z)
        pose.orientation.w = 1.0
        pc.constraint_region.primitives.append(SolidPrimitive(type=SolidPrimitive.SPHERE, dimensions=[0.001]))
        pc.constraint_region.primitive_poses.append(pose)
        pc.weight = 1.0
        oc = OrientationConstraint()
        oc.header.frame_id = BASE
        oc.link_name = EEF
        oc.orientation.x, oc.orientation.y, oc.orientation.z, oc.orientation.w = q
        oc.absolute_x_axis_tolerance = oc.absolute_y_axis_tolerance = oc.absolute_z_axis_tolerance = 0.01
        oc.weight = 1.0
        r.goal_constraints.append(Constraints(position_constraints=[pc], orientation_constraints=[oc]))
        goal.planning_options.plan_only = False
        fut = self.cli.send_goal_async(goal)
        self.rclpy.spin_until_future_complete(self.node, fut)
        gh = fut.result()
        if not gh.accepted:
            print("  goal rejected")
            return False
        res = gh.get_result_async()
        self.rclpy.spin_until_future_complete(self.node, res)
        code = res.result().result.error_code.val
        print("  MoveIt result %d (%s)" % (code, "SUCCESS" if code == 1 else "failed, see move_group terminal"))
        return code == 1

    def _set(self, output, on):
        from igus_rebel_msgs.srv import SetDigitalOutput
        if self.io is None:
            self.io = self.node.create_client(SetDigitalOutput, "/set_digital_output")
            if not self.io.wait_for_service(timeout_sec=5.0):
                raise RuntimeError("/set_digital_output not available")
        req = SetDigitalOutput.Request()
        req.output.output = output
        req.output.is_on = on
        fut = self.io.call_async(req)
        self.rclpy.spin_until_future_complete(self.node, fut, timeout_sec=5.0)
        res = fut.result()
        if res is None or not res.success:
            raise RuntimeError("setting output %d failed: %s" % (output, getattr(res, "message", "no answer")))

    def gripper(self, close, seconds=GRIP_TIME):
        on, other = (OUT_CLOSE, OUT_OPEN) if close else (OUT_OPEN, OUT_CLOSE)
        self._set(other, False)
        self._set(on, True)
        time.sleep(seconds)
        self._set(on, False)
        print("  gripper %s" % ("closed" if close else "opened"))

    def shutdown(self):
        if self.own:
            self.node.destroy_node()
            self.rclpy.shutdown()


def step(text, confirm):
    print("\n>> " + text)
    if confirm:
        input("   Enter = do it, Ctrl-C = stop ")


def pick_sequence(rb, x, y, yaw_deg, grasp_z, pre_z, grip_time, confirm):
    """pre-grasp -> open -> down -> close -> up. Returns True when every step worked."""
    q = down_quat(yaw_deg)
    step("1/5 move to pre-grasp (x %.4f y %.4f z %.3f, yaw %.1f deg)" % (x, y, pre_z, yaw_deg), confirm)
    if not rb.move(x, y, pre_z, q):
        return False
    step("2/5 open gripper", confirm)
    rb.gripper(False, grip_time)
    step("3/5 move down to grasp (z %.3f). Check from above that the block is centred." % grasp_z, confirm)
    if not rb.move(x, y, grasp_z, q):
        return False
    step("4/5 close gripper", confirm)
    rb.gripper(True, grip_time)
    step("5/5 lift back to pre-grasp (z %.3f)" % pre_z, confirm)
    return rb.move(x, y, pre_z, q)


def place_sequence(rb, x, y, yaw_deg, place_z, pre_z, grip_time, confirm):
    """above the place spot (link6 turned to the place yaw) -> down -> open -> up.
    Call it right after pick_sequence, with the block still in the gripper.
    Returns True when every step worked."""
    q = down_quat(yaw_deg)
    step("1/4 move above the place spot (x %.4f y %.4f z %.3f, yaw %.1f deg)" % (x, y, pre_z, yaw_deg), confirm)
    if not rb.move(x, y, pre_z, q):
        return False
    step("2/4 move down to release height (z %.3f). Check the block is above the spot." % place_z, confirm)
    if not rb.move(x, y, place_z, q):
        return False
    step("3/4 open gripper (release the block)", confirm)
    rb.gripper(False, grip_time)
    step("4/4 lift back up (z %.3f)" % pre_z, confirm)
    return rb.move(x, y, pre_z, q)


def check_args(a):
    speed = min(max(a.speed, 0.01), MAX_SPEED)
    if speed != a.speed:
        print("speed limited to %.2f" % speed)
    if a.grasp_z > 1.0:
        print("grasp z %.3f: values are in METRES, e.g. --grasp-z 0.150 (not 150)" % a.grasp_z)
        sys.exit(1)
    if a.grasp_z < 0.10:
        print("grasp z %.3f is below 0.10 m: too low, refusing (the fingers would hit the table)" % a.grasp_z)
        sys.exit(1)
    if a.grasp_z > 0.35 or not 0.0 < a.pre_height <= 0.2:
        print("grasp z %.3f m / pre-height %.3f m look wrong (expected grasp 0.10..0.35, pre-height up to 0.20)"
              % (a.grasp_z, a.pre_height))
        sys.exit(1)
    return speed


def add_common_args(ap):
    ap.add_argument("--group", default="igus_rebel_arm", help="MoveIt planning group (default %(default)s)")
    ap.add_argument("--grasp-z", type=float, default=GRASP_Z)
    ap.add_argument("--pre-height", type=float, default=PRE_HEIGHT)
    ap.add_argument("--speed", type=float, default=SPEED)
    ap.add_argument("--grip-time", type=float, default=GRIP_TIME)
    ap.add_argument("--yaw-offset", type=float, default=GRIPPER_YAW_OFFSET,
                    help="deg between link6 x and the finger closing direction (default %(default)s)")
    ap.add_argument("--dry", action="store_true", help="only print, do not move")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("u", type=float)
    ap.add_argument("v", type=float)
    ap.add_argument("--go", choices=["pre", "grasp", "pick"], default="pre")
    ap.add_argument("--angle-img", type=float, default=None)
    ap.add_argument("--yaw-deg", type=float, default=-4.2)
    ap.add_argument("--no-confirm", action="store_true")
    add_common_args(ap)
    a = ap.parse_args()
    speed = check_args(a)

    x, y = pixel_to_base(a.u, a.v)
    yaw = gripper_yaw_deg(a.u, a.v, a.angle_img, a.yaw_offset) if a.angle_img is not None else a.yaw_deg
    zg, zp = a.grasp_z, a.grasp_z + a.pre_height
    print("block pixel  : %.1f %.1f" % (a.u, a.v))
    print("link6 x, y   : %.4f  %.4f" % (x, y))
    print("z pre / grasp: %.4f / %.4f" % (zp, zg))
    print("link6 yaw    : %.1f deg%s" % (yaw, " (from image angle %.1f)" % a.angle_img if a.angle_img is not None else ""))
    print("speed        : %.0f %%   mode: %s" % (speed * 100, a.go))
    if not in_area(x, y):
        print("WARNING: target is outside the calibrated area. Check the pixel.")
    if a.dry:
        print("dry run, robot not moved")
        return

    rb = Robot(a.group, speed)
    ok = False
    try:
        if a.go == "pre":
            ok = rb.move(x, y, zp, down_quat(yaw))
        elif a.go == "grasp":
            ok = rb.move(x, y, zg, down_quat(yaw))
        else:
            ok = pick_sequence(rb, x, y, yaw, zg, zp, a.grip_time, not a.no_confirm)
            print("\npick %s" % ("done" if ok else "STOPPED: a motion failed, the robot stays where it is"))
    except KeyboardInterrupt:
        print("\nstopped by user")
    except RuntimeError as exc:
        print("ERROR: %s" % exc)
    finally:
        rb.shutdown()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
