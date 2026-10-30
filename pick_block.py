#!/usr/bin/env python3
"""
Pick a block found by the detector, and optionally place it at a fixed spot.
Reads /detector/detections (pixel + image angle), converts with homography_calib.py.

  pick:   pre-grasp, open, down, close, up
  place:  above the place spot, turn link6 to the place yaw, down, open, up

Needs in the same folder: homography_calib.py, goto_block.py. The detector node must be running,
and the robot arm must be OUT of the camera view when the detection is read.

    python3 pick_block.py                                    list blocks, you type the id, pick only, pause before every step
    python3 pick_block.py --color red                        pick the red block (most confident one)
    python3 pick_block.py --color red --mode place           pick the red block and place it at the place spot
    python3 pick_block.py --id 2 --mode place --step direct  pick block #2 and place it, no pauses
    python3 pick_block.py --color red --mode place --dry     only print the targets, no motion

Place spot (pixel in the camera image, same homography as the pick):
    --place-u 474 --place-v 113      default (474, 113)
    --place-angle 0                  block angle in the image at the place, same convention as the
                                     detector (0 = block standing like "1" in the image)
    --place-z                        link6 z at release, default = grasp z + 0.003 m (block does not get pressed)

Other options: --grasp-z 0.129  --pre-height 0.08  --speed 0.05  --grip-time 3.0
               --yaw-offset (deg between link6 x and the finger closing direction, default in goto_block.py)
               --min-conf 0.5  --timeout 5  --group (default igus_rebel_arm, set in goto_block.py)
"""
import argparse
import json
import sys
import time

from homography_calib import pixel_to_base, in_area
from goto_block import Robot, pick_sequence, place_sequence, gripper_yaw_deg, add_common_args, check_args

PLACE_U, PLACE_V, PLACE_ANGLE = 577.0, 104.0, 0.0


def read_detections(node, topic, timeout):
    """wait for a detection message that is newer than the moment we started"""
    import rclpy
    from std_msgs.msg import String
    box = {}
    t_start = time.time()
    node.create_subscription(String, topic, lambda m: box.__setitem__('m', m), 10)
    while time.time() - t_start < timeout:
        rclpy.spin_once(node, timeout_sec=0.1)
        if 'm' in box:
            d = json.loads(box['m'].data)
            if time.time() - t_start > 0.3:       # skip a message that may be from before we started
                return d
            box.pop('m')
    return None


def wrap180(a):
    """angle into [-180, 180)"""
    return (a + 180.0) % 360.0 - 180.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sel = ap.add_mutually_exclusive_group()
    sel.add_argument("--color", help="red, green, yellow, white, lightblue")
    sel.add_argument("--id", type=int, help="block number shown in the detector window")
    ap.add_argument("--mode", choices=["pick", "place"], default="pick",
                    help="pick = pick only (default), place = pick and then place at the place spot")
    ap.add_argument("--step", choices=["pause", "direct"], default="pause",
                    help="pause = Enter before every step (default), direct = straight through")
    ap.add_argument("--direct", action="store_true", help="same as --step direct (kept for old commands)")
    ap.add_argument("--place-u", type=float, default=PLACE_U, help="place pixel u (default %.0f)" % PLACE_U)
    ap.add_argument("--place-v", type=float, default=PLACE_V, help="place pixel v (default %.0f)" % PLACE_V)
    ap.add_argument("--place-angle", type=float, default=PLACE_ANGLE,
                    help="block image angle at the place, detector convention (default %.0f)" % PLACE_ANGLE)
    ap.add_argument("--place-yaw", type=float, default=None,
                    help="link6 yaw at the place in deg, overrides --place-angle (e.g. 0.8 = yaw that picks a '1' block)")
    ap.add_argument("--place-z", type=float, default=None, help="link6 z at release (default grasp z + 0.003)")
    ap.add_argument("--allow-outside", action="store_true",
                    help="place even if the place pixel is outside the calibrated area (extrapolated, less accurate)")
    ap.add_argument("--topic", default="/detector/detections")
    ap.add_argument("--min-conf", type=float, default=0.5)
    ap.add_argument("--timeout", type=float, default=5.0)
    add_common_args(ap)
    a = ap.parse_args()
    speed = check_args(a)
    direct = a.direct or a.step == "direct"
    do_place = a.mode == "place"

    if do_place and a.place_z is not None and a.place_z < 0.10:
        print("place z %.3f is below 0.10 m: too low, refusing (the fingers would hit the table)" % a.place_z)
        sys.exit(1)

    import rclpy
    from rclpy.node import Node
    rclpy.init()
    node = Node("pick_block")
    ok = False
    try:
        det = read_detections(node, a.topic, a.timeout)
        if det is None:
            print("no detections on %s within %.0f s. Is detector_node running?" % (a.topic, a.timeout))
            return
        blocks = [b for b in det["blocks"] if b["conf"] >= a.min_conf]
        if not blocks:
            print("no blocks detected (conf >= %.2f)" % a.min_conf)
            return
        print("id  class       conf   pixel u, v       angle   ->  x, y (m)           gripper yaw")
        for b in blocks:
            b["x"], b["y"] = pixel_to_base(b["u"], b["v"])
            b["yaw"] = gripper_yaw_deg(b["u"], b["v"], b["angle_img"], a.yaw_offset)
            flag = "" if in_area(b["x"], b["y"]) else "  OUTSIDE calibrated area"
            print("%2d  %-10s  %.2f   %6.1f, %6.1f   %+6.1f   ->  %.4f, %+.4f   %+6.1f deg%s" % (
                b["id"], b["name"], b["conf"], b["u"], b["v"], b["angle_img"], b["x"], b["y"], b["yaw"], flag))

        # ---- choose
        if a.color:
            cand = [b for b in blocks if b["name"] == a.color.lower()]
            if not cand:
                print("no %s block found" % a.color)
                return
            if len(cand) > 1:
                print("%d %s blocks, taking the most confident one" % (len(cand), a.color))
            target = max(cand, key=lambda b: b["conf"])
        elif a.id is not None:
            cand = [b for b in blocks if b["id"] == a.id]
            if not cand:
                print("no block with id %d" % a.id)
                return
            target = cand[0]
        else:
            s = input("\nid of the block to pick (Enter = cancel): ").strip()
            if not s:
                return
            cand = [b for b in blocks if str(b["id"]) == s]
            if not cand:
                print("no block with id %s" % s)
                return
            target = cand[0]

        if not in_area(target["x"], target["y"]):
            print("target is outside the calibrated area, not picking")
            return
        zg, zp = a.grasp_z, a.grasp_z + a.pre_height
        print("\nPICK #%d %s: link6 x %.4f y %.4f, yaw %.1f deg, z pre %.3f grasp %.3f, speed %.0f %%, %s" % (
            target["id"], target["name"], target["x"], target["y"], target["yaw"], zp, zg, speed * 100,
            "direct" if direct else "pause before each step"))

        # ---- place target, same homography and same angle convention as the pick
        place = None
        if do_place:
            px, py = pixel_to_base(a.place_u, a.place_v)
            if not in_area(px, py):
                print("place spot (%.0f, %.0f) -> x %.4f y %+.4f is OUTSIDE the calibrated area "
                      "(homography is extrapolated there)" % (a.place_u, a.place_v, px, py))
                if not a.allow_outside:
                    print("not starting. Add calibration dots near the place spot, or use --allow-outside "
                          "after checking x, y on the robot")
                    return
                print("--allow-outside given, continuing. Check the place pose with pause mode!")
            if a.place_yaw is not None:
                pyaw = a.place_yaw          # link6 yaw given directly, no image-angle conversion
            else:
                # same function as the pick, so the yaw is folded into -90..+90 like every pick yaw
                pyaw = gripper_yaw_deg(a.place_u, a.place_v, a.place_angle, a.yaw_offset)
            pz = a.place_z if a.place_z is not None else zg + 0.003
            place = (px, py, pyaw, pz, pz + a.pre_height)
            print("PLACE at pixel %.0f, %.0f: link6 x %.4f y %.4f, yaw %.1f deg (link6 turns %+.1f deg), "
                  "z pre %.3f release %.3f" % (a.place_u, a.place_v, px, py, pyaw,
                                               wrap180(pyaw - target["yaw"]), place[4], pz))

        if a.dry:
            print("dry run, robot not moved")
            ok = True
            return

        rb = Robot(a.group, speed, node=node)
        ok = pick_sequence(rb, target["x"], target["y"], target["yaw"], zg, zp, a.grip_time, not direct)
        if not ok:
            print("\npick STOPPED: a motion failed, the robot stays where it is")
        else:
            print("\npick done")
            if place is not None:
                px, py, pyaw, pz, ppre = place
                ok = place_sequence(rb, px, py, pyaw, pz, ppre, a.grip_time, not direct)
                print("\nplace %s" % ("done" if ok else "STOPPED: a motion failed, the robot stays where it is"))
    except KeyboardInterrupt:
        print("\nstopped by user")
    except RuntimeError as exc:
        print("ERROR: %s" % exc)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
