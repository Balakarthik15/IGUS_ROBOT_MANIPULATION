#!/usr/bin/env python3
"""
Automatic OBB labels (YOLO-OBB) for the 5 target blocks + the distractors, for real camera images.

Class ids are the same as in the simulation dataset:
    0 red   1 green   2 yellow   3 white   4 lightblue

Run on all images:
    python3 auto_label_obb.py --images dataset --labels labels_auto
Check one image (saves its masks and prints what was found):
    python3 auto_label_obb.py --debug dataset/day1/img_0042.png
Then correct by hand where needed:
    python3 review_labels.py --images dataset --labels labels_auto

Output (same sub-folders as the images):
    labels_auto/<session>/<image>.txt                 targets, YOLO-OBB: id x1 y1 x2 y2 x3 y3 x4 y4 (0..1)
    labels_auto/_distractors/<session>/<image>.txt    distractors, same corners but with a NAME
                                                      instead of an id, e.g. d_2x2_skyblue
                                                      (not used for training, only for review and stats)
    labels_auto/_review/<session>__<image>.jpg        boxes drawn (targets thick, distractors thin)
    labels_auto/_review/review.csv                    counts per image + "check" notes
    labels_auto/data.yaml, classes.txt

Why it also works when the light changes: every image is first colour-corrected so that
the brown board looks the same as in a normal image (the board is the reference). Then the
colour rules and the brick-size rules are applied, all in mm via the camera scale.
"""
import argparse
import csv
import os

import cv2
import numpy as np

CLASSES = ["red", "green", "yellow", "white", "lightblue"]
TARGET_COLOUR = {"red": 0, "green": 1, "yellow": 2, "white": 3, "lightblue": 4}

REF_BOARD_BGR = np.array([92.0, 114.0, 155.0])   # median board colour in a well lit image
BOARD_U = (388, 752)          # board columns in the 1280 x 720 image (outer limit)
MM_PER_PX = 1.17

# colour rules on the corrected image (OpenCV HSV: H 0..179). First match wins.
# (name, H range or None, S range, V range)
COLOUR_RULES = [
    ("black",       None,       (0, 255),   (0, 60)),
    ("white",       None,       (0, 24),    (175, 255)),
    ("lightblue",   (70, 104),  (24, 100),  (165, 255)),
    ("red",         (168, 179), (130, 255), (60, 255)),
    ("red",         (0, 6),     (130, 255), (60, 255)),
    ("yellow",      (13, 24),   (140, 255), (140, 255)),
    ("paleyellow",  (16, 34),   (45, 140),  (185, 255)),
    ("paleyellow",  (24, 34),   (140, 255), (170, 255)),
    ("lime",        (33, 50),   (80, 255),  (90, 255)),
    ("green",       (51, 90),   (65, 255),  (50, 255)),
    ("skyblue",     (85, 97),   (120, 255), (100, 255)),
    ("blue",        (97, 125),  (120, 255), (60, 255)),
]

# brick shapes as the camera sees them: the footprint plus a bit of the side faces
# (up to about 10 mm, depending on where the block is). (name, long mm range, long/short ratio range)
SHAPES = [
    ("2x2",      (24, 52),  (1.0, 1.3)),
    ("plate4x4", (56, 88),  (1.0, 1.3)),
    ("2x3",      (42, 64),  (1.3, 1.62)),
    ("2x4",      (58, 84),  (1.62, 2.7)),
    ("2x6",      (86, 120), (2.4, 4.0)),
]
MIN_FILL = 0.62


# ------------------------------------------------------------------ colour correction
def board_colour(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, np.array((3, 35, 25), np.uint8), np.array((20, 200, 255), np.uint8))
    m[:, :BOARD_U[0] + 15] = 0
    m[:, BOARD_U[1] - 15:] = 0
    px = img[m > 0]
    if len(px) < 5000:
        return None
    return np.median(px, axis=0).astype(np.float64)


def normalise(img):
    """scale each colour channel so the board matches the reference board colour"""
    b = board_colour(img)
    if b is None:
        return img.copy(), np.ones(3)
    gain = np.clip(REF_BOARD_BGR / np.maximum(b, 1.0), 0.4, 4.0)
    out = np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    return out, gain


def board_region(hsv):
    h, w = hsv.shape[:2]
    m = cv2.inRange(hsv, np.array((4, 50, 80), np.uint8), np.array((18, 160, 225), np.uint8))
    m[:, :BOARD_U[0]] = 0
    m[:, BOARD_U[1]:] = 0
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    region = np.zeros((h, w), np.uint8)
    if not cnts:
        region[:, BOARD_U[0]:BOARD_U[1]] = 255
        return region
    cv2.drawContours(region, [cv2.convexHull(max(cnts, key=cv2.contourArea))], -1, 255, -1)
    return cv2.erode(region, np.ones((9, 9), np.uint8))


def colour_masks(hsv, roi):
    masks = {}
    taken = np.zeros(hsv.shape[:2], bool)
    for name, hr, sr, vr in COLOUR_RULES:
        lo = (hr[0] if hr else 0, sr[0], vr[0])
        hi = (hr[1] if hr else 179, sr[1], vr[1])
        m = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8)) > 0
        m &= ~taken
        taken |= m
        masks[name] = masks.get(name, np.zeros_like(m)) | m
    out = {}
    for name, m in masks.items():
        m = (m & (roi > 0)).astype(np.uint8) * 255
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        out[name] = m
    return out


def shape_of(long_mm, short_mm):
    ratio = long_mm / max(short_mm, 1e-6)
    for name, (l0, l1), (r0, r1) in SHAPES:
        if l0 <= long_mm <= l1 and r0 <= ratio <= r1:
            return name
    return "unknown"


def overlap(r1, r2):
    a = cv2.rotatedRectangleIntersection(r1, r2)
    if a[0] == cv2.INTERSECT_NONE or a[1] is None:
        return 0.0
    inter = cv2.contourArea(cv2.convexHull(a[1]))
    return inter / max(1e-6, min(r1[1][0] * r1[1][1], r2[1][0] * r2[1][1]))


# colours that can show up on the SAME block (edges, highlights): never split between them,
# decide the colour from the median of the whole block instead
GROUP = {"white": "pale", "lightblue": "pale", "yellow": "yel", "paleyellow": "yel",
         "skyblue": "blu", "blue": "blu"}


def pick_colour(group, votes, hsv, region):
    if group not in ("pale", "yel", "blu"):
        return group
    h, s_, v = np.median(hsv[region], axis=0)       # whole block: very bright pixels lose their tint
    if group == "pale":
        # white: S about 4-11, lightblue: S about 28-38 with a cyan hue (about 85-90)
        return "lightblue" if (s_ >= 18 and 74 <= h <= 102) else "white"
    if group == "yel":
        return "yellow" if (s_ >= 125 and h <= 24) else "paleyellow"
    return "skyblue" if h < 97 else "blue"


def blob_info(mask_c):
    cnts, _ = cv2.findContours(mask_c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    rect = cv2.minAreaRect(c)
    a, b = rect[1]
    if area < 300 or min(a, b) <= 0:
        return None
    shape = shape_of(max(a, b) * MM_PER_PX, min(a, b) * MM_PER_PX)
    return rect, shape, area / (a * b), area


def detect(img):
    """returns targets [(id, rect)], distractors [(name, rect)], notes [str], masks, corrected image"""
    norm, gain = normalise(img)
    # dark images were brightened, which also brightens the sensor noise: smooth more then
    smooth = cv2.medianBlur(norm, 5) if gain.max() > 1.8 else cv2.GaussianBlur(norm, (3, 3), 0)
    hsv = cv2.cvtColor(smooth, cv2.COLOR_BGR2HSV)
    roi = board_region(hsv)
    # pixels that look like the board are never part of a block (kills noise and shadows)
    lab_img = cv2.cvtColor(smooth, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref = cv2.cvtColor(REF_BOARD_BGR.reshape(1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0]
    d_ab = np.linalg.norm(lab_img[:, :, 1:] - ref[1:], axis=2)
    d_l = lab_img[:, :, 0] - ref[0]
    board_like = (d_ab < 14) & (d_l > -45) & (d_l < 28)
    roi = roi & (~board_like).astype(np.uint8) * 255
    masks = colour_masks(hsv, roi)

    # 1) all coloured pixels together = objects (a block whose top, sides and highlights
    #    fall into different colour rules is still one object)
    obj = np.zeros(img.shape[:2], np.uint8)
    for m in masks.values():
        obj |= m
    obj = cv2.morphologyEx(obj, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab = cv2.connectedComponents(obj)

    found = []                                   # (colour, rect, shape, fill, area)
    for i in range(1, n):
        comp = (lab == i)
        size = int(comp.sum())
        if size < 300:
            continue
        votes = {c: int((m[comp] > 0).sum()) for c, m in masks.items()}
        gvotes = {}
        for c, v in votes.items():
            gvotes[GROUP.get(c, c)] = gvotes.get(GROUP.get(c, c), 0) + v
        gdom = max(gvotes, key=gvotes.get)
        info = blob_info(comp.astype(np.uint8) * 255)
        if info is None:
            continue
        rect, shape, fill, area = info
        share = gvotes[gdom] / max(1, sum(gvotes.values()))
        if shape != "unknown" and share >= 0.6:
            found.append((pick_colour(gdom, votes, hsv, comp), rect, shape, fill, area))
            continue
        # probably two different blocks touching: split by colour group
        split = False
        for g, v in gvotes.items():
            if v < 250 or v < 0.2 * size:
                continue
            part = np.zeros(img.shape[:2], bool)
            for c, m in masks.items():
                if GROUP.get(c, c) == g:
                    part |= (m > 0)
            part &= comp
            part8 = cv2.morphologyEx(part.astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
            pi = blob_info(part8)
            if pi:
                found.append((pick_colour(g, votes, hsv, part8 > 0), pi[0], pi[1], pi[2], pi[3]))
                split = True
        if not split:
            found.append((pick_colour(gdom, votes, hsv, comp), rect, shape, fill, area))

    targets, distractors, notes = [], [], []
    for colour, rect, shape, fill, area in found:
        if colour in TARGET_COLOUR and shape == "2x4" and fill >= MIN_FILL:
            targets.append((TARGET_COLOUR[colour], rect))
        else:
            name = "d_%s_%s" % (shape, colour)
            if colour in TARGET_COLOUR and shape == "2x4":
                name = "d_step_2x4_%s" % colour       # 2x4 size but not a full rectangle
                notes.append("%s 2x4-sized but fill %.2f: target or stepped brick?" % (colour, fill))
            if shape == "unknown":
                (cx, cy), (a, b), _ = rect
                notes.append("unknown %s blob %.0fx%.0f mm at %.0f,%.0f (touching blocks?)" % (
                    colour, max(a, b) * MM_PER_PX, min(a, b) * MM_PER_PX, cx, cy))
            if colour == "lime" and shape == "2x4":
                name = "d_step_2x4_lime"
            distractors.append((name, rect))
    if np.any(np.abs(gain - 1) > 0.6):
        notes.append("strong light correction (gain %.1f %.1f %.1f)" % tuple(gain))
    ids = [t[0] for t in targets]
    for k in set(ids):
        if ids.count(k) > 1:
            notes.append("%s found %d times" % (CLASSES[k], ids.count(k)))
    return targets, distractors, notes, masks, norm


# ------------------------------------------------------------------ files
def corners_norm(rect, w, h):
    pts = cv2.boxPoints(rect)
    pts[:, 0] = np.clip(pts[:, 0] / w, 0, 1)
    pts[:, 1] = np.clip(pts[:, 1] / h, 0, 1)
    return " ".join("%.6f %.6f" % (x, y) for x, y in pts)


def find_images(root):
    out = []
    for d, _, files in os.walk(root):
        if "_review" in d or "_distractors" in d:
            continue
        for f in sorted(files):
            if f.lower().endswith((".png", ".jpg", ".jpeg")) and "_debug" not in f:
                out.append(os.path.join(d, f))
    return sorted(out)


TARGET_DRAW = [(40, 40, 230), (40, 170, 40), (0, 200, 240), (240, 240, 240), (230, 200, 90)]


def draw(img, targets, distractors):
    out = img.copy()
    for name, r in distractors:
        box = cv2.boxPoints(r).astype(np.int32)
        cv2.polylines(out, [box], True, (255, 0, 255), 1)
        cv2.putText(out, name, (int(box[:, 0].max()) + 3, int(box[:, 1].max())),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
    for k, r in targets:
        box = cv2.boxPoints(r).astype(np.int32)
        cv2.polylines(out, [box], True, (0, 0, 0), 4)
        cv2.polylines(out, [box], True, TARGET_DRAW[k], 2)
        cv2.putText(out, "%d %s" % (k, CLASSES[k]), (int(box[:, 0].min()), int(box[:, 1].min()) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(out, "%d %s" % (k, CLASSES[k]), (int(box[:, 0].min()), int(box[:, 1].min()) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, TARGET_DRAW[k], 1)
    return out


def debug_view(path):
    img = cv2.imread(path)
    targets, distractors, notes, masks, norm = detect(img)
    tiles = [draw(img, targets, distractors)[:, BOARD_U[0]:BOARD_U[1]], norm[:, BOARD_U[0]:BOARD_U[1]].copy()]
    cv2.putText(tiles[1], "corrected", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    for name in ["red", "green", "yellow", "white", "lightblue", "paleyellow", "lime", "skyblue", "blue", "black"]:
        t = cv2.cvtColor(masks.get(name, np.zeros(img.shape[:2], np.uint8)), cv2.COLOR_GRAY2BGR)[:, BOARD_U[0]:BOARD_U[1]]
        cv2.putText(t, name, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        tiles.append(t)
    rows = [np.hstack(tiles[i:i + 6]) for i in (0, 6)]
    rows[1] = np.hstack([rows[1], np.zeros((rows[1].shape[0], rows[0].shape[1] - rows[1].shape[1], 3), np.uint8)])
    name = os.path.splitext(os.path.basename(path))[0] + "_debug.jpg"
    cv2.imwrite(name, np.vstack(rows))
    print("saved %s (in the current folder)" % name)
    for k, r in targets:
        print("  target     %-10s centre %4.0f,%4.0f  %.0fx%.0f mm" % (
            CLASSES[k], r[0][0], r[0][1], max(r[1]) * MM_PER_PX, min(r[1]) * MM_PER_PX))
    for n, r in distractors:
        print("  distractor %-22s centre %4.0f,%4.0f  %.0fx%.0f mm" % (
            n, r[0][0], r[0][1], max(r[1]) * MM_PER_PX, min(r[1]) * MM_PER_PX))
    for n in notes:
        print("  note: " + n)


def write_yaml(labels_dir):
    with open(os.path.join(labels_dir, "classes.txt"), "w") as f:
        f.write("\n".join(CLASSES) + "\n")
    with open(os.path.join(labels_dir, "data.yaml"), "w") as f:
        f.write("# real images; set path/train/val to your split folders\n")
        f.write("names:\n" + "".join("  %d: %s\n" % (i, c) for i, c in enumerate(CLASSES)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images")
    ap.add_argument("--labels", default="labels_auto")
    ap.add_argument("--debug", help="one image: save masks, print what was found")
    ap.add_argument("--overwrite", action="store_true",
                    help="also overwrite label files you already corrected by hand (marked in _reviewed.txt)")
    a = ap.parse_args()
    if a.debug:
        debug_view(a.debug)
        return
    if not a.images:
        ap.error("--images is required")

    done_by_hand = set()
    rf = os.path.join(a.labels, "_reviewed.txt")
    if os.path.exists(rf) and not a.overwrite:
        done_by_hand = set(open(rf).read().split())

    rev = os.path.join(a.labels, "_review")
    os.makedirs(rev, exist_ok=True)
    rows, skipped = [], 0
    for p in find_images(a.images):
        rel = os.path.splitext(os.path.relpath(p, a.images))[0]
        if rel in done_by_hand:
            skipped += 1
            continue
        img = cv2.imread(p)
        if img is None:
            continue
        h, w = img.shape[:2]
        targets, distractors, notes, _, _ = detect(img)
        for base, lines in ((a.labels, ["%d %s" % (k, corners_norm(r, w, h)) for k, r in targets]),
                            (os.path.join(a.labels, "_distractors"),
                             ["%s %s" % (n, corners_norm(r, w, h)) for n, r in distractors])):
            lp = os.path.join(base, rel + ".txt")
            os.makedirs(os.path.dirname(lp), exist_ok=True)
            with open(lp, "w") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))
        counts = [sum(1 for k, _ in targets if k == i) for i in range(len(CLASSES))]
        rows.append([rel] + counts + [len(distractors), "; ".join(notes)])
        cv2.imwrite(os.path.join(rev, rel.replace(os.sep, "__") + ".jpg"), draw(img, targets, distractors),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])

    with open(os.path.join(rev, "review.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["image"] + CLASSES + ["distractors", "check"])
        wr.writerows(rows)
    write_yaml(a.labels)
    tot = np.sum([r[1:1 + len(CLASSES)] for r in rows], axis=0) if rows else np.zeros(len(CLASSES), int)
    print("labelled %d images (%d skipped: already corrected by hand)" % (len(rows), skipped))
    for i, (c, n) in enumerate(zip(CLASSES, tot)):
        print("  %d %-10s %d" % (i, c, n))
    print("images with a note to check: %d  (%s)" % (sum(1 for r in rows if r[-1]), os.path.join(rev, "review.csv")))


if __name__ == "__main__":
    main()
