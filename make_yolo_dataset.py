#!/usr/bin/env python3
"""
Build the final YOLO-OBB dataset folder from your images + (reviewed) labels.

    python3 make_yolo_dataset.py --images dataset --labels labels_auto --out yolo_real

Result:
    yolo_real/
      images/train  images/val  images/test     (copies of your images)
      labels/train  labels/val  labels/test     (the matching .txt, from the CURRENT label files)
      data.yaml                                  (same class ids/names as the simulation)
      split.csv                                  (which image went where)
      preview/train ... preview/test             (every image with its final labels drawn)

Split (default 70 / 20 / 10 %):
  Images taken one after another look almost the same. If they land in both train and test,
  the test score looks better than it really is. So the images of each session are cut into
  chunks of --chunk consecutive images (default 10), and whole chunks are given to
  train / val / test at random.
  If you have 3 or more sessions you can instead keep whole sessions apart:
      --test-session day3   (all images of that session go to test, the rest is split as above)

Checks before copying: every image needs a label file (an empty one is fine), every line must be
"class x1 y1 x2 y2 x3 y3 x4 y4" with class 0..4 and values 0..1. Images not saved in the
review tool are counted and listed in the report (they keep their automatic labels).
"""
import argparse
import csv
import os
import random
import shutil
from collections import Counter, defaultdict

import cv2
import numpy as np

CLASSES = ["red", "green", "yellow", "white", "lightblue"]
COL = [(40, 40, 230), (40, 170, 40), (0, 200, 240), (240, 240, 240), (230, 200, 90)]


def find_images(root):
    out = []
    for d, _, files in os.walk(root):
        if "_review" in d or "_distractors" in d:
            continue
        for f in sorted(files):
            if f.lower().endswith((".png", ".jpg", ".jpeg")) and "_debug" not in f:
                out.append(os.path.join(d, f))
    return sorted(out)


def check_label(path):
    """returns (list of (cls, pts[4,2] normalised), errors)"""
    rows, errs = [], []
    with open(path) as f:
        for n, line in enumerate(f, 1):
            v = line.split()
            if not v:
                continue
            if len(v) != 9:
                errs.append("line %d: %d values (need 9)" % (n, len(v)))
                continue
            try:
                c = int(v[0])
                pts = np.array([float(x) for x in v[1:]]).reshape(4, 2)
            except ValueError:
                errs.append("line %d: not a number" % n)
                continue
            if not 0 <= c < len(CLASSES):
                errs.append("line %d: class %d not in 0..%d" % (n, c, len(CLASSES) - 1))
            if pts.min() < 0 or pts.max() > 1:
                errs.append("line %d: coordinate outside 0..1" % n)
            rows.append((c, pts))
    return rows, errs


def draw(img, rows):
    h, w = img.shape[:2]
    out = img.copy()
    for c, pts in rows:
        box = (pts * [w, h]).astype(np.int32)
        col = COL[c] if 0 <= c < len(COL) else (0, 0, 255)
        cv2.polylines(out, [box], True, (0, 0, 0), 4)
        cv2.polylines(out, [box], True, col, 2)
        name = CLASSES[c] if 0 <= c < len(CLASSES) else "?%d" % c
        cv2.putText(out, "%d %s" % (c, name), (int(box[:, 0].min()), int(box[:, 1].min()) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", default="yolo_real")
    ap.add_argument("--train", type=float, default=0.7)
    ap.add_argument("--val", type=float, default=0.2)
    ap.add_argument("--chunk", type=int, default=10)
    ap.add_argument("--test-session", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-preview", action="store_true")
    a = ap.parse_args()

    imgs = find_images(a.images)
    if not imgs:
        print("no images found in", a.images)
        return
    rf = os.path.join(a.labels, "_reviewed.txt")
    reviewed = set(open(rf).read().split()) if os.path.exists(rf) else set()

    items, problems, not_reviewed = [], [], []
    for p in imgs:
        rel = os.path.splitext(os.path.relpath(p, a.images))[0]
        lp = os.path.join(a.labels, rel + ".txt")
        if not os.path.exists(lp):
            problems.append("%s: no label file" % rel)
            continue
        rows, errs = check_label(lp)
        for e in errs:
            problems.append("%s: %s" % (rel, e))
        if rel not in reviewed:
            not_reviewed.append(rel)
        sess = rel.split(os.sep)[0] if os.sep in rel else "(root)"
        items.append({"img": p, "lab": lp, "rel": rel, "session": sess, "rows": rows})

    if problems:
        print("PROBLEMS (fix these first, nothing was copied):")
        for x in problems[:50]:
            print("  " + x)
        return

    # ---------- split
    rng = random.Random(a.seed)
    by_sess = defaultdict(list)
    for it in items:
        by_sess[it["session"]].append(it)
    split = {}
    pool = []
    for s, lst in sorted(by_sess.items()):
        lst.sort(key=lambda it: it["rel"])
        if a.test_session and s == a.test_session:
            for it in lst:
                split[it["rel"]] = "test"
            continue
        pool += [lst[i:i + a.chunk] for i in range(0, len(lst), a.chunk)]
    if a.test_session and a.test_session not in by_sess:
        print("session '%s' not found. Sessions: %s" % (a.test_session, ", ".join(sorted(by_sess))))
        return
    rng.shuffle(pool)
    n_pool = sum(len(c) for c in pool)
    if a.test_session:
        fr = {"train": a.train / (a.train + a.val), "val": a.val / (a.train + a.val), "test": 0.0}
    else:
        fr = {"train": a.train, "val": a.val, "test": max(0.0, 1 - a.train - a.val)}
    target = {k: v * n_pool for k, v in fr.items()}
    got = Counter()
    for ch in pool:
        # give the chunk to the split that is furthest below its target
        k = max(("train", "val", "test"), key=lambda s: (target[s] - got[s]) / max(target[s], 1e-9) if target[s] else -1)
        for it in ch:
            split[it["rel"]] = k
        got[k] += len(ch)

    # ---------- copy
    if os.path.exists(a.out):
        shutil.rmtree(a.out)
    for s in ("train", "val", "test"):
        for sub in ("images", "labels") + (() if a.no_preview else ("preview",)):
            os.makedirs(os.path.join(a.out, sub, s), exist_ok=True)
    counts = {s: Counter() for s in ("train", "val", "test")}
    n_img = Counter()
    with open(os.path.join(a.out, "split.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["split", "new_name", "original_image", "reviewed"])
        for it in items:
            s = split[it["rel"]]
            stem = it["rel"].replace(os.sep, "__")
            ext = os.path.splitext(it["img"])[1].lower()
            shutil.copy2(it["img"], os.path.join(a.out, "images", s, stem + ext))
            shutil.copy2(it["lab"], os.path.join(a.out, "labels", s, stem + ".txt"))
            if not a.no_preview:
                img = cv2.imread(it["img"])
                cv2.imwrite(os.path.join(a.out, "preview", s, stem + ".jpg"), draw(img, it["rows"]),
                            [cv2.IMWRITE_JPEG_QUALITY, 85])
            wr.writerow([s, stem + ext, it["img"], "yes" if it["rel"] in reviewed else "no"])
            n_img[s] += 1
            for c, _ in it["rows"]:
                counts[s][c] += 1

    with open(os.path.join(a.out, "data.yaml"), "w") as f:
        f.write("path: %s\n" % os.path.abspath(a.out))
        f.write("train: images/train\nval: images/val\ntest: images/test\n")
        f.write("names:\n" + "".join("  %d: %s\n" % (i, c) for i, c in enumerate(CLASSES)))

    print("dataset written to %s" % os.path.abspath(a.out))
    print("%-6s %6s   " % ("split", "images") + " ".join("%9s" % c for c in CLASSES))
    for s in ("train", "val", "test"):
        print("%-6s %6d   " % (s, n_img[s]) + " ".join("%9d" % counts[s][i] for i in range(len(CLASSES))))
    print("sessions: %s" % ", ".join("%s (%d)" % (s, len(l)) for s, l in sorted(by_sess.items())))
    if not_reviewed:
        print("NOTE: %d images were not saved in review_labels.py (automatic labels used), e.g. %s"
              % (len(not_reviewed), ", ".join(not_reviewed[:5])))
    else:
        print("all images were reviewed")
    if not a.no_preview:
        print("check the final labels by eye in %s" % os.path.join(a.out, "preview"))


if __name__ == "__main__":
    main()
