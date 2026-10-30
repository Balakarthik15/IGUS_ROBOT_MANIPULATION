#!/usr/bin/env python3
"""
Check a YOLO dataset for gaps and mistakes before training.

Works in 2 stages:
  Before labelling (images only):
      python3 check_dataset.py --images dataset
  After labelling (YOLO txt labels, normal boxes or OBB):
      python3 check_dataset.py --images dataset --labels labels \
          --classes red_block,green_block,white_block,orange_block,mint_block --draw 30

--images   folder with images, searched recursively (sub-folders = sessions)
--labels   folder with the .txt files. Same relative path and name as the image.
           If left out, the script also looks for the .txt next to the image and
           in the usual YOLO layout (.../images/... -> .../labels/...).
--draw N   saves N random images with the labels drawn on them, to check by eye.

Everything goes to the folder given by --out (default: dataset_report):
  report.txt          all numbers and warnings
  class_counts.png    instances per class
  targets_per_image.png
  positions.png       where each class appears on the board (one panel per class)
  brightness.png      image brightness per session
  drawn/              images with labels drawn (--draw)
"""
import argparse
import os
import random
from collections import Counter, defaultdict

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp")
CAL_U, CAL_V = (410, 720), (95, 635)   # area covered by the camera calibration (full 1280 x 720 image)
MM_PER_PX = 1.17
INK, MUTED, BAR = "#2b2b2b", "#8a8a8a", "#3d6fb0"


# ------------------------------------------------------------------ helpers
def find_images(root):
    out = []
    for d, _, files in os.walk(root):
        for f in sorted(files):
            if f.lower().endswith(IMG_EXT):
                out.append(os.path.join(d, f))
    return sorted(out)


def session_of(path, root):
    rel = os.path.relpath(path, root)
    parts = rel.split(os.sep)
    parts = [p for p in parts[:-1] if p not in ("images", "train", "val", "test")]
    return parts[0] if parts else "(root)"


def label_path(img, img_root, lab_root):
    stem = os.path.splitext(img)[0]
    cands = []
    if lab_root:
        rel = os.path.splitext(os.path.relpath(img, img_root))[0]
        cands.append(os.path.join(lab_root, rel + ".txt"))
        cands.append(os.path.join(lab_root, os.path.basename(stem) + ".txt"))
    cands.append(stem + ".txt")
    if os.sep + "images" + os.sep in img:
        cands.append(stem.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep) + ".txt")
    for c in cands:
        if os.path.exists(c):
            return c
    return None


def dhash(gray):
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def parse_label_line(vals, w, h):
    """returns (cls, cx, cy, long_px, short_px, angle_deg or None, corners) in pixels, or raises ValueError"""
    cls = int(float(vals[0]))
    nums = [float(v) for v in vals[1:]]
    if any(v < -0.01 or v > 1.01 for v in nums):
        raise ValueError("coordinate outside 0..1")
    if len(nums) == 4:                      # normal box: cx cy bw bh
        cx, cy, bw, bh = nums
        if bw <= 0 or bh <= 0:
            raise ValueError("zero size box")
        a, b = bw * w, bh * h
        x0, y0 = cx * w - a / 2, cy * h - b / 2
        pts = np.array([[x0, y0], [x0 + a, y0], [x0 + a, y0 + b], [x0, y0 + b]])
        return cls, cx * w, cy * h, max(a, b), min(a, b), None, pts
    if len(nums) == 8:                      # OBB: 4 corners
        pts = np.array(nums).reshape(4, 2) * [w, h]
        (cx, cy), (a, b), ang = cv2.minAreaRect(pts.astype(np.float32))
        if a <= 0 or b <= 0:
            raise ValueError("zero size box")
        e1 = pts[1] - pts[0]
        e2 = pts[2] - pts[1]
        e = e1 if np.linalg.norm(e1) >= np.linalg.norm(e2) else e2
        long_ang = float(np.degrees(np.arctan2(e[1], e[0]))) % 180.0
        return cls, cx, cy, max(a, b), min(a, b), long_ang, pts
    raise ValueError("expected 5 or 9 numbers per line, got %d" % (len(nums) + 1))


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=9)
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.8)
    ax.set_axisbelow(True)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels")
    ap.add_argument("--classes", default="", help="class names in id order, comma separated")
    ap.add_argument("--out", default="dataset_report")
    ap.add_argument("--draw", type=int, default=0)
    a = ap.parse_args()

    names = [c.strip() for c in a.classes.split(",") if c.strip()]
    cname = lambda i: names[i] if i < len(names) else "class_%d" % i  # noqa: E731
    os.makedirs(a.out, exist_ok=True)
    rep = []
    warn = []
    say = rep.append

    imgs = find_images(a.images)
    if not imgs:
        print("no images found in", a.images)
        return
    say("images found: %d" % len(imgs))

    # ---------------- image checks
    sizes = Counter()
    sess_count = Counter()
    bright = defaultdict(list)
    blur = []
    hashes = []
    over = []
    mean_img = None
    n_mean = 0
    for p in imgs:
        im = cv2.imread(p)
        if im is None:
            warn.append("cannot read %s" % p)
            continue
        h, w = im.shape[:2]
        sizes[(w, h)] += 1
        s = session_of(p, a.images)
        sess_count[s] += 1
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        bright[s].append(float(g.mean()))
        blur.append((float(cv2.Laplacian(g, cv2.CV_64F).var()), p))
        hashes.append((dhash(g), p))
        over.append((float((g >= 250).mean()) * 100, p))
        if n_mean < 40 and (w, h) == (1280, 720):
            f = im.astype(np.float32)
            mean_img = f if mean_img is None else mean_img + f
            n_mean += 1

    say("\n== images per session ==")
    for s, n in sorted(sess_count.items()):
        say("  %-30s %4d   brightness mean %.0f (min %.0f, max %.0f)" % (
            s, n, np.mean(bright[s]), np.min(bright[s]), np.max(bright[s])))
    if len(sess_count) < 3:
        warn.append("only %d session(s). Use at least 3 (different times of day) and keep one whole session for test." % len(sess_count))
    all_b = np.concatenate([bright[s] for s in bright])
    if all_b.max() - all_b.min() < 25:
        warn.append("brightness hardly varies (%.0f to %.0f). Add sessions with other light." % (all_b.min(), all_b.max()))

    say("\n== image size ==")
    for (w, h), n in sizes.items():
        say("  %dx%d: %d" % (w, h, n))
    if len(sizes) > 1 or (1280, 720) not in sizes:
        warn.append("not all images are 1280x720. The calibration and detection must use the same size.")

    blur.sort()
    med = np.median([b for b, _ in blur])
    blurry = [(b, p) for b, p in blur if b < 0.4 * med]
    say("\n== sharpness (variance of Laplacian, median %.0f) ==" % med)
    say("  possibly blurred images: %d" % len(blurry))
    for b, p in blurry[:10]:
        say("    %.0f  %s" % (b, p))

    over.sort(reverse=True)
    say("\n== overexposure (pixels at 250 or more) ==")
    for v, p in over[:5]:
        say("  %.2f %%  %s" % (v, p))
    if over and over[0][0] > 2.0:
        warn.append("some images have more than 2 %% fully white pixels. The white block may lose detail: lower exposure.")

    dups = []
    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            if bin(hashes[i][0] ^ hashes[j][0]).count("1") <= 2:
                dups.append((hashes[i][1], hashes[j][1]))
    say("\n== near duplicates (almost the same image) ==")
    say("  pairs: %d" % len(dups))
    for p1, p2 in dups[:15]:
        say("    %s  ~  %s" % (os.path.basename(p1), os.path.basename(p2)))
    if len(dups) > 0.1 * len(imgs):
        warn.append("many near-duplicate images (%d pairs). Move blocks more between shots." % len(dups))

    # brightness plot
    ss = sorted(bright)
    fig, ax = plt.subplots(figsize=(max(5, 1.2 * len(ss) + 2), 3.4))
    for i, s in enumerate(ss):
        y = np.array(bright[s])
        x = i + (np.random.default_rng(0).random(len(y)) - 0.5) * 0.35
        ax.scatter(x, y, s=10, color=BAR, alpha=0.6, linewidths=0)
        ax.hlines(np.median(y), i - 0.3, i + 0.3, color=INK, linewidth=2)
    ax.set_xticks(range(len(ss)))
    ax.set_xticklabels(ss, rotation=20, ha="right")
    ax.set_ylabel("mean brightness (0-255)", color=INK)
    ax.set_title("Image brightness per session (line = median)", color=INK, fontsize=11, loc="left")
    style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "brightness.png"), dpi=130)
    plt.close(fig)

    # ---------------- label checks
    if a.labels is not None or any(label_path(p, a.images, None) for p in imgs[:20]):
        lab_root = a.labels
        inst = Counter()
        img_with = Counter()
        per_img = Counter()
        pos = defaultdict(list)
        dims = defaultdict(list)
        angles = defaultdict(list)
        min_dist = []
        missing, empty, bad = [], [], []
        outside = 0
        labelled = {}
        for p in imgs:
            lp = label_path(p, a.images, lab_root)
            if lp is None:
                missing.append(p)
                continue
            im_h, im_w = 720, 1280
            sz = cv2.imread(p)
            if sz is not None:
                im_h, im_w = sz.shape[:2]
            rows = []
            with open(lp) as f:
                for k, line in enumerate(f):
                    v = line.split()
                    if not v:
                        continue
                    try:
                        rows.append(parse_label_line(v, im_w, im_h))
                    except ValueError as exc:
                        bad.append("%s line %d: %s" % (lp, k + 1, exc))
            labelled[p] = rows
            if not rows:
                empty.append(p)
            per_img[len(rows)] += 1
            seen = set()
            for cls, cx, cy, lo, sh, ang, _ in rows:
                inst[cls] += 1
                seen.add(cls)
                pos[cls].append((cx, cy))
                dims[cls].append((lo, sh))
                if ang is not None:
                    angles[cls].append(ang)
                if not (CAL_U[0] <= cx <= CAL_U[1] and CAL_V[0] <= cy <= CAL_V[1]):
                    outside += 1
            for c in seen:
                img_with[c] += 1
            if len(rows) >= 2:
                c = np.array([[r[1], r[2]] for r in rows])
                d = np.linalg.norm(c[:, None] - c[None], axis=2)
                d[d == 0] = 1e9
                min_dist.append(d.min() * MM_PER_PX)

        say("\n== labels ==")
        say("  images with a label file: %d, without: %d, empty label file: %d" % (
            len(labelled), len(missing), len(empty)))
        if missing:
            warn.append("%d images have no label file (first: %s). Distractor-only images still need an EMPTY .txt." % (
                len(missing), missing[0]))
        if bad:
            warn.append("%d broken label lines, see report" % len(bad))
            for b in bad[:20]:
                say("    BAD " + b)

        classes = sorted(set(inst) | set(range(len(names))))
        say("\n== instances per class ==")
        for c in classes:
            say("  %-14s %4d boxes in %4d images" % (cname(c), inst[c], img_with[c]))
        if inst:
            lo, hi = min(inst[c] for c in classes), max(inst[c] for c in classes)
            if lo < 0.6 * hi:
                warn.append("classes are unbalanced (%d to %d boxes). Add images with the rare class." % (lo, hi))

        say("\n== targets per image ==")
        for k in sorted(per_img):
            say("  %d targets: %d images" % (k, per_img[k]))
        if len(per_img) <= 2:
            warn.append("the number of targets per image hardly varies. Add images with 0, 1, 2 and 3 targets.")

        say("\n== position coverage (calibrated area split into 4 x 3 cells) ==")
        eu = np.linspace(CAL_U[0], CAL_U[1], 5)
        ev = np.linspace(CAL_V[0], CAL_V[1], 4)
        for c in classes:
            pts = np.array(pos[c]) if pos[c] else np.zeros((0, 2))
            grid, _, _ = np.histogram2d(pts[:, 1], pts[:, 0], bins=[ev, eu]) if len(pts) else (np.zeros((3, 4)), 0, 0)
            emptyc = int((grid == 0).sum())
            say("  %-14s empty cells: %d of 12" % (cname(c), emptyc))
            say("".join("      " + " ".join("%3d" % x for x in row) + "\n" for row in grid).rstrip())
            if emptyc > 3:
                warn.append("%s never appears in %d of 12 board areas. Place it in more places." % (cname(c), emptyc))
        say("  box centres outside the calibrated area: %d (fine for detection, the robot cannot pick there)" % outside)

        say("\n== box size per class (long x short side, mm, using %.2f mm/px) ==" % MM_PER_PX)
        for c in classes:
            if not dims[c]:
                continue
            d = np.array(dims[c]) * MM_PER_PX
            say("  %-14s median %.0f x %.0f  (long %.0f..%.0f, short %.0f..%.0f)" % (
                cname(c), *np.median(d, axis=0), d[:, 0].min(), d[:, 0].max(), d[:, 1].min(), d[:, 1].max()))
            med_l = np.median(d[:, 0])
            odd = int((np.abs(d[:, 0] - med_l) > 0.3 * med_l).sum())
            if odd:
                warn.append("%s: %d boxes with a very different size. Check them (wrong class or a distractor labelled by mistake)." % (cname(c), odd))

        if any(angles.values()):
            say("\n== orientation (OBB long-side angle, 0-180 deg, counts per 30 deg) ==")
            for c in classes:
                if angles[c]:
                    hst, _ = np.histogram(angles[c], bins=6, range=(0, 180))
                    say("  %-14s %s" % (cname(c), " ".join("%3d" % x for x in hst)))
                    if (hst == 0).sum() >= 3:
                        warn.append("%s: few angles covered. Rotate it more." % cname(c))

        if min_dist:
            md = np.array(min_dist)
            say("\n== closest pair of targets per image (mm) ==")
            say("  below 80 mm: %d images, 80-140 mm: %d, above 140 mm: %d" % (
                (md < 80).sum(), ((md >= 80) & (md < 140)).sum(), (md >= 140).sum()))
            if (md < 80).sum() < 0.1 * len(md):
                warn.append("few images with blocks close together (<80 mm). Add some.")

        # plots
        fig, ax = plt.subplots(figsize=(6, 0.45 * len(classes) + 1.4))
        y = np.arange(len(classes))
        vals = [inst[c] for c in classes]
        ax.barh(y, vals, color=BAR, height=0.6)
        for yi, v in zip(y, vals):
            ax.text(v, yi, " %d" % v, va="center", color=INK, fontsize=9)
        ax.set_yticks(y)
        ax.set_yticklabels([cname(c) for c in classes])
        ax.invert_yaxis()
        ax.set_xlabel("labelled boxes", color=INK)
        ax.set_title("Instances per class", color=INK, fontsize=11, loc="left")
        style(ax)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color="#e6e6e6", linewidth=0.8)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, "class_counts.png"), dpi=130)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5, 3.2))
        ks = list(range(0, max(per_img) + 1)) if per_img else [0]
        ax.bar(ks, [per_img[k] for k in ks], color=BAR, width=0.7)
        ax.set_xticks(ks)
        ax.set_xlabel("targets in the image", color=INK)
        ax.set_ylabel("images", color=INK)
        ax.set_title("Targets per image", color=INK, fontsize=11, loc="left")
        style(ax)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, "targets_per_image.png"), dpi=130)
        plt.close(fig)

        ncls = max(1, len(classes))
        cols = min(3, ncls)
        rows_n = int(np.ceil(ncls / cols))
        fig, axs = plt.subplots(rows_n, cols, figsize=(4.2 * cols, 2.6 * rows_n), squeeze=False)
        bg = (mean_img / n_mean)[:, :, ::-1].astype(np.uint8) if mean_img is not None else None
        for i in range(rows_n * cols):
            ax = axs[i // cols][i % cols]
            ax.axis("off")
            if i >= ncls:
                continue
            c = classes[i]
            if bg is not None:
                ax.imshow(bg, alpha=0.35)
            ax.add_patch(plt.Rectangle((CAL_U[0], CAL_V[0]), CAL_U[1] - CAL_U[0], CAL_V[1] - CAL_V[0],
                                       fill=False, edgecolor=MUTED, linestyle="--", linewidth=1))
            pts = np.array(pos[c]) if pos[c] else np.zeros((0, 2))
            if len(pts):
                ax.scatter(pts[:, 0], pts[:, 1], s=9, color=BAR, linewidths=0, alpha=0.8)
            ax.set_xlim(0, 1280)
            ax.set_ylim(720, 0)
            ax.set_title("%s (%d)" % (cname(c), len(pts)), color=INK, fontsize=10, loc="left")
        fig.suptitle("Where each class appears (dashed = calibrated area)", color=INK, fontsize=11, x=0.01, ha="left")
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, "positions.png"), dpi=130)
        plt.close(fig)

        if a.draw and labelled:
            dd = os.path.join(a.out, "drawn")
            os.makedirs(dd, exist_ok=True)
            for p in random.Random(1).sample(sorted(labelled), min(a.draw, len(labelled))):
                im = cv2.imread(p)
                for cls, cx, cy, lo, sh, ang, pts in labelled[p]:
                    box = np.round(pts).astype(np.int32)
                    cv2.polylines(im, [box], True, (255, 0, 255), 2)
                    cv2.putText(im, cname(cls), (int(box[:, 0].min()), int(box[:, 1].min()) - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
                cv2.imwrite(os.path.join(dd, os.path.basename(p)), im)
            say("\nsaved %d drawn images in %s (check the labels by eye)" % (min(a.draw, len(labelled)), dd))
    else:
        say("\n(no labels found: only image checks were done)")

    say("\n================ WARNINGS ================")
    if warn:
        for w_ in warn:
            say("  - " + w_)
    else:
        say("  none")
    text = "\n".join(rep)
    with open(os.path.join(a.out, "report.txt"), "w") as f:
        f.write(text + "\n")
    print(text)
    print("\nreport and plots in %s" % a.out)


if __name__ == "__main__":
    main()
