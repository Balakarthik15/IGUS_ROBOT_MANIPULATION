#!/usr/bin/env python3
"""
Look through the labels and correct them by hand (OBB boxes, YOLO-OBB files).

    python3 review_labels.py --images dataset --labels labels_auto
    python3 review_labels.py --images dataset --labels labels_auto --only-notes   (only images with a note in review.csv)
    python3 review_labels.py --images dataset --labels labels_auto --start 120

Reads and writes the same files as auto_label_obb.py:
    labels_auto/<session>/<image>.txt                targets   id x1 y1 ... x4 y4
    labels_auto/_distractors/<session>/<image>.txt   distractors (name instead of id)
Every image you save is listed in labels_auto/_reviewed.txt, so running
auto_label_obb.py again will not overwrite your corrections.

MOUSE
  left click on a box        select it
  left click on empty space  unselect

KEYS (click on the image window first)
  n / space    save, next image          b       save, previous image
  0 1 2 3 4    selected box becomes red / green / yellow / white / lightblue
  x            selected box becomes a distractor
  d            delete selected box
  a            add a box: click 3 points: 2 corners along one LONG side, then 1 point
               on the opposite long side. New box gets the class of the last used key.
  i j k l      move selected box up / left / down / right (1 px)
  u o          rotate selected box -1 / +1 degree   (U O: 5 degrees)
  + -          longer / shorter                     ] [ : wider / narrower
  h            show / hide distractors              f   full image / board zoom
  r            undo all changes on this image (reload)
  s            save                                 Esc save and quit
Boxes: thick colour = target, thin magenta = distractor, yellow outline = selected.
"""
import argparse
import csv
import os

import cv2
import numpy as np

CLASSES = ["red", "green", "yellow", "white", "lightblue"]
COL = [(40, 40, 230), (40, 170, 40), (0, 200, 240), (240, 240, 240), (230, 200, 90)]
BOARD_U = (360, 780)
ZOOM = 1.25


# ------------------------------------------------------------------ files
def find_images(root):
    out = []
    for d, _, files in os.walk(root):
        if "_review" in d or "_distractors" in d:
            continue
        for f in sorted(files):
            if f.lower().endswith((".png", ".jpg", ".jpeg")) and "_debug" not in f:
                out.append(os.path.join(d, f))
    return sorted(out)


def read_boxes(path, w, h, named):
    boxes = []
    if not os.path.exists(path):
        return boxes
    with open(path) as f:
        for line in f:
            v = line.split()
            if len(v) != 9:
                continue
            pts = (np.array([float(x) for x in v[1:]]).reshape(4, 2) * [w, h]).astype(np.float32)
            rect = cv2.minAreaRect(pts)
            boxes.append({"cls": v[0] if named else int(v[0]), "rect": rect, "dis": named})
    return boxes


def corners_norm(rect, w, h):
    pts = cv2.boxPoints(rect)
    pts[:, 0] = np.clip(pts[:, 0] / w, 0, 1)
    pts[:, 1] = np.clip(pts[:, 1] / h, 0, 1)
    return " ".join("%.6f %.6f" % (x, y) for x, y in pts)


def rect_from_3(p1, p2, p3):
    """p1, p2 = corners along one side, p3 = any point on the opposite side"""
    p1, p2, p3 = (np.array(p, float) for p in (p1, p2, p3))
    e = p2 - p1
    L = np.linalg.norm(e)
    if L < 3:
        return None
    u = e / L
    n = np.array([-u[1], u[0]])
    d = float(np.dot(p3 - p1, n))
    if abs(d) < 3:
        return None
    c = p1 + e / 2 + n * d / 2
    ang = float(np.degrees(np.arctan2(u[1], u[0])))
    return ((float(c[0]), float(c[1])), (float(L), float(abs(d))), ang)


def point_in_rect(pt, rect):
    box = cv2.boxPoints(rect).astype(np.float32)
    return cv2.pointPolygonTest(box, (float(pt[0]), float(pt[1])), False) >= 0


# ------------------------------------------------------------------ editor
class Editor:
    def __init__(self, images, img_root, lab_root, start=0):
        self.images, self.img_root, self.lab_root = images, img_root, lab_root
        self.i = start
        self.sel = None
        self.adding = []
        self.last_cls = 0
        self.show_dis = True
        self.full = False
        self.dirty = False
        self.reviewed_file = os.path.join(lab_root, "_reviewed.txt")
        self.reviewed = set(open(self.reviewed_file).read().split()) if os.path.exists(self.reviewed_file) else set()
        self.load()

    # --- paths
    def rel(self):
        return os.path.splitext(os.path.relpath(self.images[self.i], self.img_root))[0]

    def tpath(self):
        return os.path.join(self.lab_root, self.rel() + ".txt")

    def dpath(self):
        return os.path.join(self.lab_root, "_distractors", self.rel() + ".txt")

    # --- io
    def load(self):
        self.img = cv2.imread(self.images[self.i])
        self.h, self.w = self.img.shape[:2]
        self.boxes = read_boxes(self.tpath(), self.w, self.h, False) + read_boxes(self.dpath(), self.w, self.h, True)
        self.sel, self.adding, self.dirty = None, [], False

    def save(self):
        t = ["%d %s" % (b["cls"], corners_norm(b["rect"], self.w, self.h)) for b in self.boxes if not b["dis"]]
        d = ["%s %s" % (b["cls"], corners_norm(b["rect"], self.w, self.h)) for b in self.boxes if b["dis"]]
        for p, lines in ((self.tpath(), t), (self.dpath(), d)):
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))
        if self.rel() not in self.reviewed:
            self.reviewed.add(self.rel())
            with open(self.reviewed_file, "a") as f:
                f.write(self.rel() + "\n")
        self.dirty = False

    # --- view <-> image coordinates
    def to_img(self, x, y):
        if self.full:
            return x, y
        return x / ZOOM + BOARD_U[0], y / ZOOM

    def render(self):
        out = self.img.copy()
        for k, b in enumerate(self.boxes):
            if b["dis"] and not self.show_dis:
                continue
            box = cv2.boxPoints(b["rect"]).astype(np.int32)
            if b["dis"]:
                cv2.polylines(out, [box], True, (255, 0, 255), 1)
                cv2.putText(out, str(b["cls"]), (int(box[:, 0].max()) + 3, int(box[:, 1].max())),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 255), 1)
            else:
                cv2.polylines(out, [box], True, (0, 0, 0), 4)
                cv2.polylines(out, [box], True, COL[b["cls"]], 2)
                cv2.putText(out, "%d %s" % (b["cls"], CLASSES[b["cls"]]),
                            (int(box[:, 0].min()), int(box[:, 1].min()) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL[b["cls"]], 1)
            if k == self.sel:
                cv2.polylines(out, [box], True, (0, 255, 255), 1)
                for p in box:
                    cv2.circle(out, tuple(int(v) for v in p), 3, (0, 255, 255), -1)
        for p in self.adding:
            cv2.circle(out, (int(p[0]), int(p[1])), 4, (0, 255, 255), -1)
        view = out if self.full else cv2.resize(out[:, BOARD_U[0]:BOARD_U[1]], None, fx=ZOOM, fy=ZOOM)
        bar = np.full((52, view.shape[1], 3), 30, np.uint8)
        n_t = sum(1 for b in self.boxes if not b["dis"])
        done = "reviewed" if self.rel() in self.reviewed else "not reviewed"
        line1 = "%d/%d  %s  [%s]%s" % (self.i + 1, len(self.images), self.rel(), done, " *changed*" if self.dirty else "")
        sel = ""
        if self.sel is not None:
            b = self.boxes[self.sel]
            sel = "selected: %s" % (b["cls"] if b["dis"] else CLASSES[b["cls"]])
        mode = "ADD: click point %d of 3" % (len(self.adding) + 1) if self.adding or self.mode_add else ""
        line2 = "targets %d  %s  %s   new box class: %s" % (n_t, sel, mode, CLASSES[self.last_cls])
        cv2.putText(bar, line1, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(bar, line2, (6, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        return np.vstack([bar, view])

    mode_add = False

    # --- actions
    def click(self, x, y):
        y -= 52                                   # info bar
        if y < 0:
            return
        px, py = self.to_img(x, y)
        if self.mode_add:
            self.adding.append((px, py))
            if len(self.adding) == 3:
                r = rect_from_3(*self.adding)
                self.adding, self.mode_add = [], False
                if r is not None:
                    self.boxes.append({"cls": self.last_cls, "rect": r, "dis": False})
                    self.sel = len(self.boxes) - 1
                    self.dirty = True
            return
        hits = [k for k, b in enumerate(self.boxes)
                if (self.show_dis or not b["dis"]) and point_in_rect((px, py), b["rect"])]
        if hits:   # smallest box under the mouse
            self.sel = min(hits, key=lambda k: self.boxes[k]["rect"][1][0] * self.boxes[k]["rect"][1][1])
        else:
            self.sel = None

    def edit(self, dx=0, dy=0, da=0, dl=0, dw=0):
        if self.sel is None:
            return
        (cx, cy), (a, b), ang = self.boxes[self.sel]["rect"]
        # keep "a" as the long side so + / - always change the length
        if a < b:
            a, b, ang = b, a, ang + 90
        self.boxes[self.sel]["rect"] = ((cx + dx, cy + dy), (max(4, a + dl), max(4, b + dw)), ang + da)
        self.dirty = True

    def key(self, k):
        """returns 'quit' to stop"""
        c = chr(k) if 0 <= k < 256 else ""
        if k == 27:
            self.save()
            return "quit"
        if c in "01234" and c:
            self.last_cls = int(c)
            if self.sel is not None:
                b = self.boxes[self.sel]
                b["cls"], b["dis"] = int(c), False
                self.dirty = True
        elif c == "x" and self.sel is not None:
            b = self.boxes[self.sel]
            if not b["dis"]:
                b["cls"], b["dis"] = "d_manual", True
                self.dirty = True
        elif c == "d" or k in (8, 127, 255):
            if self.sel is not None:
                self.boxes.pop(self.sel)
                self.sel, self.dirty = None, True
        elif c == "a":
            self.mode_add, self.adding, self.sel = True, [], None
        elif c in "ijkl" and c:
            self.edit(dx={"j": -1, "l": 1}.get(c, 0), dy={"i": -1, "k": 1}.get(c, 0))
        elif c in "uoUO" and c:
            self.edit(da={"u": -1, "o": 1, "U": -5, "O": 5}[c])
        elif c in "+=":
            self.edit(dl=1)
        elif c == "-":
            self.edit(dl=-1)
        elif c == "]":
            self.edit(dw=1)
        elif c == "[":
            self.edit(dw=-1)
        elif c == "h":
            self.show_dis = not self.show_dis
        elif c == "f":
            self.full = not self.full
        elif c == "r":
            self.load()
        elif c == "s":
            self.save()
        elif c in "n " and c:
            self.save()
            if self.i < len(self.images) - 1:
                self.i += 1
                self.load()
        elif c == "b":
            self.save()
            if self.i > 0:
                self.i -= 1
                self.load()
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--start", type=int, default=0, help="start at this image number (0 = first)")
    ap.add_argument("--only-notes", action="store_true", help="only images with a note in _review/review.csv")
    ap.add_argument("--only-unreviewed", action="store_true", help="skip images you already saved here")
    a = ap.parse_args()

    images = find_images(a.images)
    if a.only_notes:
        rc = os.path.join(a.labels, "_review", "review.csv")
        noted = set()
        if os.path.exists(rc):
            with open(rc) as f:
                for row in csv.DictReader(f):
                    if row.get("check"):
                        noted.add(row["image"])
        images = [p for p in images if os.path.splitext(os.path.relpath(p, a.images))[0] in noted]
    if a.only_unreviewed:
        rf = os.path.join(a.labels, "_reviewed.txt")
        done = set(open(rf).read().split()) if os.path.exists(rf) else set()
        images = [p for p in images if os.path.splitext(os.path.relpath(p, a.images))[0] not in done]
    if not images:
        print("no images to show")
        return
    ed = Editor(images, a.images, a.labels, min(a.start, len(images) - 1))
    win = "review labels  (keys: see top of review_labels.py)"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, lambda ev, x, y, fl, p: ed.click(x, y) if ev == cv2.EVENT_LBUTTONDOWN else None)
    print(__doc__.split("MOUSE")[1])
    while True:
        cv2.imshow(win, ed.render())
        k = cv2.waitKeyEx(30)
        if k == -1:
            continue
        if k in (0xFFFF, 0xFF9F, 3014656):     # Delete key on Linux / Windows
            k = 127
        if ed.key(k & 0xFF) == "quit":
            break
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            ed.save()
            break
    cv2.destroyAllWindows()
    print("reviewed so far: %d images (listed in %s)" % (len(ed.reviewed), ed.reviewed_file))


if __name__ == "__main__":
    main()
