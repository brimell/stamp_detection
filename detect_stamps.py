#!/usr/bin/env python3
"""
detect_stamp_boxes_with_margin_and_size_filter.py  –  v3

• Detects stamps on a 3024×4032 album-page scan.
• Closes perforation holes, then dilates the mask so the outer teeth
  are guaranteed to be inside the contour.
• Filters out any detected region that is smaller than a minimum area,
  a minimum width, or a minimum height.
• Enlarges each bounding box by either --pad-px or --pad-pct on top of that.
• Draws rectangles and spits out an HTML <map>.

-------------------------------------------------
pip install opencv-python numpy
-------------------------------------------------
"""

from __future__ import annotations
import cv2
import numpy as np
import argparse
from pathlib import Path


# ────────────────────────────── CLI ────────────────────────────── #

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("-i", "--input", required=True,
                   help="Path to the 3024×4032 scan.")
    p.add_argument("-o", "--output", default="boxed_stamps.jpg",
                   help="Image with red rectangles.")
    p.add_argument("--min-area", type=int, default=100_000,
                   help="Ignore contours smaller than this (px²).")
    p.add_argument("--min-width", type=int, default=0,
                   help="Ignore detected boxes narrower than this many pixels.")
    p.add_argument("--min-height", type=int, default=0,
                   help="Ignore detected boxes shorter than this many pixels.")

    # extra dilation step
    p.add_argument("--dilate-px", type=int, default=6,
                   help="Grow the stamp mask outward by this many pixels per side "
                        "before boxing (catches faint perforations).")

    pad = p.add_mutually_exclusive_group()
    pad.add_argument("--pad-pct", type=float, default=0.05,
                     help="Extra margin per side as a fraction of box size.")
    pad.add_argument("--pad-px", type=int, default=None,
                     help="Fixed extra pixels per side (overrides --pad-pct).")

    p.add_argument("--debug", action="store_true",
                   help="Show intermediate images.")
    return p.parse_args()


# ────────────────────── IMAGE PRE-PROCESSING ───────────────────── #

def make_mask(gray: np.ndarray, debug=False) -> np.ndarray:
    """Binary mask: stamps = white."""
    # blur → Otsu threshold (INV) → close (15×15) → open (5×5)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blurred, 0, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)),
                          iterations=2)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN,
                          cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
                          iterations=1)
    if debug:
        cv2.imshow("mask after close+open", th); cv2.waitKey(0)
        cv2.destroyAllWindows()
    return th


def dilate_mask(mask: np.ndarray, px: int, debug=False) -> np.ndarray:
    if px <= 0:
        return mask
    k = 2 * px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    dil = cv2.dilate(mask, kernel, iterations=1)
    if debug:
        cv2.imshow(f"mask after {px}px dilation", dil); cv2.waitKey(0)
        cv2.destroyAllWindows()
    return dil


# ───────────────────────── BOX LOGIC ───────────────────────────── #

def find_boxes(mask: np.ndarray, img_hw: tuple[int, int],
               min_area: int, min_w: int, min_h: int,
               pad_px: int | None, pad_pct: float) -> list[tuple[int,int,int,int]]:
    H, W = img_hw
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    raw = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w < min_w or h < min_h:
            continue
        if x == 0 or y == 0 or x + w == W or y + h == H:
            continue
        raw.append((x, y, w, h))

    # enlarge per user pad
    boxes = []
    for x, y, w, h in raw:
        padx = pad_px if pad_px is not None else int(w * pad_pct)
        pady = pad_px if pad_px is not None else int(h * pad_pct)
        x1 = max(0, x - padx)
        y1 = max(0, y - pady)
        x2 = min(W - 1, x + w + padx)
        y2 = min(H - 1, y + h + pady)
        boxes.append((x1, y1, x2, y2))
    return boxes


# ───────────────────── IMAGE-MAP GENERATOR ─────────────────────── #

def to_imagemap(boxes, name="albumpage"):
    parts = []
    for i, (x1, y1, x2, y2) in enumerate(boxes, 1):
        parts.append(f'<area id="{i}" shape="rect" '
                     f'coords="{x1},{y1},{x2},{y2}" '
                     f'alt="Stamp" href="javascript:void(0)">')
    return f'<map name="{name}">' + "".join(parts) + "</map>"


# ────────────────────────────── MAIN ───────────────────────────── #

def main():
    a = parse_args()
    img = cv2.imread(a.input)
    if img is None:
        raise FileNotFoundError(a.input)
    H, W = img.shape[:2]
    print(f"loaded {W}×{H}")

    mask = make_mask(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), a.debug)
    mask = dilate_mask(mask, a.dilate_px, a.debug)

    boxes = find_boxes(mask,
                       (H, W),
                       min_area=a.min_area,
                       min_w=a.min_width,
                       min_h=a.min_height,
                       pad_px=a.pad_px,
                       pad_pct=a.pad_pct)
    print(f"{len(boxes)} stamps found")

    # draw
    vis = img.copy()
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imwrite(a.output, vis)
    print("boxed image →", a.output)

    html = to_imagemap(boxes)
    Path("albumpage_map.html").write_text(html, encoding="utf-8")
    print("image-map written to albumpage_map.html\n")
    print(html)


if __name__ == "__main__":
    main()
