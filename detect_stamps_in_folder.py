#!/usr/bin/env python3
"""
detect_stamps.py

Scans all images in "downloaded_images", detects stamps (with perforations),
and writes a series of SQL INSERT statements into "sql_output.txt" for table `billalbumpages`.
Optionally draws bounding boxes on detected stamps and saves output images.

Usage:
    python detect_stamps.py [--draw-bboxes] [--outimgdir OUTPUT_DIR]

Dependencies:
    pip install opencv-python numpy tqdm
"""

import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import argparse

# ───────────────────── IMAGE PRE-PROCESSING (DEFAULT) ───────────────────── #

def make_mask_default(gray: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    th = cv2.morphologyEx(
        th,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)),
        iterations=2
    )
    th = cv2.morphologyEx(
        th,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        iterations=1
    )
    return th

def dilate_mask(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask
    k = 2 * px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    return cv2.dilate(mask, kernel, iterations=1)

# ───────────────────── IMAGE PRE-PROCESSING (DARK) ───────────────────── #

def make_edges_dark(gray: np.ndarray, debug: bool = False) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=1)
    return dilated

# ───────────────────────── BOX DETECTION ────────────────────────── #

def find_boxes_default(mask: np.ndarray,
                       img_hw: tuple[int, int],
                       min_area: int,
                       min_w: int,
                       min_h: int,
                       pad_px: int) -> list[tuple[int, int, int, int]]:
    H, W = img_hw
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w < min_w or h < min_h:
            continue
        if x == 0 or y == 0 or x + w == W or y + h == H:
            continue
        x1 = max(0, x - pad_px)
        y1 = max(0, y - pad_px)
        x2 = min(W - 1, x + w + pad_px)
        y2 = min(H - 1, y + h + pad_px)
        boxes.append((x1, y1, x2, y2))
    return boxes

def find_boxes_dark(edge_img: np.ndarray,
                    img_hw: tuple[int, int],
                    min_area: int,
                    min_w: int,
                    min_h: int,
                    pad_px: int) -> list[tuple[int, int, int, int]]:
    H, W = img_hw
    contours, _ = cv2.findContours(edge_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w < min_w or h < min_h:
            continue
        if x == 0 or y == 0 or x + w == W or y + h == H:
            continue
        x1 = max(0, x - pad_px)
        y1 = max(0, y - pad_px)
        x2 = min(W - 1, x + w + pad_px)
        y2 = min(H - 1, y + h + pad_px)
        boxes.append((x1, y1, x2, y2))
    return boxes

# ───────────────────────────────────────────────────────────────────── #

def main():
    parser = argparse.ArgumentParser(description="Detect stamps and optionally draw bounding boxes.")
    parser.add_argument("--input-dir", type=Path, default=Path("downloaded_images"),
                        help="Directory of input images")
    parser.add_argument("--output-sql", type=Path, default=Path("sql_output.txt"),
                        help="Path to write SQL insert statements")
    parser.add_argument("--draw-bboxes", action="store_true",
                        help="Enable saving of images with drawn bounding boxes")
    parser.add_argument("--outimgdir", type=Path, default=Path("bboxes_output"),
                        help="Output directory for images with bounding boxes")
    args = parser.parse_args()

    INPUT_DIR     = args.input_dir
    OUTPUT_SQL    = args.output_sql
    DRAW          = args.draw_bboxes
    OUT_IMG_DIR   = args.outimgdir

    MIN_AREA      = 15_000
    PAD_PX        = 10
    DILATE_PX     = 20
    MIN_WIDTH     = 150
    MIN_HEIGHT    = 100
    dark_initials = {"R", "D", "H", "I", "J", "K", "O", "P"}

    if not INPUT_DIR.is_dir():
        raise FileNotFoundError(f"Directory not found: {INPUT_DIR}")

    if DRAW:
        OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(f for f in os.listdir(INPUT_DIR)
                   if f.lower().endswith((".jpg", ".jpeg", ".png")))

    with OUTPUT_SQL.open("w", encoding="utf-8") as sql_file:
        for fname in tqdm(files, desc="Detecting stamps"):
            img_path = INPUT_DIR / fname
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            H, W = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            first = fname[0].upper()

            if first in dark_initials:
                edge_img = make_edges_dark(gray)
                boxes = find_boxes_dark(edge_img, (H, W),
                                        min_area=MIN_AREA,
                                        min_w=MIN_WIDTH,
                                        min_h=MIN_HEIGHT,
                                        pad_px=PAD_PX)
            else:
                mask = make_mask_default(gray)
                mask = dilate_mask(mask, DILATE_PX)
                boxes = find_boxes_default(mask, (H, W),
                                           min_area=MIN_AREA,
                                           min_w=MIN_WIDTH,
                                           min_h=MIN_HEIGHT,
                                           pad_px=PAD_PX)

            has_wm = 1 if "_watermark" in fname.lower() else 0
            pageid = Path(fname).stem.replace("'", "''")

            # Draw and save bboxes if requested
            if DRAW and boxes:
                img_copy = img.copy()
                for (x1, y1, x2, y2) in boxes:
                    cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.imwrite(str(OUT_IMG_DIR / fname), img_copy)

            for x1, y1, x2, y2 in boxes:
                x3, y3 = x2, y1
                x4, y4 = x1, y2
                sql = (
                    "INSERT INTO billalbumpages\n"
                    "(pageid, x1, y1, x2, y2, x3, y3, x4, y4, shape,\n"
                    " stampid, imgWidth, imgHeight, plateid,\n"
                    " perfin_text, comments, isGroup, quantity, hasWatermarkImage)\n"
                    "VALUES\n"
                    f"('{pageid}', {x1}, {y1}, {x2}, {y2}, {x3}, {y3}, {x4}, {y4}, 'rect',\n"
                    f" NULL, {W}, {H}, NULL,\n"
                    " NULL, NULL, 0, 1, "
                    f"{has_wm}"
                    ");\n"
                )
                sql_file.write(sql)

    print(f"✅ All INSERT statements written to '{OUTPUT_SQL}'")

if __name__ == "__main__":
    main()
