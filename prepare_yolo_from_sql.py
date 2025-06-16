#!/usr/bin/env python3
"""
prepare_yolo_from_sql.py

Extracts stamp polygons (rect or arbitrary poly) from your albumPages INSERTs
and writes YOLOv8 detection labels directly into downloaded_images/:

  <class> <xc> <yc> <w> <h>

All coords are normalised to [0,1].
"""

import re, sys, os
from tqdm import tqdm

CLASS_ID    = 0
OUT_DIR     = "downloaded_images"   # directory with your images (and where .txt labels go)
# these indices assume INSERT has:
# (id, pageid, x1, y1, x2, y2, x3, y3, x4, y4, shape, stampid, imgW, imgH, …)
SHAPE_IDX   = 10
IMGW_IDX    = 12
IMGH_IDX    = 13
COORD_START = 2   # begin of x1
COORD_END   = 10  # end of y4 (exclusive)


def parse_sql(text):
    rows = []
    # grab all albumPages INSERT blocks
    blocks = re.findall(r"INSERT INTO `albumPages` VALUES\s*(.+?);",
                        text, flags=re.DOTALL)
    for block in blocks:
        for entry in re.findall(r"\(([^)]+)\)", block):
            # split on commas not inside quotes
            parts, cur, in_str = [], "", False
            for ch in entry:
                if ch == "'":
                    in_str = not in_str
                    cur += ch
                elif ch == "," and not in_str:
                    parts.append(cur.strip())
                    cur = ""
                else:
                    cur += ch
            parts.append(cur.strip())

            # ensure shape + img dims exist
            if len(parts) <= IMGH_IDX:
                continue

            pageid = parts[1].strip("' ")
            shape  = parts[SHAPE_IDX].strip("' ").lower()
            if shape not in ("rect", "poly"):
                continue

            # collect any non-NULL coords among x1→y4
            coords = []
            for i in range(COORD_START, COORD_END, 2):
                xs = parts[i].upper()
                ys = parts[i+1].upper()
                if xs != "NULL" and ys != "NULL":
                    try:
                        coords.append((int(xs), int(ys)))
                    except ValueError:
                        pass

            # need at least 2 pts for rect, ≥3 for poly
            if shape == "rect" and len(coords) < 2:
                continue
            if shape == "poly"  and len(coords) < 3:
                continue

            # parse image size
            try:
                imgW = int(parts[IMGW_IDX])
                imgH = int(parts[IMGH_IDX])
            except ValueError:
                continue

            rows.append((pageid, shape, coords, imgW, imgH))
    return rows


def write_labels(rows):
    # ensure output directory exists
    os.makedirs(OUT_DIR, exist_ok=True)

    # group annotations by pageid
    by_page = {}
    for pageid, shape, coords, imgW, imgH in rows:
        by_page.setdefault(pageid, []).append((shape, coords, imgW, imgH))

    for pageid, items in tqdm(by_page.items(), desc="Writing labels"):
        txt_path = os.path.join(OUT_DIR, f"{pageid}.txt")
        with open(txt_path, "w") as f:
            for shape, coords, imgW, imgH in items:
                # build corner list: rect→2 pts, poly→all pts
                if shape == "rect":
                    (x1,y1), (x2,y2) = coords[:2]
                    xs, ys = [x1, x2], [y1, y2]
                else:
                    xs = [x for x,y in coords]
                    ys = [y for x,y in coords]

                # compute bounding box
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)

                # normalize
                xc = ((x_min + x_max) / 2) / imgW
                yc = ((y_min + y_max) / 2) / imgH
                w  = (x_max - x_min)        / imgW
                h  = (y_max - y_min)        / imgH

                # clamp to [0,1]
                xc, yc, w, h = [
                    max(0.0, min(1.0, v)) for v in (xc, yc, w, h)
                ]
                if w <= 0 or h <= 0:
                    continue

                # write detection label: class xc yc w h
                f.write(f"{CLASS_ID} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")


def main():
    if len(sys.argv) != 2:
        print("Usage: python prepare_yolo_from_sql.py stamps.sql")
        sys.exit(1)
    sql_file = sys.argv[1]
    if not os.path.isfile(sql_file):
        print(f"File not found: {sql_file}")
        sys.exit(1)

    text = open(sql_file, encoding="utf-8").read()
    rows = parse_sql(text)
    if not rows:
        print("❌ No valid annotations found. Check your SQL.")
        sys.exit(1)

    write_labels(rows)
    print(f"✅ Wrote {len(rows)} label files into '{OUT_DIR}/'.")


if __name__ == "__main__":
    main()
