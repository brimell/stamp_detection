#!/usr/bin/env python3
"""
filter_large_boxes.py

Remove any YOLO labels whose box area exceeds a given fraction of the image.
"""

import os

# where your splits live
SPLIT_DIR = "training_images"   # contains 'train', 'val', 'test' subfolders
THRESHOLD = 0.5                 # max allowed box area (w * h), normalized

def filter_labels(folder: str, threshold: float):
    txts = [f for f in os.listdir(folder) if f.endswith(".txt")]
    for txt in txts:
        path = os.path.join(folder, txt)
        kept = []
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                # YOLO format: class, xc, yc, w, h, [seg...]
                if len(parts) < 5:
                    continue
                w = float(parts[3])
                h = float(parts[4])
                if w * h <= threshold:
                    kept.append(line)
        # only rewrite if we actually dropped anything
        if len(kept) != sum(1 for _ in open(path)):
            with open(path, "w") as f:
                f.writelines(kept)

if __name__ == "__main__":
    for split in ("train", "val", "test"):
        folder = os.path.join(SPLIT_DIR, split)
        if os.path.isdir(folder):
            print(f"Filtering {split} labels…")
            filter_labels(folder, THRESHOLD)
        else:
            print(f"Warning: folder not found: {folder}")
    print("Done.")
