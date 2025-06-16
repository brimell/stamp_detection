#!/usr/bin/env python3
# train_yolo.py  (detection with augmentations & total-time reporting)
#
# Runs a straight 50-epoch YOLOv8 detection training (with mosaic, mixup, etc.), then prints the duration.

import os
import time
import warnings
import torch
from ultralytics import YOLO

# ─── Silence the harmless augment warnings ────────────────────
warnings.filterwarnings("ignore", module="ultralytics.data.augment")

# ─── Bump the NMS time limit so you never see those warnings ──
os.environ.setdefault('YOLO_NMS_TIME_LIMIT', '60')

# ─── Configuration ────────────────────────────────────────────
DATA_YAML   = 'data.yaml'        # points at your downloaded_images/ & .txt labels
DET_MODEL   = 'yolov8n.pt'       # or whichever detection model you're using
EPOCHS      = 50
IMG_SIZE    = 640
BATCH_SIZE  = 16
PROJECT     = 'runs/train'
RUN_NAME    = 'stamp_yolo8_timed_aug'
# ──────────────────────────────────────────────────────────────

# Pick the best device
if torch.backends.mps.is_available():
    DEVICE = 'mps'
elif torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

print(f"Using device: {DEVICE}")

def format_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h {m}m {s}s"

def main():
    model = YOLO(DET_MODEL)

    t0 = time.time()
    model.train(
        data         = DATA_YAML,
        epochs       = EPOCHS,
        imgsz        = IMG_SIZE,
        batch        = BATCH_SIZE,
        device       = DEVICE,
        project      = PROJECT,
        name         = RUN_NAME,
        exist_ok     = True,
        # ─── enable standard augmentations ───────────────────────
        augment      = True,
        auto_augment = 'randaugment',
        mosaic       = 1.0,
        mixup        = 0.0,
        degrees      = 0.0,
        translate    = 0.1,
        shear        = 0.0,
        perspective  = 0.0,
        # ─── per-epoch validation ─────────────────────────────────
        val          = True
    )
    total = time.time() - t0

    print(f"\n✔️  Training complete in {format_duration(total)}")
    print(f"   Results saved to {PROJECT}/{RUN_NAME}")

if __name__ == '__main__':
    main()
