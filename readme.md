# Stamp Detection

This repository contains the end-to-end pipeline for downloading album pages, extracting stamp annotations, preparing a YOLOv8 dataset, training a stamp detector, and running inference.

---

## Repository Layout

YOLOv8/
├── Ultralytics YOLOv8 library (submodule or copy)

downloaded_images/
├── All raw images fetched from the album

downloaded_images_unlabeled/
├── Images without any labels (if any)

training_images/
├── train/ # 80% of images + labels
├── val/ # 10% of images + labels
└── test/ # 10% of images + labels

runs/
├── YOLOv8 training outputs (weights, logs, plots)

data.yaml
├── YOLOv8 dataset configuration (paths + class names)

stamps.sql
├── Raw SQL dump containing albumPages INSERT statements

prepare_yolo_from_sql.py
├── Converts rect/poly annotations in the SQL → YOLOv8 .txt labels

split.py
├── Randomly splits downloaded_images/ into training_images/ (80/10/10)

remove_multiple_stamp_outline.py
├── Filters out annotations whose bounding box covers too much of the page

train_yolo.py
├── 50-epoch YOLOv8-detect training script with per-epoch validation and device auto-selection

detect_stamps.py
├── Run inference on a single image

detect_stamps_in_folder.py
├── Run batch inference on all images in a folder

download_images.py
├── Fetch album page images by ID

Additional artifacts:

sql_output.sql

all_pages_stamp_maps.json

albumpage_map.html

---

## 1. Downloading Images

```bash
python download_images.py \
  --outdir downloaded_images \
  --pages <page_list.txt>
```

Downloads pages by ID

Saves images to downloaded_images/

2. Preparing Labels
2.1 From SQL to YOLO
```bash
python prepare_yolo_from_sql.py stamps.sql
```
Parses albumPages INSERT statements

Converts rect or polygon annotations into bounding rectangles

Writes YOLOv8 label files (.txt) next to each image

2.2 Splitting into Train / Val / Test
```bash
python split.py
```

Reads downloaded_images/ and their labels

Randomly splits into 80% train, 10% val, 10% test

Copies files into training_images/{train,val,test}/

3. Filtering Large Boxes
```bash
python remove_multiple_stamp_outline.py \
  --labels-dir downloaded_images \
  --max-area 0.25
```

Removes any annotation whose normalised area exceeds 25%

4. Training
```bash
python train_yolo.py
```

Automatically selects CUDA / MPS / CPU

Trains a YOLOv8-detect model for 50 epochs

Uses standard augmentations (mosaic, mixup, randaugment)

Validates after every epoch

Prints total training time at the end

Key parameters inside train_yolo.py:

```python
DATA_YAML   = 'data.yaml'
MODEL       = 'yolov8n.pt'
EPOCHS      = 50
IMG_SIZE    = 640
BATCH_SIZE  = 16
PROJECT     = 'runs'
RUN_NAME    = 'stamp_yolo8_detect'
```

5. Inference
5.1 Single Image

```bash
python detect_stamps.py \
  --model runs/stamp_yolo8_detect/weights/best.pt \
  --source downloaded_images/A1.jpg \
  --outdir output/
```

5.2 Batch Folder

```bash
python detect_stamps_in_folder.py \
  --model runs/stamp_yolo8_detect/weights/best.pt \
  --folder downloaded_images/ \
  --outdir output/
```
