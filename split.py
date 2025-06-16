import os, random, shutil

SRC = 'downloaded_images'     # where your source .jpg/.png + .txt live
DST = 'training_images'       # new root for train/val/test splits
R   = (0.8, 0.1, 0.1)         # fractions for train, val, test

# make your target directories
for split in ('train', 'val', 'test'):
    os.makedirs(os.path.join(DST, split), exist_ok=True)

# grab all images
files = [f for f in os.listdir(SRC) if f.lower().endswith(('.jpg', '.png'))]
random.shuffle(files)

n = len(files)
train_end = int(R[0] * n)
val_end   = train_end + int(R[1] * n)

splits = {
    'train': files[:train_end],
    'val':   files[train_end:val_end],
    'test':  files[val_end:],
}

# copy images + their .txt labels
for split, flist in splits.items():
    out_dir = os.path.join(DST, split)
    for img in flist:
        src_img   = os.path.join(SRC, img)
        dst_img   = os.path.join(out_dir, img)
        shutil.copy(src_img, dst_img)

        label = img.rsplit('.', 1)[0] + '.txt'
        src_lbl = os.path.join(SRC, label)
        if os.path.exists(src_lbl):
            dst_lbl = os.path.join(out_dir, label)
            shutil.copy(src_lbl, dst_lbl)
