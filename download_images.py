#!/usr/bin/env python3
"""
download_images.py

Hard-codes the maximum pages per album and downloads every image
from page 1 to max for each album letter from the S3 bucket,
including the watermark variants for album D.

Run:
    python download_images.py

Dependencies:
    pip install requests tqdm
"""

import os
import requests
from urllib.parse import urljoin
from tqdm import tqdm

# ─── CONFIG ────────────────────────────────────────────────────────────
BUCKET_BASE_URL = "https://radon9stampss.s3.eu-west-2.amazonaws.com/"
OUTPUT_DIR = "downloaded_images"

# Hard-coded maximum page counts per album letter:
ALBUM_MAX_PAGES = {
    "A": 110,
    "B": 48,
    "C": 33,
    "D": 53,
    "E": 72,
    "F": 60,
    "G": 73,
    "H": 60,
    "I": 60,
    "J": 59,
    "K": 60,
    "L": 60,
    "O": 60,
    "P": 60,
    "R": 60,
}
# ───────────────────────────────────────────────────────────────────────


def download_image(fname):
    """Downloads an image from S3 if not already present."""
    out_path = os.path.join(OUTPUT_DIR, fname)
    if os.path.exists(out_path):
        return
    url = urljoin(BUCKET_BASE_URL, fname)
    response = requests.get(url, stream=True)
    response.raise_for_status()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(out_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)


def main():
    # Build list of filenames based on hard-coded pages
    filenames = []
    for album, max_page in ALBUM_MAX_PAGES.items():
        for page in range(1, max_page + 1):
            filenames.append(f"{album}{page}.jpg")

        # If this is album D, also grab the watermark variants:
        if album == "D":
            for page in range(1, max_page + 1):
                filenames.append(f"D{page}_watermark.jpg")

    print(
        f"Downloading {len(filenames)} images across {len(ALBUM_MAX_PAGES)} albums "
        f"(including D-watermarks)…"
    )
    for fname in tqdm(filenames, desc="Downloading images"):
        try:
            download_image(fname)
        except Exception as e:
            print(f"⚠️ Failed to download {fname}: {e}")

    print(f"✅ All images downloaded into '{OUTPUT_DIR}/'.")


if __name__ == "__main__":
    main()
