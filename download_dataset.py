"""
Prepare the LeapGestRecog dataset.

Usage — two ways:

  1. Point to a zip you already downloaded (from Kaggle browser, Google Drive, etc.):
       python download_dataset.py --zip /path/to/leapgestrecog.zip

  2. Auto-download via Kaggle API (needs ~/.kaggle/kaggle.json):
       python download_dataset.py --kaggle
"""

import argparse
import os
import subprocess
import sys
import zipfile

OUT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DATASET  = "gti-upm/leapgestrecog"


def extract(zip_path: str):
    print(f"Extracting {zip_path} …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(OUT_DIR)
    print(f"Extracted to {OUT_DIR}/")
    _verify()


def _verify():
    expected = os.path.join(OUT_DIR, "leapGestRecog")
    if os.path.exists(expected):
        n = sum(1 for _, _, files in os.walk(expected) for f in files if f.endswith(".png"))
        print(f"Ready: {n} images found in {expected}")
    else:
        folders = os.listdir(OUT_DIR)
        print(f"Note: 'leapGestRecog' folder not found. Contents: {folders}")
        print("Update DATA_DIR in config.py if the folder name differs.")


def from_zip(zip_path: str):
    if not os.path.exists(zip_path):
        print(f"ERROR: File not found: {zip_path}")
        sys.exit(1)
    os.makedirs(OUT_DIR, exist_ok=True)
    extract(zip_path)


def from_kaggle():
    kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.exists(kaggle_json):
        print("ERROR: ~/.kaggle/kaggle.json not found.")
        print("Get it from: https://www.kaggle.com/settings → API → Create New Token")
        sys.exit(1)
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Downloading '{DATASET}' via Kaggle API …")
    r = subprocess.run(["kaggle", "datasets", "download", "-d", DATASET, "-p", OUT_DIR])
    if r.returncode != 0:
        sys.exit(1)
    zips = [f for f in os.listdir(OUT_DIR) if f.endswith(".zip")]
    if not zips:
        print("No zip found after download.")
        sys.exit(1)
    zip_path = os.path.join(OUT_DIR, zips[0])
    extract(zip_path)
    os.remove(zip_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--zip",    metavar="PATH", help="Path to downloaded zip file")
    group.add_argument("--kaggle", action="store_true", help="Download via Kaggle API")
    args = parser.parse_args()

    if args.zip:
        from_zip(args.zip)
    else:
        from_kaggle()
