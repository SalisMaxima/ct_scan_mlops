from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

IMG_EXTS = {".png", ".jpg", ".jpeg"}


def iter_images(root: Path):
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p


def image_to_array(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.float32)
    return arr


def compute_stats(arr: np.ndarray) -> dict:
    flat = arr.reshape(-1)
    return {
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "p01": float(np.percentile(flat, 1)),
        "p50": float(np.percentile(flat, 50)),
        "p99": float(np.percentile(flat, 99)),
        "height": int(arr.shape[0]),
        "width": int(arr.shape[1]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=Path, required=True, help="Folder with images (recursive).")
    ap.add_argument("--out", type=Path, required=True, help="Output CSV path.")
    ap.add_argument("--limit", type=int, default=0, help="Optional max number of images (0 = all).")
    args = ap.parse_args()

    rows = []
    for i, img_path in enumerate(iter_images(args.data_dir), start=1):
        arr = image_to_array(img_path)
        row = compute_stats(arr)
        row["path"] = str(img_path)
        rows.append(row)
        if args.limit and i >= args.limit:
            break

    if not rows:
        raise RuntimeError(f"No images found under: {args.data_dir}")

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()
