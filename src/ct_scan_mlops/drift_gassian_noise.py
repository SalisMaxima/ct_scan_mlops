from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image

IMG_EXTS = {".png", ".jpg", ".jpeg"}


def iter_images(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]


def add_gaussian_noise_uint8(img: Image.Image, mean: float, std: float) -> Image.Image:
    """
    Add Gaussian noise to an image.

    mean/std are in pixel-intensity units (0..255 space).
    - mean shifts brightness (intensity offset)
    - std controls noise spread
    """
    arr = np.asarray(img, dtype=np.float32)
    noise = np.random.normal(loc=mean, scale=std, size=arr.shape).astype(np.float32)
    out = arr + noise
    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Create a noisy copy of an image dataset (Gaussian noise).")
    ap.add_argument("--src", type=Path, required=True, help="Source image root (e.g., training folder).")
    ap.add_argument("--dst", type=Path, required=True, help="Output root for noisy images.")
    ap.add_argument("--mean", type=float, default=0.0, help="Gaussian mean in 0..255 intensity units.")
    ap.add_argument("--std", type=float, default=10.0, help="Gaussian std in 0..255 intensity units.")
    ap.add_argument("--fraction", type=float, default=1.0, help="Fraction of images to corrupt (0..1).")
    ap.add_argument("--seed", type=int, default=0, help="Random seed (0 for nondeterministic).")
    ap.add_argument("--mode", choices=["copy", "only_noisy"], default="copy",
                    help="copy: copy clean images too; only_noisy: write only corrupted images.")
    args = ap.parse_args()

    if not args.src.exists():
        raise SystemExit(f"--src not found: {args.src}")
    if not (0.0 < args.fraction <= 1.0):
        raise SystemExit("--fraction must be in (0, 1].")
    if args.std < 0:
        raise SystemExit("--std must be >= 0.")

    if args.seed != 0:
        random.seed(args.seed)
        np.random.seed(args.seed)

    imgs = iter_images(args.src)
    if not imgs:
        raise SystemExit(f"No images found under: {args.src}")

    # Choose which images to corrupt
    n_corrupt = int(round(len(imgs) * args.fraction))
    corrupt_set = set(random.sample(imgs, k=n_corrupt))

    for p in imgs:
        rel = p.relative_to(args.src)
        out_path = args.dst / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if p in corrupt_set:
            img = Image.open(p)
            noisy = add_gaussian_noise_uint8(img, mean=args.mean, std=args.std)
            noisy.save(out_path)
        else:
            if args.mode == "copy":
                # Keep same file bytes if possible
                out_path.write_bytes(p.read_bytes())

    print(f"Source:   {args.src}")
    print(f"Output:   {args.dst}")
    print(f"Images:   {len(imgs)}")
    print(f"Corrupt:  {n_corrupt} ({args.fraction*100:.1f}%)")
    print(f"mean/std: {args.mean}/{args.std} (intensity units)")


if __name__ == "__main__":
    main()
