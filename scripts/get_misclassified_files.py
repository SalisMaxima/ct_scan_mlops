import json
from pathlib import Path

# Constants from data.py
CLASSES = [
    "adenocarcinoma",
    "large_cell_carcinoma",
    "squamous_cell_carcinoma",
    "normal",
]

PREFIX_TO_CLASS = {
    "adenocarcinoma": "adenocarcinoma",
    "large.cell.carcinoma": "large_cell_carcinoma",
    "large_cell_carcinoma": "large_cell_carcinoma",
    "squamous.cell.carcinoma": "squamous_cell_carcinoma",
    "squamous_cell_carcinoma": "squamous_cell_carcinoma",
    "normal": "normal",
}

IMG_EXTS = {".png", ".jpg", ".jpeg"}


def _infer_label_from_folder(folder_name):
    name = folder_name.lower()
    for prefix, cls in PREFIX_TO_CLASS.items():
        if name.startswith(prefix):
            return cls
    return None


def main():
    # 1. Load error cases
    try:
        with Path("reports/error_analysis/error_cases.json").open() as f:
            error_cases = json.load(f)
    except FileNotFoundError:
        print("Error: reports/error_analysis/error_cases.json not found.")
        return

    # 2. Replicate file collection logic
    data_root = Path("data/raw/chest-ctscan-images/Data/test")

    all_samples = []

    if not data_root.exists():
        print(f"Error: Data root {data_root} does not exist.")
        return

    sorted_folders = sorted([p for p in data_root.iterdir() if p.is_dir()])

    for class_folder in sorted_folders:
        label_str = _infer_label_from_folder(class_folder.name)
        if label_str is None:
            continue

        # Logic matches src/ct_scan_mlops/data.py:preprocess exactly
        for img_path in class_folder.rglob("*"):
            if img_path.is_file() and img_path.suffix.lower() in IMG_EXTS:
                all_samples.append((img_path, label_str))

    print(f"Total test samples collected: {len(all_samples)}")

    # 3. Map errors
    print("\nMisclassified Samples:")
    print("-" * 120)
    print(f"{'Index':<6} | {'True Label':<25} | {'Pred Label':<25} | {'Filepath'}")
    print("-" * 120)

    mismatch_count = 0
    for error in error_cases:
        idx = error["sample_idx"]
        true_label = error["true_label"]
        pred_label = error["pred_label"]

        if idx < len(all_samples):
            filepath, file_label = all_samples[idx]

            # Sanity check
            if file_label != true_label:
                mismatch_count += 1
                print(
                    f"{idx:<6} | {true_label:<25} | {pred_label:<25} | {filepath} <--- LABEL MISMATCH (File has {file_label})"
                )
            else:
                print(f"{idx:<6} | {true_label:<25} | {pred_label:<25} | {filepath}")
        else:
            print(f"Error: Index {idx} out of bounds (max {len(all_samples) - 1})")

    if mismatch_count > 0:
        print("\nWARNING: Label mismatches detected!")
        print("This means the file iteration order differs from when 'preprocess' was run.")
        print("The filepaths shown above for mismatches are likely INCORRECT.")


if __name__ == "__main__":
    main()
