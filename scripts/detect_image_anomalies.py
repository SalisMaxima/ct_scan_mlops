"""
CT Scan Training Image Anomaly Detection Script.

Checks all training images for anomalies that could corrupt model learning:
- Color overlays (red lines, colored annotations)
- Text overlays (patient info, metadata, labels)
- Arrows, circles, or other markup
- Watermarks
- Non-CT-scan content
- Unusual pixel distributions

Saves a detailed report and thumbnail samples of flagged images.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

# === Configuration ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "chest-ctscan-images" / "Data"
REPORT_DIR = PROJECT_ROOT / "reports" / "image_anomalies"
THUMBNAIL_DIR = REPORT_DIR / "thumbnails"
THUMBNAIL_SIZE = (128, 128)

# Thresholds
COLOR_SATURATION_THRESHOLD = 30  # HSV saturation above this is "colored"
COLOR_PIXEL_FRACTION_THRESHOLD = 0.005  # >0.5% colored pixels = anomaly
HIGH_SATURATION_THRESHOLD = 80  # Very saturated pixels (strong colors)
HIGH_SATURATION_FRACTION = 0.001  # >0.1% very saturated = anomaly

BORDER_WIDTH = 15  # pixels to check for border anomalies
BORDER_INTENSITY_THRESHOLD = 200  # bright border pixels (text/overlay)
BORDER_BRIGHT_FRACTION = 0.15  # >15% bright border pixels = anomaly

TEXT_REGION_SIZE = 40  # corner region size for text detection
TEXT_CONTRAST_THRESHOLD = 150  # high-contrast threshold for text regions
TEXT_EDGE_DENSITY_THRESHOLD = 0.15  # edge density threshold for text

HISTOGRAM_DEVIATION_THRESHOLD = 3.0  # std devs from class mean

# Severity classification for anomaly types
SEVERITY_MAP: dict[str, str] = {
    "text_overlay": "critical",
    "colored_pixels": "critical",
    "colored_circle": "critical",
    "colored_lines": "critical",
    "red_overlay": "critical",
    "green_overlay": "critical",
    "blue_overlay": "critical",
    "yellow_overlay": "critical",
    "border_anomaly": "warning",
    "bright_frame": "warning",
    "high_saturation": "warning",
    "possible_watermark": "info",
    "unusual_histogram": "info",
    "extreme_brightness": "info",
}
SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2}


def classify_severity(anomalies: list[str]) -> str:
    """Return the highest severity tier among the given anomaly types."""
    best = "info"
    for anomaly in anomalies:
        sev = SEVERITY_MAP.get(anomaly, "info")
        if SEVERITY_ORDER[sev] < SEVERITY_ORDER[best]:
            best = sev
    return best


def detect_color_anomalies(img_bgr: np.ndarray) -> dict:
    """Detect colored annotations/overlays in what should be a grayscale image."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]

    total_pixels = saturation.size

    # Moderate saturation (any color present)
    colored_pixels = np.sum(saturation > COLOR_SATURATION_THRESHOLD)
    colored_fraction = colored_pixels / total_pixels

    # High saturation (strong/vivid colors like red lines, colored arrows)
    high_sat_pixels = np.sum(saturation > HIGH_SATURATION_THRESHOLD)
    high_sat_fraction = high_sat_pixels / total_pixels

    # Detect specific colors: red, green, blue
    h_channel = hsv[:, :, 0]
    s_channel = hsv[:, :, 1]
    v_channel = hsv[:, :, 2]

    # Red detection (hue wraps around: 0-10 and 170-180)
    red_mask = ((h_channel < 10) | (h_channel > 170)) & (s_channel > 50) & (v_channel > 50)
    red_fraction = np.sum(red_mask) / total_pixels

    # Green detection (hue ~35-85)
    green_mask = (h_channel > 35) & (h_channel < 85) & (s_channel > 50) & (v_channel > 50)
    green_fraction = np.sum(green_mask) / total_pixels

    # Blue detection (hue ~100-130)
    blue_mask = (h_channel > 100) & (h_channel < 130) & (s_channel > 50) & (v_channel > 50)
    blue_fraction = np.sum(blue_mask) / total_pixels

    # Yellow detection (hue ~20-35) - common for annotations
    yellow_mask = (h_channel > 20) & (h_channel < 35) & (s_channel > 50) & (v_channel > 50)
    yellow_fraction = np.sum(yellow_mask) / total_pixels

    anomalies = []
    details = {}

    if colored_fraction > COLOR_PIXEL_FRACTION_THRESHOLD:
        anomalies.append("colored_pixels")
        details["colored_pixel_fraction"] = round(float(colored_fraction), 5)

    if high_sat_fraction > HIGH_SATURATION_FRACTION:
        anomalies.append("high_saturation")
        details["high_saturation_fraction"] = round(float(high_sat_fraction), 5)

    if red_fraction > 0.001:
        anomalies.append("red_overlay")
        details["red_fraction"] = round(float(red_fraction), 5)

    if green_fraction > 0.001:
        anomalies.append("green_overlay")
        details["green_fraction"] = round(float(green_fraction), 5)

    if blue_fraction > 0.001:
        anomalies.append("blue_overlay")
        details["blue_fraction"] = round(float(blue_fraction), 5)

    if yellow_fraction > 0.001:
        anomalies.append("yellow_overlay")
        details["yellow_fraction"] = round(float(yellow_fraction), 5)

    return {"anomalies": anomalies, "details": details}


def detect_text_overlays(img_gray: np.ndarray) -> dict:
    """Detect text overlays, especially in corners and borders."""
    h, w = img_gray.shape
    anomalies: list[str] = []
    details: dict[str, object] = {}

    # Check four corners for text-like content
    corners = {
        "top_left": img_gray[:TEXT_REGION_SIZE, :TEXT_REGION_SIZE],
        "top_right": img_gray[:TEXT_REGION_SIZE, w - TEXT_REGION_SIZE :],
        "bottom_left": img_gray[h - TEXT_REGION_SIZE :, :TEXT_REGION_SIZE],
        "bottom_right": img_gray[h - TEXT_REGION_SIZE :, w - TEXT_REGION_SIZE :],
    }

    # Also check top and bottom bars (full width, thin strip)
    bars = {
        "top_bar": img_gray[:BORDER_WIDTH, :],
        "bottom_bar": img_gray[h - BORDER_WIDTH :, :],
        "left_bar": img_gray[:, :BORDER_WIDTH],
        "right_bar": img_gray[:, w - BORDER_WIDTH :],
    }

    text_regions = []

    for name, region in corners.items():
        if region.size == 0:
            continue
        # Text detection: look for high-frequency edges in corners
        edges = cv2.Canny(region, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size

        # Also check for bright text on dark background or dark text on bright bg
        std_dev = np.std(region)
        mean_val = np.mean(region)

        # High edge density + high contrast = likely text
        if edge_density > TEXT_EDGE_DENSITY_THRESHOLD and std_dev > 40:
            text_regions.append(name)
            details[f"{name}_edge_density"] = round(float(edge_density), 4)
            details[f"{name}_std"] = round(float(std_dev), 2)

        # Very bright corner (white text overlay)
        if mean_val > TEXT_CONTRAST_THRESHOLD and std_dev > 30:
            if name not in text_regions:
                text_regions.append(name)
            details[f"{name}_bright_mean"] = round(float(mean_val), 2)

    # Check bars for consistent bright lines (info panels)
    for name, region in bars.items():
        if region.size == 0:
            continue
        bright_fraction = np.sum(region > BORDER_INTENSITY_THRESHOLD) / region.size
        if bright_fraction > BORDER_BRIGHT_FRACTION:
            text_regions.append(name)
            details[f"{name}_bright_fraction"] = round(float(bright_fraction), 4)

    if text_regions:
        anomalies.append("text_overlay")
        details["text_regions"] = text_regions

    return {"anomalies": anomalies, "details": details}


def detect_edge_anomalies(img_gray: np.ndarray) -> dict:
    """Detect consistent non-CT content at image borders (info panels, text bars)."""
    h, w = img_gray.shape
    anomalies: list[str] = []
    details: dict[str, object] = {}

    # Check if borders have very different statistics from the center
    border_top = img_gray[:BORDER_WIDTH, :]
    border_bottom = img_gray[h - BORDER_WIDTH :, :]
    border_left = img_gray[:, :BORDER_WIDTH]
    border_right = img_gray[:, w - BORDER_WIDTH :]

    center = img_gray[
        h // 4 : 3 * h // 4,
        w // 4 : 3 * w // 4,
    ]

    center_mean = np.mean(center)

    border_anomalies = []
    for name, border in [
        ("top", border_top),
        ("bottom", border_bottom),
        ("left", border_left),
        ("right", border_right),
    ]:
        border_mean = np.mean(border)
        border_std = np.std(border)

        # Very different mean from center
        mean_diff = abs(border_mean - center_mean)
        if mean_diff > 80:
            border_anomalies.append(name)
            details[f"{name}_border_mean_diff"] = round(float(mean_diff), 2)

        # Very uniform border (solid color panel)
        if border_std < 5 and border_mean > 100:
            if name not in border_anomalies:
                border_anomalies.append(name)
            details[f"{name}_uniform_border"] = round(float(border_std), 2)

    if border_anomalies:
        anomalies.append("border_anomaly")
        details["anomalous_borders"] = border_anomalies

    # Check for white/bright frame around the image
    all_borders = np.concatenate(
        [
            border_top.flatten(),
            border_bottom.flatten(),
            border_left.flatten(),
            border_right.flatten(),
        ]
    )
    bright_border_fraction = np.sum(all_borders > 240) / all_borders.size
    if bright_border_fraction > 0.3:
        anomalies.append("bright_frame")
        details["bright_border_fraction"] = round(float(bright_border_fraction), 4)

    return {"anomalies": anomalies, "details": details}


def detect_markup(img_gray: np.ndarray, img_bgr: np.ndarray) -> dict:
    """Detect arrows, circles, and other markup shapes."""
    anomalies = []
    details = {}

    # Use Hough Circle detection for annotated circles
    circles = cv2.HoughCircles(
        img_gray,
        cv2.HOUGH_GRADIENT,
        dp=1.5,
        minDist=50,
        param1=100,
        param2=40,
        minRadius=15,
        maxRadius=min(img_gray.shape) // 3,
    )

    if circles is not None and len(circles[0]) > 0:
        # Filter: only flag if circles have colored pixels nearby (annotation circles)
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        for circle in circles[0]:
            cx, cy, r = int(circle[0]), int(circle[1]), int(circle[2])
            # Create ring mask around circle
            mask = np.zeros(img_gray.shape, dtype=np.uint8)
            cv2.circle(mask, (cx, cy), r + 3, 255, 6)
            ring_saturation = hsv[:, :, 1][mask > 0]
            if len(ring_saturation) > 0 and np.mean(ring_saturation) > 30:
                anomalies.append("colored_circle")
                details["circle_count"] = len(circles[0])
                break

    # Detect lines using Hough Line Transform (for arrows/annotation lines)
    edges = cv2.Canny(img_gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=10)

    if lines is not None:
        # Check if lines are colored (annotations vs. CT scan structure)
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        colored_lines = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Sample pixels along the line
            num_samples = max(abs(x2 - x1), abs(y2 - y1), 1)
            num_samples = min(num_samples, 50)
            for i in range(num_samples):
                t = i / num_samples
                px = int(x1 + t * (x2 - x1))
                py = int(y1 + t * (y2 - y1))
                if 0 <= py < hsv.shape[0] and 0 <= px < hsv.shape[1] and hsv[py, px, 1] > 50:
                    colored_lines += 1
                    break

        if colored_lines > 2:
            anomalies.append("colored_lines")
            details["colored_line_count"] = colored_lines

    return {"anomalies": anomalies, "details": details}


def compute_histogram(img_gray: np.ndarray) -> np.ndarray:
    """Compute normalized histogram for an image."""
    hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
    return hist.flatten() / hist.sum()


def detect_histogram_anomalies(hist: np.ndarray, class_mean_hist: np.ndarray, class_std_hist: np.ndarray) -> dict:
    """Detect images with unusual pixel distributions compared to class average."""
    anomalies = []
    details = {}

    # Chi-squared distance between image hist and class mean
    # Avoid division by zero
    denominator = class_mean_hist + 1e-10
    chi_sq = np.sum((hist - class_mean_hist) ** 2 / denominator)

    # Bhattacharyya distance
    bhatt = cv2.compareHist(
        hist.astype(np.float32),
        class_mean_hist.astype(np.float32),
        cv2.HISTCMP_BHATTACHARYYA,
    )

    # Compute z-score of chi-squared distance
    # We'll use the overall deviation
    hist_diff = np.abs(hist - class_mean_hist)
    safe_std = np.maximum(class_std_hist, 1e-10)
    z_scores = hist_diff / safe_std
    max_z = np.max(z_scores)
    mean_z = np.mean(z_scores)

    if mean_z > HISTOGRAM_DEVIATION_THRESHOLD:
        anomalies.append("unusual_histogram")
        details["mean_z_score"] = round(float(mean_z), 3)
        details["max_z_score"] = round(float(max_z), 3)
        details["chi_squared"] = round(float(chi_sq), 5)
        details["bhattacharyya"] = round(float(bhatt), 5)

    # Also flag very unusual overall brightness
    mean_intensity = np.sum(np.arange(256) * hist)
    if mean_intensity > 200 or mean_intensity < 10:
        anomalies.append("extreme_brightness")
        details["mean_intensity"] = round(float(mean_intensity), 2)

    return {"anomalies": anomalies, "details": details}


def detect_watermark(img_gray: np.ndarray) -> dict:
    """Detect potential watermarks (semi-transparent overlays)."""
    anomalies = []
    details = {}

    # Watermarks often show as consistent low-contrast patterns
    # Check center region for unusual repeating patterns
    h, w = img_gray.shape
    center = img_gray[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]

    # Apply high-pass filter to detect overlay patterns
    blurred = cv2.GaussianBlur(center, (21, 21), 0)
    highpass = cv2.subtract(center, blurred)

    # Look for structured high-frequency content (watermark text)
    _, thresh = cv2.threshold(highpass, 20, 255, cv2.THRESH_BINARY)
    connected_ratio = np.sum(thresh > 0) / thresh.size

    # Watermarks tend to have moderate connected components in high-pass
    if connected_ratio > 0.10 and connected_ratio < 0.3:
        # Additional check: look for horizontal/vertical alignment (text-like)
        # Use morphological operations
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        dilated = cv2.dilate(thresh, kernel_h, iterations=1)
        text_like_ratio = np.sum(dilated > 0) / dilated.size

        if text_like_ratio > 0.20:
            anomalies.append("possible_watermark")
            details["highpass_ratio"] = round(float(connected_ratio), 4)
            details["text_like_ratio"] = round(float(text_like_ratio), 4)

    return {"anomalies": anomalies, "details": details}


def save_thumbnail(img_bgr: np.ndarray, save_path: Path) -> None:
    """Save a thumbnail of the image for visual review."""
    thumb = cv2.resize(img_bgr, THUMBNAIL_SIZE)
    cv2.imwrite(str(save_path), thumb)


def scan_all_images():
    """Main function: scan all training images for anomalies."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    THUMBNAIL_DIR.mkdir(parents=True, exist_ok=True)

    # Scan all splits
    splits = ["train", "test", "valid"]

    all_results = {}
    summary = {
        "total_images": 0,
        "total_flagged": 0,
        "by_split": {},
        "by_class": {},
        "anomaly_type_counts": defaultdict(int),
        "severity_counts": {"critical": 0, "warning": 0, "info": 0},
    }

    # Per-class tracking for correlation table:
    # class_name -> {anomaly_type -> count}
    class_anomaly_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    # class_name -> total image count
    class_image_counts: dict[str, int] = defaultdict(int)

    for split in splits:
        split_dir = DATA_ROOT / split
        if not split_dir.exists():
            print(f"[WARN] Split directory not found: {split_dir}")
            continue

        print(f"\n{'=' * 60}")
        print(f"Scanning {split} split...")
        print(f"{'=' * 60}")

        split_results = {}
        split_summary = {"total": 0, "flagged": 0, "by_class": {}}

        class_dirs = sorted(split_dir.iterdir())
        for class_dir in class_dirs:
            if not class_dir.is_dir():
                continue

            class_name = class_dir.name
            print(f"\n  Class: {class_name}")

            # Collect all image files
            image_files = sorted(
                list(class_dir.glob("*.png"))
                + list(class_dir.glob("*.jpg"))
                + list(class_dir.glob("*.jpeg"))
                + list(class_dir.glob("*.bmp"))
                + list(class_dir.glob("*.tiff"))
            )

            print(f"  Found {len(image_files)} images")

            # First pass: compute class histograms for histogram anomaly detection
            class_hists = []
            loaded_images = {}

            for img_path in image_files:
                try:
                    img_bgr = cv2.imread(str(img_path))
                    if img_bgr is None:
                        print(f"    [ERROR] Could not load: {img_path.name}")
                        continue
                    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                    hist = compute_histogram(img_gray)
                    class_hists.append(hist)
                    loaded_images[img_path] = (img_bgr, img_gray, hist)
                except Exception as e:
                    print(f"    [ERROR] Failed to process {img_path.name}: {e}")

            if not class_hists:
                continue

            # Compute class histogram statistics
            class_hists_arr = np.array(class_hists)
            class_mean_hist = np.mean(class_hists_arr, axis=0)
            class_std_hist = np.std(class_hists_arr, axis=0)

            # Second pass: run all anomaly detections
            class_flagged = []
            class_total = 0

            for img_path, (img_bgr, img_gray, hist) in loaded_images.items():
                class_total += 1
                all_anomalies = []
                all_details = {}

                # 1. Color detection
                result = detect_color_anomalies(img_bgr)
                all_anomalies.extend(result["anomalies"])
                all_details.update(result["details"])

                # 2. Text detection
                result = detect_text_overlays(img_gray)
                all_anomalies.extend(result["anomalies"])
                all_details.update(result["details"])

                # 3. Edge/border anomalies
                result = detect_edge_anomalies(img_gray)
                all_anomalies.extend(result["anomalies"])
                all_details.update(result["details"])

                # 4. Markup detection (circles, arrows, lines)
                result = detect_markup(img_gray, img_bgr)
                all_anomalies.extend(result["anomalies"])
                all_details.update(result["details"])

                # 5. Histogram anomalies
                result = detect_histogram_anomalies(hist, class_mean_hist, class_std_hist)
                all_anomalies.extend(result["anomalies"])
                all_details.update(result["details"])

                # 6. Watermark detection
                result = detect_watermark(img_gray)
                all_anomalies.extend(result["anomalies"])
                all_details.update(result["details"])

                if all_anomalies:
                    rel_path = str(img_path.relative_to(PROJECT_ROOT))
                    severity = classify_severity(all_anomalies)
                    entry = {
                        "path": str(img_path),
                        "relative_path": rel_path,
                        "split": split,
                        "class": class_name,
                        "severity": severity,
                        "anomalies": all_anomalies,
                        "details": all_details,
                    }
                    class_flagged.append(entry)

                    # Save thumbnail
                    thumb_name = f"{split}_{class_name}_{img_path.stem}.png"
                    # Sanitize filename
                    thumb_name = thumb_name.replace(" ", "_").replace("(", "").replace(")", "")
                    save_thumbnail(img_bgr, THUMBNAIL_DIR / thumb_name)

                    # Count anomaly types and severity
                    for anomaly in all_anomalies:
                        summary["anomaly_type_counts"][anomaly] += 1
                        class_anomaly_counts[class_name][anomaly] += 1
                    summary["severity_counts"][severity] += 1

            # Track per-class image totals (across all splits)
            class_image_counts[class_name] += class_total

            # Update summaries
            split_summary["total"] += class_total
            split_summary["flagged"] += len(class_flagged)
            split_summary["by_class"][class_name] = {
                "total": class_total,
                "flagged": len(class_flagged),
            }

            full_class_key = f"{split}/{class_name}"
            summary["by_class"][full_class_key] = {
                "total": class_total,
                "flagged": len(class_flagged),
            }

            if class_flagged:
                split_results[class_name] = class_flagged
                print(f"  => FLAGGED: {len(class_flagged)}/{class_total} images")
                for entry in class_flagged:
                    sev_tag = entry["severity"].upper()
                    print(f"     - [{sev_tag}] {Path(entry['path']).name}: {', '.join(entry['anomalies'])}")
            else:
                print(f"  => All {class_total} images passed checks")

        summary["total_images"] += split_summary["total"]
        summary["total_flagged"] += split_summary["flagged"]
        summary["by_split"][split] = split_summary
        all_results[split] = split_results

    # === Generate Reports ===
    print(f"\n{'=' * 60}")
    print("ANOMALY DETECTION REPORT")
    print(f"{'=' * 60}")

    print(f"\nTotal images scanned: {summary['total_images']}")
    print(f"Total flagged: {summary['total_flagged']}")
    print(f"Flag rate: {summary['total_flagged'] / max(summary['total_images'], 1) * 100:.1f}%")

    print("\n--- By Split ---")
    for split_name, split_info in summary["by_split"].items():
        print(f"  {split_name}: {split_info['flagged']}/{split_info['total']} flagged")
        for cls_name, cls_info in split_info["by_class"].items():
            print(f"    {cls_name}: {cls_info['flagged']}/{cls_info['total']}")

    print("\n--- Anomaly Types ---")
    for anomaly_type, count in sorted(summary["anomaly_type_counts"].items(), key=lambda x: -x[1]):
        print(f"  {anomaly_type}: {count}")

    # --- Severity Summary ---
    sev = summary["severity_counts"]
    print("\n--- By Severity ---")
    print(f"  CRITICAL: {sev['critical']} images (must investigate - text/color that could bias model)")
    print(f"  WARNING:  {sev['warning']} images (review - border/saturation artifacts)")
    print(f"  INFO:     {sev['info']} images (informational - minor anomalies)")

    # --- Per-class anomaly correlation table ---
    all_anomaly_types = sorted({a for counts in class_anomaly_counts.values() for a in counts})
    if all_anomaly_types and class_image_counts:
        print("\n--- Per-Class Anomaly Rates (poison pill detection) ---")
        # Column widths
        class_col_w = max(len(c) for c in class_image_counts) + 2
        col_w = 16
        header = " " * class_col_w + "".join(f"{a:>{col_w}}" for a in all_anomaly_types)
        print(header)

        for cls in sorted(class_image_counts):
            total = class_image_counts[cls]
            if total == 0:
                continue
            row = f"{cls:<{class_col_w}}"
            for a_type in all_anomaly_types:
                count = class_anomaly_counts[cls].get(a_type, 0)
                rate = count / total * 100
                cell = f"{rate:.1f}%"
                row += f"{cell:>{col_w}}"
            print(row)

        # Flag suspicious class-correlated patterns
        print()
        for a_type in all_anomaly_types:
            rates = {}
            for cls in class_image_counts:
                total = class_image_counts[cls]
                if total > 0:
                    rates[cls] = class_anomaly_counts[cls].get(a_type, 0) / total
            if rates:
                max_rate = max(rates.values())
                min_rate = min(rates.values())
                if max_rate > 0.1 and max_rate > 3 * max(min_rate, 0.01):
                    worst_class = max(rates, key=rates.get)  # type: ignore[arg-type]
                    print(
                        f"  [!] '{a_type}' is heavily correlated with class "
                        f"'{worst_class}' ({max_rate * 100:.1f}% vs others "
                        f"<= {min_rate * 100:.1f}%) — potential bias source"
                    )

    # Save detailed JSON report
    # Convert defaultdict for JSON serialization
    summary["anomaly_type_counts"] = dict(summary["anomaly_type_counts"])

    # Build per-class correlation data for JSON
    correlation_data = {}
    for cls in sorted(class_image_counts):
        total = class_image_counts[cls]
        if total > 0:
            correlation_data[cls] = {
                "total_images": total,
                "anomaly_rates": {
                    a: round(class_anomaly_counts[cls].get(a, 0) / total * 100, 1) for a in all_anomaly_types
                },
            }

    report = {
        "summary": summary,
        "per_class_correlation": correlation_data,
        "flagged_images": all_results,
    }

    report_path = REPORT_DIR / "anomaly_report.json"
    with report_path.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"\nDetailed report saved to: {report_path}")

    # Save flat list of flagged image paths
    flagged_paths = []
    for split_data in all_results.values():
        for class_data in split_data.values():
            for entry in class_data:
                flagged_paths.append(entry["path"])

    flagged_paths_file = REPORT_DIR / "flagged_image_paths.txt"
    with flagged_paths_file.open("w") as f:
        for p in sorted(flagged_paths):
            f.write(p + "\n")
    print(f"Flagged image paths saved to: {flagged_paths_file}")

    # Save CSV report for easy spreadsheet analysis
    csv_path = REPORT_DIR / "anomaly_report.csv"
    with csv_path.open("w") as f:
        f.write("split,class,filename,severity,anomalies,details\n")
        for split_data in all_results.values():
            for class_data in split_data.values():
                for entry in class_data:
                    filename = Path(entry["path"]).name
                    anomalies_str = "|".join(entry["anomalies"])
                    details_str = json.dumps(entry["details"]).replace(",", ";")
                    f.write(
                        f"{entry['split']},{entry['class']},{filename},"
                        f"{entry['severity']},{anomalies_str},{details_str}\n"
                    )
    print(f"CSV report saved to: {csv_path}")

    # Print all flagged files
    if flagged_paths:
        print(f"\n{'=' * 60}")
        print("ALL FLAGGED IMAGE PATHS:")
        print(f"{'=' * 60}")
        for p in sorted(flagged_paths):
            print(f"  {p}")

    print(f"\nThumbnails of flagged images saved to: {THUMBNAIL_DIR}")
    print(f"Total thumbnails: {len(flagged_paths)}")

    return summary, all_results


if __name__ == "__main__":
    summary, results = scan_all_images()
    sys.exit(0)
