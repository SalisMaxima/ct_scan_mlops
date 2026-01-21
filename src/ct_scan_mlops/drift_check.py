from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from evidently.legacy.metric_preset import DataDriftPreset
from evidently.legacy.report import Report
from pandas.errors import EmptyDataError


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"CSV not found: {path}\n"
            f"Tip: generate it first (e.g. with an image-stats script)."
        )

    if path.stat().st_size == 0:
        raise ValueError(
            f"CSV is empty (0 bytes): {path}\n"
            f"Tip: you created the file but never wrote rows to it."
        )

    try:
        df = pd.read_csv(path)
    except EmptyDataError as e:
        raise ValueError(
            f"CSV has no readable columns: {path}\n"
            f"Tip: ensure the file has a header row and at least one data row."
        ) from e

    if df.empty:
        raise ValueError(
            f"CSV has headers but no rows: {path}\n"
            f"Tip: your extraction ran but produced zero samples."
        )

    return df



def run_drift(reference_csv: Path, current_csv: Path, out_html: Path) -> None:
    ref = _load_csv(reference_csv)
    cur = _load_csv(current_csv)

    # Keep numeric columns only (image stats should be numeric).
    ref_num = ref.select_dtypes(include="number")
    cur_num = cur.select_dtypes(include="number")

    if ref_num.shape[1] == 0:
        raise ValueError(
            f"No numeric columns found in reference CSV: {reference_csv}\n"
            f"Columns: {list(ref.columns)}"
        )

    # Align columns (in case one CSV has extra columns)
    common_cols = [c for c in ref_num.columns if c in cur_num.columns]
    if not common_cols:
        raise ValueError(
            "No overlapping numeric columns between reference and current CSVs.\n"
            f"Reference numeric cols: {list(ref_num.columns)}\n"
            f"Current numeric cols: {list(cur_num.columns)}"
        )

    ref_num = ref_num[common_cols].copy()
    cur_num = cur_num[common_cols].copy()

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_num, current_data=cur_num)

    out_html.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(out_html))
    print(f"Saved drift report to: {out_html}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a data drift report from two CSV files.")
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("data/drift/reference.csv"),
        help="Path to reference CSV (baseline, usually training stats).",
    )
    parser.add_argument(
        "--current",
        type=Path,
        default=Path("data/drift/current.csv"),
        help="Path to current CSV (new/production stats).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/drift/drift_report.html"),
        help="Where to write the HTML drift report.",
    )
    args = parser.parse_args()
    run_drift(args.reference, args.current, args.out)


if __name__ == "__main__":
    main()
