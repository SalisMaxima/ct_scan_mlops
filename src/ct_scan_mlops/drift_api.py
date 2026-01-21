# src/ct_scan_mlops/drift_api.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from evidently.legacy.report import Report
from evidently.legacy.metric_preset import DataDriftPreset, DataQualityPreset


DEFAULT_DRIFT_DIR = Path("data") / "drift"
REFERENCE_PATH = Path(os.environ.get("DRIFT_REFERENCE_PATH", str(DEFAULT_DRIFT_DIR / "reference.csv")))
CURRENT_PATH = Path(os.environ.get("DRIFT_CURRENT_PATH", str(DEFAULT_DRIFT_DIR / "current.csv")))
REPORT_PATH = Path(os.environ.get("DRIFT_REPORT_PATH", str(DEFAULT_DRIFT_DIR / "drift_report.html")))


def _load_csv_strict(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"CSV is empty: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"CSV has no rows: {path}")
    return df


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    num = df.select_dtypes(include=["number"]).columns
    if len(num) == 0:
        return df

    df[num] = df[num].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=num, how="any")

    constant = [c for c in num if df[c].nunique(dropna=True) <= 1]
    if constant:
        df = df.drop(columns=constant)

    return df


def _first_found(obj: Any, key: str):
    """Return first occurrence of `key` in nested dict/list, else None."""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            found = _first_found(v, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for it in obj:
            found = _first_found(it, key)
            if found is not None:
                return found
    return None


def _extract_drift_summary(report_dict: dict[str, Any]) -> dict[str, Any]:
    # Most robust across Evidently versions:
    dataset_drift = _first_found(report_dict, "dataset_drift")
    share = _first_found(report_dict, "share_drifted_columns")
    n_cols = _first_found(report_dict, "number_of_columns")
    n_drifted = _first_found(report_dict, "number_of_drifted_columns")

    return {
        "dataset_drift": dataset_drift,
        "share_drifted_columns": share,
        "number_of_columns": n_cols,
        "number_of_drifted_columns": n_drifted,
    }



class DriftResponse(BaseModel):
    reference_rows: int
    current_rows: int
    dataset_drift: bool | None = None
    share_drifted_columns: float | None = None
    number_of_columns: int | None = None
    number_of_drifted_columns: int | None = None
    report_written: bool
    report_path: str


app = FastAPI(title="Drift Detection API", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "ok": "true",
        "reference_path": str(REFERENCE_PATH),
        "current_path": str(CURRENT_PATH),
        "report_path": str(REPORT_PATH),
    }


@app.get("/drift", response_model=DriftResponse)
def drift(write_html: bool = False) -> DriftResponse:
    try:
        ref = _clean(_load_csv_strict(REFERENCE_PATH))
        cur = _clean(_load_csv_strict(CURRENT_PATH))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
    report.run(reference_data=ref, current_data=cur)

    d = report.as_dict()
    drift_summary = _extract_drift_summary(d)

    if write_html:
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        report.save_html(str(REPORT_PATH))
        report_written = True
    else:
        report_written = False

    return DriftResponse(
        reference_rows=int(len(ref)),
        current_rows=int(len(cur)),
        report_written=report_written,
        report_path=str(REPORT_PATH),
        **drift_summary,
    )
