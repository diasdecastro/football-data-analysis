from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from src.common import io


MONITORING_DIR = Path("data/monitoring")
INFERENCE_LOG_PATH = MONITORING_DIR / "xg_inference_log.parquet"
DRIFT_REPORT_PATH = MONITORING_DIR / "xg_drift_report.html"


def load_reference_data(sample_size: int = 5000) -> pd.DataFrame:
    """Load a sample of the training (gold) data as reference."""
    ref = io.read_table(io.xg_features_gold_path())
    # Only shot_distance and shot_angle
    ref = ref[["shot_distance", "shot_angle"]].copy()
    if len(ref) > sample_size:
        ref = ref.sample(n=sample_size, random_state=42)
    return ref


def load_current_data(sample_size: int = 5000) -> Optional[pd.DataFrame]:
    """Load a sample of recent inferences as current data."""
    if not INFERENCE_LOG_PATH.exists():
        return None
    cur = pd.read_parquet(INFERENCE_LOG_PATH)
    cur = cur[["shot_distance", "shot_angle"]].copy()
    if len(cur) > sample_size:
        cur = cur.tail(sample_size)  # most recent
    return cur


def build_drift_report(output_path: Path | None = None) -> Optional[Path]:
    """
    Build a data drift report comparing training vs recent inferences.
    """
    ref = load_reference_data()
    cur = load_current_data()

    if cur is None or cur.empty:
        return None

    report = Report([DataDriftPreset()])
    report_run = report.run(reference_data=ref, current_data=cur)

    out = output_path or DRIFT_REPORT_PATH
    report_run.save_html(str(out))
    return out
