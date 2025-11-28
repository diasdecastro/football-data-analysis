from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

LOG_DIR = Path("data/monitoring")
LOG_PATH = LOG_DIR / "xg_inference_log.parquet"


# Simple monitoring for learning purposes
def log_xg_inference(
    *,
    shot_distance: float,
    shot_angle: float,
    xg: float,
    model_id: str,
) -> None:
    """
    Add a record for xG inference
    """
    record = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "shot_distance": float(shot_distance),
        "shot_angle": float(shot_angle),
        "xg": float(xg),
        "model_id": model_id,
    }
    df = pd.DataFrame([record])

    if LOG_PATH.exists():
        existing = pd.read_parquet(LOG_PATH)
        combined = pd.concat([existing, df], ignore_index=True)
        combined.to_parquet(LOG_PATH, index=False)
    else:
        df.to_parquet(LOG_PATH, index=False)
