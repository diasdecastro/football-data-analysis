"""IO utilities for the football data analysis project."""

from pathlib import Path
from typing import Union

import pandas as pd

# Data directory paths
DATA_ROOT = Path("data")
BRONZE_ROOT = DATA_ROOT / "bronze"
SILVER_ROOT = DATA_ROOT / "silver"
GOLD_ROOT = DATA_ROOT / "gold"


def bronze(subpath: str = "") -> Path:
    return BRONZE_ROOT / subpath


def silver(subpath: str = "") -> Path:
    return SILVER_ROOT / subpath


def gold(subpath: str = "") -> Path:
    return GOLD_ROOT / subpath


def shots_silver_path() -> Path:
    return silver("shots.parquet")


def xg_features_gold_path(version: str | None = None):
    """Standard location for gold-level xG features."""
    name = (
        "xg_features.parquet" if version is None else f"xg_features_{version}.parquet"
    )
    return gold(name)


def write_table(
    df: pd.DataFrame, path: Union[str, Path], index: bool = False, **kwargs
) -> None:
    """Write DataFrame to parquet file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=index, **kwargs)
    elif path.suffix.lower() == ".csv":
        df.to_csv(path, index=index, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def read_table(path: Union[str, Path]) -> pd.DataFrame:
    """Read DataFrame from parquet or CSV file."""
    path = Path(path)

    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")
