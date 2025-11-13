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
    """Get path to bronze data directory or subdirectory."""
    return BRONZE_ROOT / subpath


def silver(subpath: str = "") -> Path:
    """Get path to silver data directory or subdirectory."""
    return SILVER_ROOT / subpath


def gold(subpath: str = "") -> Path:
    """Get path to gold data directory or subdirectory."""
    return GOLD_ROOT / subpath


def shots_silver_path() -> Path:
    """Get the standard path for silver shots data."""
    return SILVER_ROOT / "shots.parquet"


def write_table(
    df: pd.DataFrame, path: Union[str, Path], index: bool = False, **kwargs
) -> None:
    """Write DataFrame to parquet file, creating directories as needed."""
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
