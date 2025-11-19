"""Data validation utilities for the football data analysis project."""

from typing import List, Optional, Union
import pandas as pd


def require_columns(df: pd.DataFrame, columns: List[str]) -> None:
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def assert_bounds(
    df: pd.DataFrame,
    column: str,
    min_val: Optional[Union[int, float]] = None,
    max_val: Optional[Union[int, float]] = None,
    allow_null: bool = True,
) -> None:
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")

    series = df[column]

    if not allow_null and series.isnull().any():
        raise ValueError(f"Column '{column}' contains null values")

    non_null_series = series.dropna()

    if min_val is not None and (non_null_series < min_val).any():
        min_violations = non_null_series[non_null_series < min_val]
        raise ValueError(
            f"Column '{column}' has {len(min_violations)} values below minimum {min_val}"
        )

    if max_val is not None and (non_null_series > max_val).any():
        max_violations = non_null_series[non_null_series > max_val]
        raise ValueError(
            f"Column '{column}' has {len(max_violations)} values above maximum {max_val}"
        )


def assert_no_duplicates(df: pd.DataFrame, columns: List[str]) -> None:
    require_columns(df, columns)

    duplicates = df.duplicated(subset=columns, keep=False)
    if duplicates.any():
        num_duplicates = duplicates.sum()
        raise ValueError(
            f"Found {num_duplicates} duplicate rows based on columns: {columns}"
        )


def assert_valid_types(df: pd.DataFrame, type_map: dict) -> None:
    for column, expected_type in type_map.items():
        if column not in df.columns:
            continue

        actual_type = df[column].dtype

        # Handle pandas dtype comparisons
        if hasattr(expected_type, "name"):
            expected_name = expected_type.name
        else:
            expected_name = str(expected_type)

        if str(actual_type) != expected_name:
            raise ValueError(
                f"Column '{column}' has type {actual_type}, expected {expected_type}"
            )
