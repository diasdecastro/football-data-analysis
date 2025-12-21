from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import pandas as pd

from src.common.geometry import distance_to_goal, shot_angle

BODY_PART_CATEGORIES = ["Right Foot", "Left Foot", "Head", "Other"]
NUMERIC_BOOL_FEATURES = [
    "is_open_play",
    "one_on_one",
    "is_penalty",
    "is_freekick",
    "first_time",
]
TIME_FEATURES = ["period", "minute", "second"]
SPACE_FEATURES = ["shot_distance", "shot_angle", "x", "y", "end_x", "end_y"]
BODY_PART_FEATURES = [
    f"body_part_{name.lower().replace(' ', '_')}" for name in BODY_PART_CATEGORIES
]

DEFAULT_FEATURE_COLUMNS: List[str] = (
    SPACE_FEATURES + NUMERIC_BOOL_FEATURES + TIME_FEATURES + BODY_PART_FEATURES
)


def normalize_body_part(body_part: Optional[str]) -> str:
    """
    Normalize body_part.
    """
    if not body_part:
        return "Right Foot"

    value = str(body_part).strip()
    if value in BODY_PART_CATEGORIES:
        return value

    lower = value.lower()
    if "right" in lower:
        return "Right Foot"
    if "left" in lower:
        return "Left Foot"
    if "head" in lower:
        return "Head"

    return "Other"


class FeatureStep:
    """
    Base interface for feature pipeline steps.
    """

    name: str

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


@dataclass
class SpaceFeatureStep(FeatureStep):
    """
    Ensure shot distance/angle columns exist by computing them from x/y when needed.

    For inputs:
    - shot_distance (float)
    - shot_angle (float)
    - x (float)
    - y (float)
    """

    x_col: str = "x"
    y_col: str = "y"
    end_x_col: str = "end_x"
    end_y_col: str = "end_y"
    distance_col: str = "shot_distance"
    angle_col: str = "shot_angle"
    name: str = "space"

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()

        # Compute distance if needed
        if self.distance_col not in data.columns:
            if self.x_col in data.columns and self.y_col in data.columns:
                try:
                    data[self.distance_col] = distance_to_goal(
                        data[self.x_col].fillna(0), data[self.y_col].fillna(0)
                    )
                    # NAN if coordinates were missing
                    mask = data[self.x_col].isna() | data[self.y_col].isna()
                    data.loc[mask, self.distance_col] = pd.NA
                except ValueError:
                    pass  # Don't create column if computation fails

        # Compute angle if needed
        if self.angle_col not in data.columns:
            if self.x_col in data.columns and self.y_col in data.columns:
                try:
                    data[self.angle_col] = shot_angle(
                        data[self.x_col].fillna(0), data[self.y_col].fillna(0)
                    )
                    # NAN if coordinates were missing
                    mask = data[self.x_col].isna() | data[self.y_col].isna()
                    data.loc[mask, self.angle_col] = pd.NA
                except ValueError:
                    pass  # Don't create column if computation fails

        # Ensure numeric types for computed columns
        if self.distance_col in data.columns:
            data[self.distance_col] = pd.to_numeric(
                data[self.distance_col], errors="coerce"
            )
        if self.angle_col in data.columns:
            data[self.angle_col] = pd.to_numeric(data[self.angle_col], errors="coerce")

        return data


@dataclass
class NumericFeatureStep(FeatureStep):
    """
    Coerce numeric/boolean features to floats and fill missing values.

    For inputs:
    - is_open_play (bool)
    - one_on_one (bool)
    - is_penalty (bool)
    - is_freekick (bool)
    - first_time (bool)
    - period (int)
    - minute (int)
    - second (int)
    """

    columns: Sequence[str]
    fill_value: float = 0.0
    name: str = "numeric"

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        for column in self.columns:
            if column not in data.columns:
                data[column] = self.fill_value
                continue

            data[column] = data[column].apply(self._coerce_value)

        return data

    def _coerce_value(self, value):
        # Convert value to float
        if value is None:
            return self.fill_value
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(value)
        except (TypeError, ValueError):
            return self.fill_value


@dataclass
class BodyPartEncodingStep(FeatureStep):
    """
    Normalize the `body_part` column.

    For inputs:
    - body_part (str)
    """

    source_col: str = "body_part"
    categories: Sequence[str] = tuple(BODY_PART_CATEGORIES)
    name: str = "body_part"

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        if self.source_col not in data.columns:
            data[self.source_col] = "Right Foot"

        normalized = data[self.source_col].apply(normalize_body_part)
        for category in self.categories:
            # Convert category name to lowercase with underscores
            col_name = f"body_part_{category.lower().replace(' ', '_')}"
            data[col_name] = (normalized == category).astype(float)

        return data


@dataclass
class FeaturePipeline:
    """
    Simple pipeline that sequentially applies feature steps and outputs a fixed column set.
    """

    steps: Sequence[FeatureStep]
    feature_names: Sequence[str] = tuple(DEFAULT_FEATURE_COLUMNS)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        for step in self.steps:
            data = step.transform(data)

        # Ensure all requested columns exist
        for column in self.feature_names:
            if column not in data.columns:
                data[column] = 0.0

        return data[self.feature_names].copy()

    def describe(self) -> dict:
        return {
            "steps": [step.name for step in self.steps],
            "feature_names": list(self.feature_names),
        }


def build_feature_pipeline() -> FeaturePipeline:
    """
    xG feature pipeline.

    Expected/accepted input:
    - x (float): x coordinate of the shot
    - y (float): y coordinate of the shot
    - body_part (str): body part used for the shot
    - is_open_play (bool)
    - one_on_one (bool)
    - is_penalty (bool)
    - is_freekick (bool)
    - first_time (bool)
    - period (int)
    - minute (int)
    - second (int)

    """
    steps: List[FeatureStep] = [
        SpaceFeatureStep(),
        NumericFeatureStep(columns=NUMERIC_BOOL_FEATURES + TIME_FEATURES),
        BodyPartEncodingStep(),
    ]
    return FeaturePipeline(steps=steps, feature_names=DEFAULT_FEATURE_COLUMNS)
