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
TEMPORAL_FEATURES = ["period", "minute", "second"]
GEOMETRY_FEATURES = ["shot_distance", "shot_angle"]
BODY_PART_FEATURES = [f"body_part_{name}" for name in BODY_PART_CATEGORIES]

DEFAULT_FEATURE_COLUMNS: List[str] = (
    GEOMETRY_FEATURES + NUMERIC_BOOL_FEATURES + TEMPORAL_FEATURES + BODY_PART_FEATURES
)


def normalize_body_part(body_part: Optional[str]) -> str:
    """
    Normalize body_part values to a limited set of categories.
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
class GeometryFeatureStep(FeatureStep):
    """
    Ensure shot distance/angle columns exist by computing them from x/y when needed.
    """

    x_col: str = "x"
    y_col: str = "y"
    distance_col: str = "shot_distance"
    angle_col: str = "shot_angle"
    name: str = "geometry"

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()

        if self.distance_col not in data.columns:
            data[self.distance_col] = pd.NA
        if self.angle_col not in data.columns:
            data[self.angle_col] = pd.NA

        has_x = self.x_col in data.columns
        has_y = self.y_col in data.columns
        if has_x and has_y:
            valid_coords = data[self.x_col].notna() & data[self.y_col].notna()
        else:
            valid_coords = pd.Series(False, index=data.index)

        if has_x and has_y:
            missing_distance = data[self.distance_col].isna() & valid_coords
            if missing_distance.any():
                data.loc[missing_distance, self.distance_col] = data.loc[
                    missing_distance, [self.x_col, self.y_col]
                ].apply(
                    lambda row: float(
                        distance_to_goal(row[self.x_col], row[self.y_col])
                    ),
                    axis=1,
                )

            missing_angle = data[self.angle_col].isna() & valid_coords
            if missing_angle.any():
                data.loc[missing_angle, self.angle_col] = data.loc[
                    missing_angle, [self.x_col, self.y_col]
                ].apply(
                    lambda row: float(shot_angle(row[self.x_col], row[self.y_col])),
                    axis=1,
                )

        data[self.distance_col] = pd.to_numeric(
            data[self.distance_col], errors="coerce"
        )
        data[self.angle_col] = pd.to_numeric(data[self.angle_col], errors="coerce")

        return data


@dataclass
class NumericFeatureStep(FeatureStep):
    """
    Coerce numeric/boolean features to floats and fill missing values.
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
    Normalize the `body_part` column and emit one-hot encoded columns.
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
            col_name = f"body_part_{category}"
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
    Factory for the default xG feature pipeline.
    """
    steps: List[FeatureStep] = [
        GeometryFeatureStep(),
        NumericFeatureStep(columns=NUMERIC_BOOL_FEATURES + TEMPORAL_FEATURES),
        BodyPartEncodingStep(),
    ]
    return FeaturePipeline(steps=steps, feature_names=DEFAULT_FEATURE_COLUMNS)
