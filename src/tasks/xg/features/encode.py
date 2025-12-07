from typing import Any, Dict

import pandas as pd

from src.tasks.xg.features.pipeline import (
    DEFAULT_FEATURE_COLUMNS,
    FeaturePipeline,
    build_feature_pipeline,
)
from src.tasks.xg.transform.build_shots import Shot


SHOT_FIELDS = [
    "x",
    "y",
    "shot_distance",
    "shot_angle",
    "body_part",
    "is_open_play",
    "one_on_one",
    "is_penalty",
    "is_freekick",
    "first_time",
    "period",
    "minute",
    "second",
]

_PIPELINE: FeaturePipeline | None = None


def _get_pipeline() -> FeaturePipeline:
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = build_feature_pipeline()
    return _PIPELINE


def _shot_to_dict(shot: Any) -> Dict[str, Any]:
    """
    Extract the raw fields required by the feature pipeline from a Shot-like object.
    """
    raw: Dict[str, Any] = {}
    for field in SHOT_FIELDS:
        value = getattr(shot, field, None)

        # Shots produced upstream store distance/angle as distance_to_goal/shot_angle
        if value is None and isinstance(shot, Shot):
            if field == "shot_distance":
                value = getattr(shot, "distance_to_goal", None)
            elif field == "shot_angle":
                value = getattr(shot, "shot_angle", None)

        raw[field] = value

    return raw


def encode_shot_for_xg(shot: Any) -> Dict[str, float]:
    """
    Turn a Shot-like object into a flat dict of model-ready features using the shared pipeline.
    """
    pipeline = _get_pipeline()
    raw_dict = _shot_to_dict(shot)

    features_df = pipeline.transform(pd.DataFrame([raw_dict]))
    features_series = features_df.iloc[0]

    return {col: float(features_series[col]) for col in DEFAULT_FEATURE_COLUMNS}
