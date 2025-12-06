from typing import Dict, Optional

from src.common.geometry import distance_to_goal, shot_angle
from src.tasks.xg.transform.build_shots import Shot


BODY_PART_CATEGORIES = ["Right Foot", "Left Foot", "Head", "Other"]


def _normalize_body_part(body_part: Optional[str]) -> str:
    """
    Normalize raw body_part. Defults to "Right Foot".
    """
    if not body_part:
        return "Right Foot"

    value = body_part.strip()

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


def encode_shot_for_xg(shot: Shot) -> Dict[str, float]:
    """
    Turn a domain Shot object into a flat dict of model-ready features
    for xG modelling.
    """
    # Compute distance and angle if not provided
    if shot.distance_to_goal is not None:
        distance = float(shot.distance_to_goal)
    else:
        if shot.x is None or shot.y is None:
            raise ValueError(
                "Shot has no distance_to_goal and missing x/y coordinates; "
            )
        distance = float(distance_to_goal(shot.x, shot.y))

    # Compute angle if not provided
    if shot.shot_angle is not None:
        angle = float(shot.shot_angle)
    else:
        if shot.x is None or shot.y is None:
            raise ValueError("Shot has no shot_angle and missing x/y coordinates; ")
        angle = float(shot_angle(shot.x, shot.y))

    features: Dict[str, float] = {
        "shot_distance": distance,
        "shot_angle": angle,
    }

    # Numerical / boolean features
    features.update(
        {
            "is_open_play": float(shot.is_open_play),
            "one_on_one": float(shot.one_on_one),
            "is_penalty": float(shot.is_penalty),
            "is_freekick": float(shot.is_freekick),
            "first_time": float(shot.first_time),
            "period": float(shot.period),
            "minute": float(shot.minute),
            "second": float(shot.second),
        }
    )

    # Body part one-hot encoding
    normalized_body_part = _normalize_body_part(shot.body_part)
    for part in BODY_PART_CATEGORIES:
        features[f"body_part_{part}"] = 1.0 if normalized_body_part == part else 0.0

    return features
