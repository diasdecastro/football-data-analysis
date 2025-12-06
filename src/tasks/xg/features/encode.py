from typing import Any, Dict, Optional

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


# TODO: create a Shot-Like type
def encode_shot_for_xg(shot: Any) -> Dict[str, float]:
    """
    Turn a Shot-Like object into a flat dict of model-ready features
    for xG modelling.
    """
    # Compute distance and angle if not provided
    if getattr(shot, "distance_to_goal", None) is not None:
        distance = float(shot.distance_to_goal)
    else:
        if getattr(shot, "x", None) is None or getattr(shot, "y", None) is None:
            raise ValueError("Shot has no distance_to_goal and missing x/y coordinates")
        distance = float(distance_to_goal(shot.x, shot.y))

    # Compute angle if not provided
    if getattr(shot, "shot_angle", None) is not None:
        angle = float(shot.shot_angle)
    else:
        if getattr(shot, "x", None) is None or getattr(shot, "y", None) is None:
            raise ValueError("Shot has no shot_angle and missing x/y coordinates")
        angle = float(shot_angle(shot.x, shot.y))

    features: Dict[str, float] = {
        "shot_distance": distance,
        "shot_angle": angle,
    }

    # Numerical / boolean feature
    for key in [
        "is_open_play",
        "one_on_one",
        "is_penalty",
        "is_freekick",
        "first_time",
        "period",
        "minute",
        "second",
    ]:
        if hasattr(shot, key):
            value = getattr(shot, key)
            if isinstance(value, bool):
                features[key] = float(value)
            elif isinstance(value, (int, float)):
                features[key] = float(value)
            else:
                continue
        else:
            continue

    # Body part one-hot encoding
    if hasattr(shot, "body_part"):
        normalized_body_part = _normalize_body_part(shot.body_part)
        for part in BODY_PART_CATEGORIES:
            features[f"body_part_{part}"] = 1.0 if normalized_body_part == part else 0.0

    return features
