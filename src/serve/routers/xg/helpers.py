"""
Helper functions for xG prediction endpoints.
"""

import pandas as pd
import numpy as np
import math
import io

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from src.common.geometry import distance_to_goal, shot_angle
from src.serve.schemas import ShotRequest


def build_features_for_model(
    model,
    **raw_features,
) -> pd.DataFrame:
    """
    Build feature DataFrame based on model's expected features.
    """
    feature_names = getattr(model, "feature_names_in_", None)

    if feature_names is None:
        return pd.DataFrame(
            {
                "shot_distance": [raw_features["shot_distance"]],
                "shot_angle": [raw_features["shot_angle"]],
            }
        )

    features_dict: dict[str, list] = {}

    # Handle body_part one-hot encoding
    body_part_value = raw_features.get("body_part", "Right Foot")
    body_part_features = [f for f in feature_names if f.startswith("body_part_")]
    for feat in body_part_features:
        part_name = feat.replace("body_part_", "")
        features_dict[feat] = [1 if body_part_value == part_name else 0]

    # All other features: take from raw_features or fallback to default
    for feat in feature_names:
        if feat in features_dict:
            continue

        if feat in raw_features:
            features_dict[feat] = [raw_features[feat]]
        else:
            features_dict[feat] = [0]

    return pd.DataFrame(features_dict)[list(feature_names)]


def build_features_from_request(model, shot: ShotRequest):
    """
    Build feature DataFrame from ShotRequest for the given model.
    """
    shot_distance, shot_angle_rad, shot_angle_deg = calculate_shot_features(
        shot.x, shot.y
    )

    features = {
        "shot_distance": float(shot_distance),
        "shot_angle": float(shot_angle_rad),
        "body_part": shot.body_part,
    }

    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is None:
        feature_names = []

    for name in feature_names:
        if name in features or name.startswith("body_part_"):
            continue

        if hasattr(shot, name):
            value = getattr(shot, name)
            if isinstance(value, bool):
                features[name] = int(value)
            elif isinstance(value, (int, float, str)):
                features[name] = value
            else:
                # Convert numpy arrays or other types to scalar
                try:
                    features[name] = float(value)
                except (TypeError, ValueError):
                    features[name] = value

    features_df = build_features_for_model(model, **features)
    return features_df, shot_distance, shot_angle_rad, shot_angle_deg


def interpret_xg(xg_value: float) -> str:
    """
    Convert xG probability to human-readable quality rating.
    """
    if xg_value > 0.3:
        return "Excellent"
    elif xg_value > 0.15:
        return "Good"
    elif xg_value > 0.08:
        return "Average"
    else:
        return "Poor"


def calculate_shot_features(x: float, y: float) -> tuple[float, float, float]:
    """
    Calculate shot distance and angle from coordinates.
    """
    shot_distance = distance_to_goal(x, y)
    shot_angle_rad = shot_angle(x, y)
    shot_angle_deg = math.degrees(shot_angle_rad)

    return float(shot_distance), float(shot_angle_rad), float(shot_angle_deg)


def generate_xg_heatmap(
    model,
    resolution: int = 50,
) -> io.BytesIO:
    """
    Generate an xG heatmap overlay for the pitch canvas.
    """
    # Define pitch area for xG heatmap (attacking half)
    x_range = np.linspace(60, 120, resolution)
    y_range = np.linspace(0, 80, int(resolution * 0.67))

    X_grid, Y_grid = np.meshgrid(x_range, y_range)
    xg_grid = np.zeros_like(X_grid)

    # Calculate xG for each grid point
    for i in range(len(y_range)):
        for j in range(len(x_range)):
            x, y = X_grid[i, j], Y_grid[i, j]

            shot_distance = distance_to_goal(x, y)
            shot_angle_rad = shot_angle(x, y)

            # Use Right Foot as default for heatmap
            features, _, _, _ = build_features_from_request(
                model, ShotRequest(x=x, y=y, body_part="Right Foot")
            )

            xg_grid[i, j] = model.predict_proba(features)[0, 1]

    # Create the plot with exact canvas dimensions (700x467 pixels) (Pitch is not exact)
    fig_width = 7.0  # 700px / 100 DPI
    fig_height = 4.67  # 467px / 100 DPI

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=100)

    colors = ["#d32f2f", "#f57c00", "#fbc02d", "#689f38", "#388e3c"]
    cmap = LinearSegmentedColormap.from_list("xg", colors, N=100)

    # Plot heatmap
    ax.contourf(X_grid, Y_grid, xg_grid, levels=20, cmap=cmap, alpha=1.0)
    ax.axis("off")

    # Save to bytes buffer
    buf = io.BytesIO()
    plt.savefig(
        buf,
        format="png",
        dpi=100,
        bbox_inches="tight",
        pad_inches=0,
        facecolor="none",
        transparent=True,
    )
    buf.seek(0)
    plt.close(fig)

    return buf


def get_model_feature_names(model) -> list[str]:
    """
    Return the ordered list of raw feature names the model expects.
    """
    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is None:
        return ["shot_distance", "shot_angle"]

    return [str(name) for name in feature_names]
