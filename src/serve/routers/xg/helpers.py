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


def build_features_for_model(
    model,
    shot_distance: float,
    shot_angle: float,
    body_part: str = "Right Foot",
    is_open_play: bool = True,
    one_on_one: bool = False,
    **extra_features,
) -> pd.DataFrame:
    """
    Dynamically build feature DataFrame based on model's expected features.
    
    Args:
        model: Trained model with feature_names_in_ attribute
        shot_distance: Distance to goal in meters
        shot_angle: Angle to goal in radians
        body_part: Body part used for shot
        is_open_play: Whether shot was from open play
        one_on_one: Whether shot was one-on-one with goalkeeper
        **extra_features: Additional features for future model versions
    
    Returns:
        DataFrame with features in the exact order the model expects
    """
    feature_names = getattr(model, "feature_names_in_", None)

    if feature_names is None:
        # Fallback for models without feature_names_in_
        return pd.DataFrame(
            {
                "shot_distance": [shot_distance],
                "shot_angle": [shot_angle],
            }
        )

    features_dict = {}

    # Known features
    if "shot_distance" in feature_names:
        features_dict["shot_distance"] = [shot_distance]
    if "shot_angle" in feature_names:
        features_dict["shot_angle"] = [shot_angle]
    if "is_open_play" in feature_names:
        features_dict["is_open_play"] = [1 if is_open_play else 0]
    if "one_on_one" in feature_names:
        features_dict["one_on_one"] = [1 if one_on_one else 0]

    # Handle body_part one-hot encoding
    body_part_features = [f for f in feature_names if f.startswith("body_part_")]
    for feat in body_part_features:
        part_name = feat.replace("body_part_", "")
        features_dict[feat] = [1 if body_part == part_name else 0]

    # Handle unknown features with defaults
    for feat in feature_names:
        if feat not in features_dict:
            # Check if provided in extra_features
            if feat in extra_features:
                features_dict[feat] = [extra_features[feat]]
            else:
                # Default to 0 for unknown features
                features_dict[feat] = [0]

    # Return DataFrame with columns in the EXACT order model expects
    return pd.DataFrame(features_dict)[list(feature_names)]


def interpret_xg(xg_value: float) -> str:
    """
    Convert xG probability to human-readable quality rating.
    
    Args:
        xg_value: xG probability between 0 and 1
    
    Returns:
        Quality rating string
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
    
    Args:
        x: X coordinate (meters from left touchline)
        y: Y coordinate (meters from bottom touchline)
    
    Returns:
        Tuple of (distance, angle_radians, angle_degrees)
    """
    shot_distance = distance_to_goal(x, y)
    shot_angle_rad = shot_angle(x, y)
    shot_angle_deg = math.degrees(shot_angle_rad)
    
    return shot_distance, shot_angle_rad, shot_angle_deg


def generate_xg_heatmap(
    model,
    resolution: int = 50,
) -> io.BytesIO:
    """
    Generate an xG heatmap overlay for the pitch canvas.
    
    Args:
        model: Trained xG model
        resolution: Grid resolution (higher = more detailed)
    
    Returns:
        BytesIO buffer containing PNG image
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
            features = build_features_for_model(
                model, shot_distance, shot_angle_rad, "Right Foot"
            )

            xg_grid[i, j] = model.predict_proba(features)[0, 1]

    # Create the plot with exact canvas dimensions (700x467 pixels)
    fig_width = 7.0  # 700px / 100 DPI
    fig_height = 4.67  # 467px / 100 DPI

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=100)

    # Color gradient: Poor shot to Excellent shot (red -> yellow -> green)
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
