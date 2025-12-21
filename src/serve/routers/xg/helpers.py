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
from src.tasks.xg.train.train_xg import prepare_features


def build_features_from_request(model, shot_req: ShotRequest):
    """
    Build feature DataFrame from ShotRequest.
    Uses prepare_features from training script for consistency.
    """
    # Calculate shot distance and angle from coordinates
    shot_distance, shot_angle_rad, shot_angle_deg = calculate_shot_features(
        shot_req.x, shot_req.y
    )

    # Build raw DataFrame with shot data
    raw_df = pd.DataFrame(
        [
            {
                "x": shot_req.x,
                "y": shot_req.y,
                "end_x": getattr(shot_req, "end_x", 120.0),  # Goal center
                "end_y": getattr(shot_req, "end_y", 40.0),  # Goal center
                "shot_distance": shot_distance,
                "shot_angle": shot_angle_rad,
                "body_part": shot_req.body_part,
                "is_open_play": shot_req.is_open_play,
                "one_on_one": shot_req.one_on_one,
                "is_goal": 0,  # Dummy value, not used for prediction
            }
        ]
    )

    # Use the same preprocessing as training
    X, _ = prepare_features(raw_df)

    return X, shot_distance, shot_angle_rad, shot_angle_deg


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

            # Use Right Foot as default for heatmap
            # TODO: add (optional) parameters for all features
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
