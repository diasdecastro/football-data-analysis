"""
src/serve/routers/xg_router.py

FastAPI router for xG prediction endpoints with multi-model support.
"""

from fastapi import APIRouter, HTTPException, status, Query
from fastapi.responses import StreamingResponse
import pandas as pd
import numpy as np
import math
import io

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from src.common.geometry import distance_to_goal, shot_angle
from src.serve.loaders import (
    get_xg_model,
    discover_models,
    set_current_model,
    get_current_model_id,
)
from src.serve.schemas import ShotRequest, ShotResponse, HealthResponse


router = APIRouter(prefix="/v1/xg", tags=["xG (Expected Goals)"])


def interpret_xg(xg_value: float) -> str:
    if xg_value > 0.3:
        return "Excellent"
    elif xg_value > 0.15:
        return "Good"
    elif xg_value > 0.08:
        return "Average"
    else:
        return "Poor"


@router.get(
    "/models",
    summary="List available models",
    description="Get list of all available xG models with metadata",
)
async def list_models():
    models = discover_models()
    current_id = get_current_model_id()

    return {"models": models, "current_model": current_id, "count": len(models)}


@router.post(
    "/models/select",
    summary="Select active model",
    description="Change the currently active model for predictions",
)
async def select_model(model_id: str = Query(..., description="Model ID to activate")):
    """
    Select which model to use for predictions.
    """
    try:
        set_current_model(model_id)
        models = discover_models()
        selected = next((m for m in models if m["model_id"] == model_id), None)

        return {
            "status": "success",
            "selected_model": selected,
            "message": f"Model '{model_id}' is now active",
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.post(
    "/score",
    response_model=ShotResponse,
    summary="Calculate xG for a shot",
    description="Calculate Expected Goals (xG) probability for a shot given coordinates",
)
async def score_xg(
    shot: ShotRequest,
    model_id: str = Query(None, description="Optional: specific model to use"),
) -> ShotResponse:
    """
    Calculate xG for a single shot using the current or specified model.
    """
    try:
        model = get_xg_model(model_id)

        shot_distance = distance_to_goal(shot.x, shot.y)
        shot_angle_rad = shot_angle(shot.x, shot.y)

        shot_angle_deg = math.degrees(shot_angle_rad)

        # Build features with one-hot encoded body_part
        features = pd.DataFrame(
            {
                "shot_distance": [shot_distance],
                "shot_angle": [shot_angle_rad],
                "body_part_Head": [1 if shot.body_part == "Head" else 0],
                "body_part_Left Foot": [1 if shot.body_part == "Left Foot" else 0],
                "body_part_Other": [1 if shot.body_part == "Other" else 0],
                "body_part_Right Foot": [1 if shot.body_part == "Right Foot" else 0],
            }
        )

        if model is None or getattr(model, "predict_proba", None) is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Model does not support probability predictions",
            )

        xg_probability = model.predict_proba(features)[0, 1]

        response = ShotResponse(
            xG=round(float(xg_probability), 4),
            shot_distance=round(float(shot_distance), 2),
            shot_angle=round(float(shot_angle_deg), 2),
            quality=interpret_xg(xg_probability),
        )

        return response

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error calculating xG: {str(e)}",
        )


@router.get(
    "/heatmap",
    summary="Generate xG heatmap overlay",
    description="Generate a transparent heatmap overlay that matches the pitch canvas",
)
async def generate_heatmap(
    model_id: str = Query(None, description="Optional: specific model to use"),
    resolution: int = Query(
        50, description="Grid resolution (higher = more detailed)", ge=20, le=200
    ),
):
    """
    Generate an xG heatmap overlay for the pitch canvas.
    """
    try:
        model = get_xg_model(model_id)

        x_range = np.linspace(60, 120, resolution)
        y_range = np.linspace(0, 80, int(resolution * 0.67))

        X_grid, Y_grid = np.meshgrid(x_range, y_range)
        xg_grid = np.zeros_like(X_grid)

        for i in range(len(y_range)):
            for j in range(len(x_range)):
                x, y = X_grid[i, j], Y_grid[i, j]

                shot_distance = distance_to_goal(x, y)
                shot_angle_rad = shot_angle(x, y)

                # Use Right Foot as default for heatmap
                features = pd.DataFrame(
                    {
                        "shot_distance": [shot_distance],
                        "shot_angle": [shot_angle_rad],
                        "body_part_Head": [0],
                        "body_part_Left Foot": [0],
                        "body_part_Other": [0],
                        "body_part_Right Foot": [1],
                    }
                )

                if model is None or getattr(model, "predict_proba", None) is None:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="Model does not support probability predictions",
                    )

                xg_grid[i, j] = model.predict_proba(features)[0, 1]

        # Create the plot with exact canvas dimensions (700x467 pixels)
        # Convert pixels to inches at 100 DPI for clean matching
        fig_width = 7.0  # 700px / 100 DPI
        fig_height = 4.67  # 467px / 100 DPI

        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=100)

        # Poor shot to Excellent shot (red -> yellow -> green)
        colors = ["#d32f2f", "#f57c00", "#fbc02d", "#689f38", "#388e3c"]
        cmap = LinearSegmentedColormap.from_list("xg", colors, N=100)

        # Plot heatmap - fill entire axes area
        contour = ax.contourf(X_grid, Y_grid, xg_grid, levels=20, cmap=cmap, alpha=1.0)

        ax.axis("off")  # Hide all axes decorations

        # Save to bytes buffer with exact dimensions, no padding
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

        return StreamingResponse(buf, media_type="image/png")

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating heatmap: {str(e)}",
        )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check if xG service is loaded and ready",
)
async def health_check() -> HealthResponse:
    try:
        model = get_xg_model()
        model_loaded = model is not None
        models = discover_models()

        return HealthResponse(
            status="healthy" if model_loaded else "degraded",
            model_loaded=model_loaded,
            version="1.0.0",
        )
    except Exception:
        return HealthResponse(status="unhealthy", model_loaded=False, version="1.0.0")
