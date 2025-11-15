from fastapi import APIRouter, HTTPException, status
import pandas as pd
import math

from src.common.geometry import distance_to_goal, shot_angle
from src.serve.loaders import get_xg_model
from src.serve.schemas import ShotRequest, ShotResponse, HealthResponse


router = APIRouter(prefix="/v1/xg", tags=["xG (Expected Goals)"])


def interpret_xg(xg_value: float) -> str:
    """
    Provide human-readable interpretation of xG value.
    """
    if xg_value > 0.3:
        return "Excellent"
    elif xg_value > 0.15:
        return "Good"
    elif xg_value > 0.08:
        return "Average"
    else:
        return "Poor"


@router.post(
    "/score",
    response_model=ShotResponse,
    summary="Calculate xG for a shot",
    description="Calculate Expected Goals (xG) probability for a shot given coordinates",
    responses={
        200: {
            "description": "Successful prediction",
            "content": {
                "application/json": {
                    "example": {
                        "xG": 0.2456,
                        "shot_distance": 12.0,
                        "shot_angle": 18.43,
                        "quality": "Good",
                    }
                }
            },
        },
        500: {"description": "Model prediction error"},
    },
)
async def score_xg(shot: ShotRequest) -> ShotResponse:
    """
    Calculate xG for a single shot.
    """
    try:
        # Get the loaded model
        model = get_xg_model()

        # Calculate features using the same geometry functions as training
        shot_distance = distance_to_goal(shot.x, shot.y)
        shot_angle_rad = shot_angle(shot.x, shot.y)

        # Convert angle to degrees for user-friendly output
        shot_angle_deg = math.degrees(shot_angle_rad)

        # Create feature DataFrame (same format as training)
        features = pd.DataFrame(
            {
                "shot_distance": [shot_distance],
                "shot_angle": [shot_angle_rad],  # Model was trained on radians
            }
        )

        # Predict xG probability
        xg_probability = model.predict_proba(features)[0, 1]

        # Create response
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
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check if xG service is loaded and ready",
)
async def health_check() -> HealthResponse:
    """
    Check health of xG service.

    Returns model status and version information.
    """
    try:
        model = get_xg_model()
        model_loaded = model is not None

        return HealthResponse(
            status="healthy" if model_loaded else "degraded",
            model_loaded=model_loaded,
            version="1.0.0",
        )
    except Exception:
        return HealthResponse(status="unhealthy", model_loaded=False, version="1.0.0")
