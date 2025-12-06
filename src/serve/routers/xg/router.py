"""
FastAPI router for xG prediction endpoints with multi-model support.
"""

from fastapi import APIRouter, HTTPException, status, Query
from fastapi.responses import StreamingResponse, HTMLResponse

from src.serve.loaders import (
    get_xg_model,
    discover_models,
    set_current_model,
    get_current_model_id,
    load_model_artifact,
)
from src.serve.schemas import (
    ShotRequest,
    ShotResponse,
    HealthResponse,
    XGModelTrainRequest,
    XGModelListResponse,
    SelectModelResponse,
    ModelFeatureSet,
    ModelFeatureListResponse,
)
from src.serve.routers.xg.helpers import (
    build_features_from_request,
    interpret_xg,
    generate_xg_heatmap,
    get_model_feature_names,
)
from src.tasks.xg.train.train_xg import train_xg_model
from src.monitoring.logger import log_xg_inference
from src.monitoring.drift import build_drift_report


router = APIRouter(prefix="/xg", tags=["xG (Expected Goals)"])


def _extract_body_part_options(feature_names: list[str]) -> list[str]:
    return sorted(
        {
            name.replace("body_part_", "", 1)
            for name in feature_names
            if name.startswith("body_part_")
        }
    )


@router.get(
    "/models",
    summary="List available models",
    description="Get list of all available xG models with metadata",
    response_model=XGModelListResponse,
)
async def list_models():
    models = discover_models()
    current_id = get_current_model_id()

    return {"models": models, "current_model": current_id, "count": len(models)}


@router.post(
    "/models/select",
    summary="Select active model by ID",
    description="Change the currently active model for predictions",
    response_model=SelectModelResponse,
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


@router.get(
    "/models/features",
    summary="List feature sets for all models",
    description="Get the feature vectors required by every registered xG model.",
    response_model=ModelFeatureListResponse,
)
async def list_model_feature_sets():
    models = discover_models()
    feature_sets = []
    errors = []

    for meta in models:
        model_id = meta["model_id"]
        try:
            model = load_model_artifact(model_id)
            feature_names = get_model_feature_names(model)
            feature_sets.append(
                {
                    "model_id": model_id,
                    "display_name": meta.get("display_name"),
                    "stage": meta.get("stage"),
                    "run_id": meta.get("run_id"),
                    "feature_count": len(feature_names),
                    "features": feature_names,
                    "body_part_options": _extract_body_part_options(feature_names),
                }
            )
        except Exception as exc:
            errors.append({"model_id": model_id, "error": str(exc)})

    response = {"models": feature_sets, "count": len(feature_sets)}
    if errors:
        response["errors"] = errors
    return response


@router.get(
    "/models/{model_id}/features",
    summary="Get feature set for a model",
    description="Return the exact feature vector required by a specific model ID (name@version).",
    response_model=ModelFeatureSet,
)
async def get_model_features(model_id: str):
    models = discover_models()
    metadata = next((m for m in models if m["model_id"] == model_id), None)
    if metadata is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_id}' not found in registry",
        )

    try:
        model = load_model_artifact(model_id)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unable to load model '{model_id}': {exc}",
        )

    feature_names = get_model_feature_names(model)
    return {
        "model_id": model_id,
        "display_name": metadata.get("display_name"),
        "stage": metadata.get("stage"),
        "run_id": metadata.get("run_id"),
        "feature_count": len(feature_names),
        "features": feature_names,
        "body_part_options": _extract_body_part_options(feature_names),
    }


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

        features, shot_distance, shot_angle_rad, shot_angle_deg = (
            build_features_from_request(model, shot)
        )

        if model is None or getattr(model, "predict_proba", None) is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Model does not support probability predictions",
            )

        xg_probability = float(model.predict_proba(features)[0, 1])

        # Monitoring log
        active_model_id = model_id or get_current_model_id() or "unknown"
        log_xg_inference(
            shot_distance=shot_distance,
            shot_angle=shot_angle_rad,
            xg=float(xg_probability),
            model_id=active_model_id,
        )

        response = ShotResponse(
            xG=round(float(xg_probability), 4),
            shot_distance=round(float(shot_distance), 2),
            shot_angle=round(float(shot_angle_deg), 2),
            quality=interpret_xg(xg_probability),
        )

        return response

    except Exception as e:
        import traceback

        traceback.print_exc()
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

        if model is None or getattr(model, "predict_proba", None) is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Model does not support probability predictions",
            )

        buf = generate_xg_heatmap(model, resolution)
        return StreamingResponse(buf, media_type="image/png")

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating heatmap: {str(e)}",
        )


@router.post(
    "/train",
    summary="Train a new xG model",
    description="Train a new xG model using the latest features dataset",
)
async def train_model(requestBody: XGModelTrainRequest):
    train_xg_model(
        features_path=requestBody.features_path,
        output_path=requestBody.output_path,
        test_size=requestBody.test_size,
        random_state=requestBody.random_state,
        max_iter=requestBody.max_iter,
        run_name=requestBody.run_name,
        experiment_name=requestBody.experiment_name,
        model_name=requestBody.model_name,
    )


@router.get(
    "/monitoring/drift",
    summary="View data drift report",
    description="Generate and return an HTML data drift report comparing training data to recent inferences.",
    response_class=HTMLResponse,
)
async def xg_drift_report():
    report_path = build_drift_report()
    if report_path is None:
        return HTMLResponse(
            "<h3>No monitoring data yet</h3><p>Make some /xg/score calls first.</p>",
            status_code=200,
        )
    with open(report_path, "r", encoding="utf-8") as f:
        html = f.read()
    return HTMLResponse(content=html, status_code=200)


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
