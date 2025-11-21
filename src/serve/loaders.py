"""
Model loader for the FastAPI serving layer.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional, Dict, List

from mlflow import sklearn as mlflow_sklearn
from mlflow.tracking import MlflowClient

import os

DEFAULT_MODEL_NAME = os.getenv("XG_MODEL_NAME", "xG Bundesliga")
DEFAULT_MODEL_STAGE = os.getenv("XG_MODEL_STAGE", None)
_CURRENT_MODEL_ID: Optional[str] = None

_client = MlflowClient()


def _parse_model_id(model_id: str) -> tuple[str, str]:
    """
    Split 'name@version_or_stage' into (name, version_or_stage).
    Example: 'xg_bundesliga@3' -> ('xg_bundesliga', '3')
             'xg_bundesliga@Production' -> ('xg_bundesliga', 'Production')
    """
    name, sep, rest = model_id.partition("@")
    if not sep or not name or not rest:
        raise ValueError(
            f"Invalid model_id: {model_id!r}, expected format 'name@version_or_stage'"
        )
    return name, rest


def discover_models(model_name: str = DEFAULT_MODEL_NAME) -> List[Dict]:
    """
    Discover all versions of a model in MLflow by name.
    """
    versions = _client.search_model_versions(f"name = '{model_name}'")
    models: List[Dict] = []

    for v in versions:
        run = _client.get_run(getattr(v, "run_id"))
        metrics = dict(run.data.metrics)
        tags = dict(run.data.tags)

        model_key = f"{v.name}@{v.version}"
        display_name = f"{v.name} v{v.version}"

        models.append(
            {
                "model_id": model_key,
                "model_key": model_key,
                "name": v.name,
                "display_name": display_name,
                "version": int(v.version),
                "stage": v.current_stage,
                "run_id": getattr(v, "run_id", "N/A"),
                "created_at": v.creation_timestamp,
                "metrics": metrics,
                "tags": tags,
            }
        )

    models.sort(key=lambda m: m["version"], reverse=True)
    return models


@lru_cache
def _load_model_from_registry(model_id: str):
    """
    Internal cached loader. Returns the underlying sklearn model.
    """
    name, version_or_stage = _parse_model_id(model_id)
    uri = f"models:/{name}/{version_or_stage}"
    print(f"📦 Loading xG model from MLflow: {uri}")
    model = mlflow_sklearn.load_model(uri)
    print(f"✅ Loaded model: {model_id}")
    return model


def get_xg_model(model_id: Optional[str] = None):
    """
    Get an xG model.
    """
    global _CURRENT_MODEL_ID

    if model_id is not None:
        _CURRENT_MODEL_ID = model_id
        return _load_model_from_registry(model_id)

    if _CURRENT_MODEL_ID is not None:
        return _load_model_from_registry(_CURRENT_MODEL_ID)

    # fallback: if no stage set, use latest version
    if DEFAULT_MODEL_STAGE is None:
        models = discover_models(DEFAULT_MODEL_NAME)
        if models:
            latest_model = models[0]
            default_id = latest_model["model_id"]
        else:
            raise FileNotFoundError(
                f"No versions found for model '{DEFAULT_MODEL_NAME}'"
            )
    else:
        default_id = f"{DEFAULT_MODEL_NAME}@{DEFAULT_MODEL_STAGE}"

    _CURRENT_MODEL_ID = default_id
    return _load_model_from_registry(default_id)


def set_current_model(model_id: str):
    global _CURRENT_MODEL_ID

    _load_model_from_registry(model_id)

    _CURRENT_MODEL_ID = model_id
    print(f"✅ Current model set to: {model_id}")


def get_current_model_id() -> Optional[str]:
    return _CURRENT_MODEL_ID
