"""
src/serve/loaders.py

Model loader for the FastAPI serving layer.
"""

from pathlib import Path
import joblib
import json
from typing import Optional, Dict, List
from datetime import datetime

from src.common import io


# Global model cache {model_id: model_object}
_MODELS_CACHE = {}
_CURRENT_MODEL_ID = None


def discover_models() -> List[Dict]:
    """
    Discover all available models in the models directory.
    """
    models_dir = Path("models")
    if not models_dir.exists():
        return []

    models = []

    # Find all .joblib files
    for model_file in sorted(models_dir.glob("*.joblib")):
        model_id = model_file.stem  # e.g., "xg_model" or "xg_model_v2"

        # Look for corresponding metadata file
        metadata_file = models_dir / f"{model_id}_metadata.json"

        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
        else:
            # Generate basic metadata if none exists
            stat = model_file.stat()
            metadata = {
                "model_id": model_id,
                "name": model_id.replace("_", " ").title(),
                "description": "No description available",
                "created_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "model_type": "Unknown",
                "metrics": {},
            }

        metadata["model_id"] = model_id
        metadata["path"] = str(model_file)
        models.append(metadata)

    return models


def load_xg_model(model_id: str = "xg_model") -> object:
    """
    Load a specific xG model by ID.
    """
    model_path = Path(f"models/{model_id}.joblib")

    if not model_path.exists():
        raise FileNotFoundError(
            f"xG model '{model_id}' not found at {model_path}. "
            f"Available models: {[m['model_id'] for m in discover_models()]}"
        )

    print(f"📦 Loading xG model '{model_id}' from {model_path}")
    model = joblib.load(model_path)
    print(f"✅ Model '{model_id}' loaded successfully")

    return model


def get_xg_model(model_id: Optional[str] = None) -> object:
    """
    Get an xG model by ID (with caching).
    """
    global _MODELS_CACHE, _CURRENT_MODEL_ID

    if model_id is None:
        if _CURRENT_MODEL_ID is None:
            available = discover_models()
            if not available:
                raise FileNotFoundError("No models found in models/ directory")
            model_id = available[0]["model_id"]
            _CURRENT_MODEL_ID = model_id
        else:
            model_id = _CURRENT_MODEL_ID

    if model_id not in _MODELS_CACHE:
        if model_id is None:
            raise ValueError("model_id cannot be None when loading model")
        _MODELS_CACHE[model_id] = load_xg_model(model_id)

    return _MODELS_CACHE[model_id]


def set_current_model(model_id: str):
    """
    Set the current active model.
    """
    global _CURRENT_MODEL_ID

    # Verify model exists
    model = get_xg_model(model_id)
    _CURRENT_MODEL_ID = model_id

    print(f"✅ Current model set to: {model_id}")


def get_current_model_id() -> Optional[str]:
    """Get the current active model ID."""
    return _CURRENT_MODEL_ID


def clear_cache():
    """Clear the model cache (useful for testing/reloading)."""
    global _MODELS_CACHE, _CURRENT_MODEL_ID
    _MODELS_CACHE.clear()
    _CURRENT_MODEL_ID = None
    print("🧹 Model cache cleared")
