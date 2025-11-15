from pathlib import Path
import joblib
from typing import Optional


_XG_MODEL = None


def load_xg_model(model_path: Optional[Path] = None):
    if model_path is None:
        model_path = Path("models/xg_model.joblib")

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"xG model not found at {model_path}.")

    print(f"Loading xG model from {model_path}")
    model = joblib.load(model_path)
    print(f"xG model loaded successfully")

    return model


def get_xg_model():
    global _XG_MODEL

    if _XG_MODEL is None:
        _XG_MODEL = load_xg_model()

    return _XG_MODEL


def reload_xg_model(model_path: Optional[Path] = None):
    """
    Force reload of the xG model.
    Useful for updating to a newly trained model without restarting the server.
    """
    global _XG_MODEL
    _XG_MODEL = load_xg_model(model_path)
    return _XG_MODEL
