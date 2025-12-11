"""
Simple, reusable MLflow logging utilities.
"""

from typing import Optional
import mlflow
from pathlib import Path

from src.common import io


# For now xG-specific, can be generalized later
def log_dataset_info(df, y, X_train=None, X_test=None, y_train=None, y_test=None):
    """
    Log dataset statistics to MLflow.
    """

    mlflow.log_param("n_samples_total", len(df))
    mlflow.log_param("n_goals_total", int(y.sum()))
    mlflow.log_param("goal_rate", float(y.mean()))

    if X_train is not None and y_train is not None:
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_goals_train", int(y_train.sum()))
        mlflow.log_param("train_goal_rate", float(y_train.mean()))

    if X_test is not None and y_test is not None:
        mlflow.log_param("n_test", len(X_test))
        mlflow.log_param("n_goals_test", int(y_test.sum()))
        mlflow.log_param("test_goal_rate", float(y_test.mean()))


# For now only scikit-learn models
# Assumes linearity in pararameters
def log_model_coefficients(model, feature_names=None):
    """
    Log model coefficients/feature importances.

    """
    if hasattr(model, "coef_"):
        coefs = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_

        for i, coef in enumerate(coefs):
            if feature_names and i < len(feature_names):
                mlflow.log_metric(f"coef_{feature_names[i]}", float(coef))
            else:
                mlflow.log_metric(f"coef_{i}", float(coef))

    # Linear model intercept
    if hasattr(model, "intercept_"):
        intercept = (
            model.intercept_[0]
            if hasattr(model.intercept_, "__iter__")
            else model.intercept_
        )
        mlflow.log_metric("intercept", float(intercept))

    # Tree-based models
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_

        for i, importance in enumerate(importances):
            if feature_names and i < len(feature_names):
                mlflow.log_metric(f"importance_{feature_names[i]}", float(importance))
            else:
                mlflow.log_metric(f"importance_{i}", float(importance))


# TODO: make more general
def log_params(
    task: str = "Unknown Task",
    model_name: str = "NA",
    model_family: str = "NA",
    framework: str = "NA",
    test_size: float = 0.0,
    random_state: int = 0,
    max_iter: int = 0,
    solver: str = "NA",
    features_path: Optional[Path] = Path("NA"),
    amount_features: int = 0,
    feature_list: str = "NA",
):
    """
    Helper for logging MLflow parameters
    """
    mlflow.set_tag("task", task)
    mlflow.set_tag("model_type", model_name)
    mlflow.set_tag("model_family", model_family)
    mlflow.set_tag("framework", framework)

    mlflow.log_param("test_size", test_size)
    mlflow.log_param("random_state", random_state)
    mlflow.log_param("max_iter", max_iter)
    mlflow.log_param("solver", solver)

    if features_path is not None:
        mlflow.log_param("features_path", str(features_path))
    else:
        mlflow.log_param("features_path", str(io.xg_features_gold_path()))

    mlflow.log_param("amount_features", amount_features)
    mlflow.log_param("feature_names", feature_list)
    mlflow.set_tag("features", feature_list)  # Also as tag for easy filtering
