"""
Simple, reusable MLflow logging utilities.
"""

import mlflow
from pathlib import Path
from sklearn.metrics import brier_score_loss, log_loss


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
