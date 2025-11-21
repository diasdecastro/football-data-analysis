"""
src/common/mlflow_utils.py

Simple, reusable MLflow logging utilities.
"""

import mlflow
from pathlib import Path
from sklearn.metrics import brier_score_loss, log_loss


def log_dataset_info(df, y, X_train=None, X_test=None, y_train=None, y_test=None):
    """
    Log dataset statistics to MLflow.
    """
    mlflow.log_param("n_samples", len(df))
    mlflow.log_param("goal_rate", float(y.mean()))

    if X_train is not None and y_train is not None:
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("train_goal_rate", float(y_train.mean()))

    if X_test is not None and y_test is not None:
        mlflow.log_param("n_test", len(X_test))
        mlflow.log_param("test_goal_rate", float(y_test.mean()))


def log_model_coefficients(model):
    """
    Log model coefficients for linear models.
    """
    if hasattr(model, "coef_"):
        for i, coef in enumerate(model.coef_[0]):
            mlflow.log_param(f"coef_{i}", float(coef))

    if hasattr(model, "intercept_"):
        mlflow.log_param("intercept", float(model.intercept_[0]))


def log_additional_metrics(y_test, y_test_proba):
    """
    Log additional metrics beyond basic classification.
    """
    mlflow.log_metric("test_brier_score", float(brier_score_loss(y_test, y_test_proba)))
    mlflow.log_metric("test_log_loss", float(log_loss(y_test, y_test_proba)))
