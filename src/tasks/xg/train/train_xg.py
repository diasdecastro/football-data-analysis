from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from src.common import io
from src.common.mlflow_utils import (
    log_dataset_info,
    log_model_coefficients,
    log_additional_metrics,
)

import mlflow
from mlflow import sklearn as mlflow_sklearn


def load_training_data(features_path: Path | None = None) -> pd.DataFrame:
    """
    Load the xG features dataset for training.
    """
    path = features_path or io.xg_features_gold_path()
    features = io.read_table(path)

    required_cols = ["shot_distance", "shot_angle", "body_part", "is_goal"]
    missing_cols = [col for col in required_cols if col not in features.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    print(
        f"📊 Loaded {len(features):,} shots (goal rate: {features['is_goal'].mean():.1%})"
    )

    return features


def prepare_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features and target for training.
    One-hot encodes body_part into binary features.
    """

    df_clean = df.dropna(subset=["shot_distance", "shot_angle", "body_part", "is_goal"])

    #  numeric features
    X = df_clean[["shot_distance", "shot_angle"]].copy()

    # encode body_part (categorical)
    body_part_dummies = pd.get_dummies(
        df_clean["body_part"], prefix="body_part", drop_first=False
    )

    # Combine numeric and categorical features
    X = pd.concat([X, body_part_dummies], axis=1)

    y = df_clean["is_goal"].copy()

    print(f"📈 Prepared {X.shape[0]:,} samples with {X.shape[1]} features")

    return X, y


def train_logistic_regression(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    max_iter: int = 1000,
) -> Tuple[LogisticRegression, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Train a logistic regression model for xG prediction.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = LogisticRegression(
        random_state=random_state,
        max_iter=max_iter,
        solver="liblinear",
    )

    model.fit(X_train, y_train)

    print(f"✅ Model trained ({len(X_train):,} train / {len(X_test):,} test)")

    return model, X_train, X_test, y_train, y_test


def evaluate_model(
    model: LogisticRegression,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict:
    """
    Evaluate the trained model on train and test sets.
    """

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "train": {
            "accuracy": accuracy_score(y_train, y_train_pred),
            "precision": precision_score(y_train, y_train_pred),
            "recall": recall_score(y_train, y_train_pred),
            "f1": f1_score(y_train, y_train_pred),
            "roc_auc": roc_auc_score(y_train, y_train_proba),
        },
        "test": {
            "accuracy": accuracy_score(y_test, y_test_pred),
            "precision": precision_score(y_test, y_test_pred),
            "recall": recall_score(y_test, y_test_pred),
            "f1": f1_score(y_test, y_test_pred),
            "roc_auc": roc_auc_score(y_test, y_test_proba),
        },
    }

    print(
        f"📊 Test Performance: ROC-AUC={metrics['test']['roc_auc']:.3f}, Accuracy={metrics['test']['accuracy']:.3f}"
    )

    return metrics


def save_model(model: LogisticRegression, output_path: Path | None = None) -> Path:
    """
    Save the trained model to disk.
    """
    if output_path is None:
        output_path = Path("models/xg_model.joblib")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, output_path)

    return output_path


def train_xg_model(
    features_path: Path | None = None,
    output_path: Path | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    max_iter: int = 1000,
) -> Tuple[LogisticRegression, dict, Path]:
    """
    Train xG model and log the run to MLflow.
    """
    print("🚀 Starting xG Training Pipeline")

    mlflow.set_experiment("xG Training")

    run_name_parts = ["xg_logreg"]
    if features_path is not None:
        run_name_parts.append(features_path.stem)
    run_name = "_".join(run_name_parts)

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("task", "xg")
        mlflow.set_tag("model_family", "logistic_regression")

        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("max_iter", max_iter)
        if features_path is not None:
            mlflow.log_param("features_path", str(features_path))
        else:
            mlflow.log_param("features_path", str(io.xg_features_gold_path()))

        df = load_training_data(features_path)
        X, y = prepare_features_target(df)

        log_dataset_info(df, y)

        model, X_train, X_test, y_train, y_test = train_logistic_regression(
            X, y, test_size=test_size, random_state=random_state, max_iter=max_iter
        )

        log_dataset_info(df, y, X_train, X_test, y_train, y_test)

        log_model_coefficients(model)

        metrics = evaluate_model(model, X_train, X_test, y_train, y_test)

        for split_name, split_metrics in metrics.items():
            for metric_name, value in split_metrics.items():
                mlflow.log_metric(f"{split_name}_{metric_name}", float(value))

        y_test_proba = model.predict_proba(X_test)[:, 1]
        log_additional_metrics(y_test, y_test_proba)

        saved_path = save_model(model, output_path)

        mlflow.log_artifact(str(saved_path), artifact_path="artifacts")

        input_example = X_test.head(1)

        REGISTERED_MODEL_NAME = "xG Bundesliga"

        mlflow_sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example,
            registered_model_name=REGISTERED_MODEL_NAME,
        )

        print(f"🎉 Training complete! Model saved to {saved_path}")

        return model, metrics, saved_path


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a logistic regression model for xG prediction (with MLflow logging)"
    )
    parser.add_argument(
        "--features-path",
        type=str,
        default=None,
        help="Path to features file (default: data/gold/xg_features.parquet)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Path to save trained model (default: models/xg_model.joblib)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of data for testing (default: 0.2)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="Maximum iterations for solver (default: 1000)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_cli()

    model, metrics, saved_path = train_xg_model(
        features_path=Path(args.features_path) if args.features_path else None,
        output_path=Path(args.output_path) if args.output_path else None,
        test_size=args.test_size,
        random_state=args.random_state,
        max_iter=args.max_iter,
    )
