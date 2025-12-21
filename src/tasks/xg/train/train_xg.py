from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import StandardScaler

from src.common import io
from src.common.mlflow_utils import (
    log_dataset_info,
    log_model_coefficients,
    log_params,
)

import mlflow
from mlflow import sklearn as mlflow_sklearn
from src.tasks.xg.features.pipeline import build_feature_pipeline


def load_training_data(features_path: Path | None = None) -> pd.DataFrame:
    """
    Load the xG features dataset for training.
    """
    path = features_path or io.xg_features_gold_path()
    features = io.read_table(path)

    required_cols = [
        "shot_distance",
        "shot_angle",
        "end_x",
        "end_y",
        "body_part",
        "is_open_play",
        "one_on_one",
        "is_goal",
    ]
    missing_cols = [col for col in required_cols if col not in features.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    print(
        f"Loaded {len(features):,} shots (goal rate: {features['is_goal'].mean():.1%})"
    )

    return features


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features for training using the feature pipeline.
    """

    included_features = [
        "shot_distance",
        "shot_angle",
        "end_x",
        "end_y",
        "body_part_right_foot",
        "body_part_left_foot",
        "body_part_head",
        "body_part_other",
        "is_open_play",
        "one_on_one",
    ]

    pipeline = build_feature_pipeline()
    X = pipeline.transform(df)[included_features].dropna()
    y = df["is_goal"].dropna()

    # Feature engineering

    X["log_angle"] = np.log(X["shot_angle"] + 1e-5)
    X["one_on_one_x_log_angle"] = X["one_on_one"].astype(int) * X["log_angle"]
    X["one_on_one_x_dist"] = X["one_on_one"].astype(int) * X["shot_distance"]
    X["head_x_dist"] = X["body_part_head"].astype(int) * X["shot_distance"]
    X["distance_x_angle"] = X["shot_distance"] * X["log_angle"]

    # Feature scaling
    scaler = StandardScaler()
    X_continous = X[
        [
            "end_x",
            "end_y",
            "shot_distance",
            "shot_angle",
            "log_angle",
            "one_on_one_x_log_angle",
            "one_on_one_x_dist",
            "head_x_dist",
            "distance_x_angle",
        ]
    ]
    X_scaled = scaler.fit_transform(X_continous)

    X.drop(
        [
            "end_x",
            "end_y",
            "shot_distance",
            "shot_angle",
            "log_angle",
            "one_on_one_x_log_angle",
            "one_on_one_x_dist",
            "head_x_dist",
            "distance_x_angle",
        ],
        axis=1,
        inplace=True,
    )
    X_scaled_df = pd.DataFrame(
        X_scaled,
        columns=[
            "end_x",
            "end_y",
            "shot_distance",
            "shot_angle",
            "log_angle",
            "one_on_one_x_log_angle",
            "one_on_one_x_dist",
            "head_x_dist",
            "distance_x_angle",
        ],
        index=X.index,
    )
    X = pd.concat([X, X_scaled_df], axis=1)

    return X, y


def train_model(
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
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    model = LogisticRegression(
        random_state=random_state,
        max_iter=max_iter,
        solver="lbfgs",
        penalty="l2",
        C=0.3,
    )

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cross_val_score(model, X_train, y_train, cv=cv, scoring="neg_log_loss")

    model.fit(X_train, y_train)

    print(f"✅ Model trained ({len(X_train):,} train / {len(X_test):,} test)")

    return model, X_train, X_test, y_train, y_test


def evaluate_model(
    model,
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

    prob_true_train, prob_pred_train = calibration_curve(
        y_train, y_train_proba, n_bins=5
    )
    prob_true_test, prob_pred_test = calibration_curve(y_test, y_test_proba, n_bins=5)

    # NOTE: Accuracy is not very informative for imbalanced data (might help detect overfitting)
    # NOTE: Confusion Matrix not important, we care about probabilities, not classifications
    metrics = {
        "train": {
            "log_loss": log_loss(y_train, y_train_proba),
            "brier_score": brier_score_loss(y_train, y_train_proba),
            "roc_auc": roc_auc_score(y_train, y_train_proba),
            "calibration_curve": (prob_true_train, prob_pred_train),
            "accuracy": accuracy_score(y_train, y_train_pred),
            "precision": precision_score(y_train, y_train_pred),
            "recall": recall_score(y_train, y_train_pred),
            "f1": f1_score(y_train, y_train_pred),
        },
        "test": {
            "log_loss": log_loss(y_test, y_test_proba),
            "brier_score": brier_score_loss(y_test, y_test_proba),
            "roc_auc": roc_auc_score(y_test, y_test_proba),
            "calibration_curve": (prob_true_test, prob_pred_test),
            "accuracy": accuracy_score(y_test, y_test_pred),
            "precision": precision_score(y_test, y_test_pred),
            "recall": recall_score(y_test, y_test_pred),
            "f1": f1_score(y_test, y_test_pred),
        },
    }

    print(
        f"Test Performance: log_loss={metrics['test']['log_loss']:.3f}, brier_score={metrics['test']['brier_score']:.3f}"
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


def train_pipeline(
    features_path: Path | None = None,
    output_path: Path | None = None,
    test_size: float = 0.3,
    random_state: int = 42,
    max_iter: int = 1000,
    run_name: str | None = None,
    experiment_name: str = "Default",
    model_name: str = "xG Model",
) -> Tuple[LogisticRegression, dict, Path]:
    """
    Train xG model and log the run to MLflow.
    """
    print("🚀 Starting xG Training Pipeline")

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):

        df = load_training_data(features_path)
        X, y = prepare_features(df)

        amount_features = X.shape[1]
        feature_names = X.columns.tolist()
        feature_list = ", ".join(feature_names)

        log_params(
            task="xg",
            model_name=model_name,
            model_family="linear",
            framework="sklearn",
            test_size=test_size,
            random_state=random_state,
            max_iter=max_iter,
            features_path=features_path,
            amount_features=amount_features,
            feature_list=feature_list,
        )

        log_dataset_info(df, y)

        model, X_train, X_test, y_train, y_test = train_model(
            X, y, test_size=test_size, random_state=random_state, max_iter=max_iter
        )

        log_dataset_info(df, y, X_train, X_test, y_train, y_test)
        log_model_coefficients(model, feature_names)

        metrics = evaluate_model(model, X_train, X_test, y_train, y_test)

        # Log metrics to MLflow
        for split_name, split_metrics in metrics.items():
            for metric_name, value in split_metrics.items():
                if metric_name != "calibration_curve":
                    mlflow.log_metric(f"{split_name}_{metric_name}", float(value))
                else:
                    prob_true, prob_pred = value
                    mlflow.log_param(
                        f"{split_name}_calibration_prob_true", str(prob_true.tolist())
                    )
                    mlflow.log_param(
                        f"{split_name}_calibration_prob_pred", str(prob_pred.tolist())
                    )

        y_test_proba = model.predict_proba(X_test)[:, 1]

        saved_path = save_model(model, output_path)

        mlflow.log_artifact(str(saved_path), artifact_path="artifacts")

        input_example = X_test.head(1)

        mlflow_sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example,
            registered_model_name=model_name,
        )

        print(f"Training complete! Model saved to {saved_path}")

        return model, metrics, saved_path


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a logistic regression model for xG prediction (with MLflow logging)"
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="MLflow run name (e.g., 'v3', 'v4'). If not provided, auto-generated.",
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
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="Default",
        help="MLflow experiment name (default: Default)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="xG Model",
        help="Registered model name in MLflow (default: xG Model)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_cli()

    model, metrics, saved_path = train_pipeline(
        features_path=Path(args.features_path) if args.features_path else None,
        output_path=Path(args.output_path) if args.output_path else None,
        test_size=args.test_size,
        random_state=args.random_state,
        max_iter=args.max_iter,
        run_name=args.run_name,
        experiment_name=args.experiment_name,
        model_name=args.model_name,
    )
