# src/tasks/xg/train/train_xg.py
from __future__ import annotations

import argparse
import joblib
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

from src.common import io


def load_training_data(features_path: Path | None = None) -> pd.DataFrame:
    """
    Load the xG features dataset for training.

    Args:
        features_path: Optional custom path to features file

    Returns:
        DataFrame with features and target
    """
    path = features_path or io.xg_features_gold_path()
    features = io.read_table(path)

    # Validate required columns
    required_cols = ["shot_distance", "shot_angle", "is_goal"]
    missing_cols = [col for col in required_cols if col not in features.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    print(f"📊 Loaded {len(features):,} shots from {path}")
    print(f"   Goal rate: {features['is_goal'].mean():.1%}")

    return features


def prepare_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features and target for training.

    Args:
        df: Raw features DataFrame

    Returns:
        Tuple of (features, target)
    """
    # Remove any rows with missing values
    df_clean = df.dropna(subset=["shot_distance", "shot_angle", "is_goal"])

    # Features: shot_distance and shot_angle
    X = df_clean[["shot_distance", "shot_angle"]].copy()

    # Target: is_goal (binary)
    y = df_clean["is_goal"].copy()

    print(f"📈 Features shape: {X.shape}")
    print(
        f"   Distance range: {X['shot_distance'].min():.1f} - {X['shot_distance'].max():.1f}"
    )
    print(
        f"   Angle range: {X['shot_angle'].min():.1f}° - {X['shot_angle'].max():.1f}°"
    )

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

    Args:
        X: Features DataFrame
        y: Target Series
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        max_iter: Maximum iterations for solver convergence

    Returns:
        Tuple of (model, X_train, X_test, y_train, y_test)
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"🔄 Training set: {len(X_train):,} samples")
    print(f"   Test set: {len(X_test):,} samples")

    # Train logistic regression
    model = LogisticRegression(
        random_state=random_state,
        max_iter=max_iter,
        solver="liblinear",  # Good for small datasets
    )

    model.fit(X_train, y_train)

    print("✅ Model trained successfully")
    print(
        f"   Coefficients: distance={model.coef_[0][0]:.4f}, angle={model.coef_[0][1]:.4f}"
    )
    print(f"   Intercept: {model.intercept_[0]:.4f}")

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

    Args:
        model: Trained logistic regression model
        X_train: Training features
        X_test: Test features
        y_train: Training target
        y_test: Test target

    Returns:
        Dictionary with evaluation metrics
    """
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Probabilities for ROC-AUC
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
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

    # Print results
    print("\n📊 Model Performance:")
    print("=" * 50)
    print(f"{'Metric':<12} {'Train':<10} {'Test':<10}")
    print("-" * 32)
    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        train_val = metrics["train"][metric]
        test_val = metrics["test"][metric]
        print(f"{metric.capitalize():<12} {train_val:<10.3f} {test_val:<10.3f}")

    print("\n📋 Detailed Test Set Report:")
    print(classification_report(y_test, y_test_pred))

    print("🎯 Confusion Matrix (Test Set):")
    cm = confusion_matrix(y_test, y_test_pred)
    print(f"   [[TN={cm[0,0]}, FP={cm[0,1]}],")
    print(f"    [FN={cm[1,0]}, TP={cm[1,1]}]]")

    return metrics


def save_model(model: LogisticRegression, output_path: Path | None = None) -> Path:
    """
    Save the trained model to disk.

    Args:
        model: Trained logistic regression model
        output_path: Optional custom output path

    Returns:
        Path where model was saved
    """
    if output_path is None:
        output_path = io.gold("xg_model.joblib")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, output_path)
    print(f"💾 Model saved to: {output_path}")

    return output_path


def train_xg_model(
    features_path: Path | None = None,
    output_path: Path | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    max_iter: int = 1000,
) -> Tuple[LogisticRegression, dict, Path]:
    """
    Complete training pipeline for xG model.

    Args:
        features_path: Path to features file
        output_path: Path to save trained model
        test_size: Test set proportion
        random_state: Random seed
        max_iter: Max iterations for training

    Returns:
        Tuple of (model, metrics, saved_path)
    """
    print("🚀 Starting xG Model Training Pipeline")
    print("=" * 50)

    # Load data
    df = load_training_data(features_path)

    # Prepare features and target
    X, y = prepare_features_target(df)

    # Train model
    model, X_train, X_test, y_train, y_test = train_logistic_regression(
        X, y, test_size=test_size, random_state=random_state, max_iter=max_iter
    )

    # Evaluate model
    metrics = evaluate_model(model, X_train, X_test, y_train, y_test)

    # Save model
    saved_path = save_model(model, output_path)

    print("\n🎉 Training pipeline completed successfully!")
    return model, metrics, saved_path


def parse_cli() -> argparse.Namespace:
    """Parse command line arguments for xG model training."""
    parser = argparse.ArgumentParser(
        description="Train a logistic regression model for xG prediction"
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
        help="Path to save trained model (default: data/gold/xg_model.joblib)",
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
