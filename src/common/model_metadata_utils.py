"""
save_model_metadata.py

Utility to save metadata alongside trained models.
Run this after training to create metadata files for the demo.
"""

import argparse
import json
from pathlib import Path
from datetime import datetime


def save_metadata(
    model_id: str,
    name: str,
    description: str,
    model_type: str = "LogisticRegression",
    metrics: dict = None,
    features: list = None,
):
    """
    Save model metadata to JSON file.
    """
    if metrics is None:
        metrics = {}

    if features is None:
        features = ["shot_distance", "shot_angle"]

    metadata = {
        "model_id": model_id,
        "name": name,
        "description": description,
        "created_at": datetime.now().isoformat(),
        "model_type": model_type,
        "features": features,
        "metrics": metrics,
    }

    output_path = Path("models") / f"{model_id}_metadata.json"

    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Metadata saved to: {output_path}")
    print(f"📊 Model: {name}")
    print(f"📝 Description: {description}")
    if metrics:
        print(f"📈 Metrics: {metrics}")


def main():
    parser = argparse.ArgumentParser(description="Save metadata for a trained model")

    parser.add_argument(
        "--model-id",
        required=True,
        help="Model identifier (e.g., 'xg_model', 'xg_model_v2')",
    )
    parser.add_argument("--name", required=True, help="Display name for the model")
    parser.add_argument(
        "--description", required=True, help="Detailed description of the model"
    )
    parser.add_argument(
        "--model-type",
        default="LogisticRegression",
        help="Type of model (default: LogisticRegression)",
    )
    parser.add_argument("--roc-auc", type=float, help="ROC AUC score")
    parser.add_argument("--brier-score", type=float, help="Brier score")
    parser.add_argument("--accuracy", type=float, help="Accuracy score")

    args = parser.parse_args()

    metrics = {}
    if args.roc_auc is not None:
        metrics["roc_auc"] = args.roc_auc
    if args.brier_score is not None:
        metrics["brier_score"] = args.brier_score
    if args.accuracy is not None:
        metrics["accuracy"] = args.accuracy

    save_metadata(
        model_id=args.model_id,
        name=args.name,
        description=args.description,
        model_type=args.model_type,
        metrics=metrics,
    )


if __name__ == "__main__":
    main()
