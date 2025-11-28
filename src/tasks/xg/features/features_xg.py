# src/tasks/xg/features/features_xg.py
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from src.common import io, validation


# Defaults aligned with our silver schema
FEATURE_COLS = [
    "match_id",
    "team_id",
    "player_id",
    "shot_distance",
    "shot_angle",
    "body_part",
    "is_open_play",
    "one_on_one",
    "is_goal",
]


def build_xg_features(
    in_path: Path | None = None,
    out_path: Path | None = None,
    exclude_penalties: bool = True,
) -> pd.DataFrame:
    """
    Build the gold/xg_features.parquet table for training/inference.
    """
    shots = io.read_table(in_path or io.shots_silver_path())

    validation.require_columns(
        shots,
        [
            "match_id",
            "team_id",
            "player_id",
            "distance_to_goal",
            "shot_angle",
            "body_part",
            "is_open_play",
            "one_on_one",
            "is_goal",
        ],
    )

    if exclude_penalties and "is_penalty" in shots.columns:
        shots = shots[shots["is_penalty"] == 0].copy()

    features = shots.rename(columns={"distance_to_goal": "shot_distance"})[
        [
            "match_id",
            "team_id",
            "player_id",
            "shot_distance",
            "shot_angle",
            "body_part",
            "is_open_play",
            "one_on_one",
            "is_goal",
        ]
    ].reset_index(drop=True)

    target = out_path or io.xg_features_gold_path()
    io.write_table(features, target, index=False)
    return features


def parse_cli() -> argparse.Namespace:
    """Parse command line arguments for xG features building."""
    ap = argparse.ArgumentParser(
        description="Build gold/xg_features.parquet from silver/shots.parquet"
    )
    ap.add_argument(
        "--in",
        dest="inp",
        type=str,
        default=None,
        help="Optional custom input path (silver shots)",
    )
    ap.add_argument(
        "--out",
        dest="out",
        type=str,
        default=None,
        help="Optional custom output path",
    )
    ap.add_argument(
        "--include-penalties",
        action="store_true",
        help="Keep penalties (default: excluded)",
    )
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_cli()

    features = build_xg_features(
        in_path=Path(args.inp) if args.inp else None,
        out_path=Path(args.out) if args.out else None,
        exclude_penalties=not args.include_penalties,
    )
    print(
        f"✅ Built features: {len(features):,} rows → {args.out or io.xg_features_gold_path()}"
    )
