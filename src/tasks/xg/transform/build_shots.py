from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

import pandas as pd

from src.common import geometry, io, lookup, validation

BRONZE_ROOT = io.bronze("statsbomb_open_data/data")


@dataclass(frozen=True)
class Shot:
    event_id: str
    match_id: int
    competition_id: int
    season_id: int
    home_team_id: int
    away_team_id: int
    team_id: int
    opponent_team_id: int
    player_id: Optional[int]
    period: int
    minute: int
    second: int
    x: Optional[float]
    y: Optional[float]
    end_x: Optional[float]
    end_y: Optional[float]
    distance_to_goal: Optional[float]
    shot_angle: Optional[float]
    is_goal: int
    is_penalty: int
    is_freekick: int
    is_open_play: int
    body_part: Optional[str]
    technique: Optional[str]
    first_time: int
    one_on_one: int


def _read_json(path: Path) -> list | dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


"""
Important competition IDs:
9 = German Bundesliga
11 = Spanish La Liga
43 = English Premier League
12 = Italian Serie A
9 = UEFA Champions League
2 = FIFA World Cup
"""


def _iterate_matches_rows(
    comps: Optional[set[int]] = None,
    seasons: Optional[set[int]] = None,
) -> Iterator[Dict]:
    """
    Yields match rows from the StatsBomb matches dataset.
    Args: (
        comps: set of competition_ids to filter
        seasons: set of season_ids to filter
    )
    Yields: {
        "match_id": int,
        "competition_id": int,
        "season_id": int,
        "home_team_id": int,
        "away_team_id": int,
    }
    """
    matches_dir = BRONZE_ROOT / "matches"
    if not matches_dir.exists():
        raise FileNotFoundError(
            f"Cannot find StatsBomb matches folder at {matches_dir}. "
            "Make sure you've added the submodule or cloned the repo under data/bronze/statsbomb_open_data."
        )

    for comp_dir in sorted(matches_dir.iterdir()):
        if not comp_dir.is_dir():
            continue
        try:
            comp_id = int(comp_dir.name)
        except ValueError:
            continue
        if comps and comp_id not in comps:
            continue

        for season_file in sorted(comp_dir.glob("*.json")):
            try:
                season_id = int(season_file.stem)
            except ValueError:
                continue
            if seasons and season_id not in seasons:
                continue

            records = _read_json(season_file)
            for m in records:
                yield {
                    "match_id": int(m["match_id"]),
                    "competition_id": comp_id,
                    "season_id": season_id,
                    "home_team_id": int(m["home_team"]["home_team_id"]),
                    "away_team_id": int(m["away_team"]["away_team_id"]),
                }


def _iterate_shots_for_match(match_row: Dict) -> Iterator[Shot]:
    """
    Yields Shot objects for all shot events in the given match.
    Args: match_row: object with match data
    Yields: Shot objects
    """
    match_id = match_row["match_id"]
    events_path = BRONZE_ROOT / "events" / f"{match_id}.json"
    if not events_path.exists():
        # Some seasons in early years may not have events (rare). Skip safely.
        return

    try:
        events = _read_json(events_path)
    except Exception as e:
        raise RuntimeError(f"Failed to parse {events_path}: {e}") from e

    for ev in events:
        # Only keep Shot events
        if ev.get("type", {}).get("name") != "Shot":
            continue

        shot = ev.get("shot", {}) or {}
        team_id = int(ev["team"]["id"])
        home_team_id = match_row["home_team_id"]
        away_team_id = match_row["away_team_id"]
        opponent_team_id = away_team_id if team_id == home_team_id else home_team_id

        # Locations can be missing in some edge cases
        loc = ev.get("location") or [None, None]
        end_loc = shot.get("end_location") or [None, None]

        def _to_float(v):
            try:
                return float(v) if v is not None else None
            except Exception:
                return None

        x_val = _to_float(loc[0]) if len(loc) > 0 else None
        y_val = _to_float(loc[1]) if len(loc) > 1 else None

        # Calculate distance to goal and shot angle
        distance_to_goal = None
        shot_angle = None
        if x_val is not None and y_val is not None:
            distance_to_goal = geometry.distance_to_goal(x_val, y_val)
            shot_angle = geometry.shot_angle(x_val, y_val)

        yield Shot(
            event_id=str(ev["id"]),
            match_id=match_id,
            competition_id=match_row["competition_id"],
            season_id=match_row["season_id"],
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            team_id=team_id,
            opponent_team_id=opponent_team_id,
            player_id=int(ev.get("player", {}).get("id")) if ev.get("player") else None,
            period=int(ev.get("period", 0)),
            minute=int(ev.get("minute", 0)),
            second=int(ev.get("second", 0)),
            x=x_val,
            y=y_val,
            end_x=_to_float(end_loc[0]) if len(end_loc) > 0 else None,
            end_y=_to_float(end_loc[1]) if len(end_loc) > 1 else None,
            distance_to_goal=distance_to_goal,
            shot_angle=shot_angle,
            is_goal=1 if (shot.get("outcome", {}) or {}).get("name") == "Goal" else 0,
            is_penalty=(
                1 if (shot.get("type", {}) or {}).get("name") == "Penalty" else 0
            ),
            is_freekick=(
                1 if (shot.get("type", {}) or {}).get("name") == "Free Kick" else 0
            ),
            is_open_play=(
                1 if (shot.get("type", {}) or {}).get("name") == "Open Play" else 0
            ),
            body_part=(shot.get("body_part") or {}).get("name"),
            technique=(shot.get("technique") or {}).get("name"),
            first_time=int(bool(shot.get("first_time"))),
            one_on_one=int(bool(shot.get("one_on_one"))),
        )


def build_shots(
    comps: Optional[Iterable[int]] = None,
    seasons: Optional[Iterable[int]] = None,
    out_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Build the silver shots table from StatsBomb Open Data.

    Args:
        comps: list/set of competition_ids to include (None = all)
        seasons: list/set of season_ids to include (None = all)
        out_path: override output path (defaults to data/silver/shots.parquet)

    Returns:
        DataFrame with one row per shot.
    """
    comps_set = set(comps) if comps else None
    seasons_set = set(seasons) if seasons else None

    # 1) Enumerate matches
    matches_rows = list(_iterate_matches_rows(comps=comps_set, seasons=seasons_set))
    if not matches_rows:
        raise RuntimeError(
            "No matches found under bronze/statsbomb_open_data. "
            "Check your submodule path and (optional) comp/season filters."
        )
    matches_df = pd.DataFrame(matches_rows)

    # 2) Extract shots per match
    shot_records: List[Shot] = []
    for row in matches_rows:
        shot_records.extend(list(_iterate_shots_for_match(row)))

    if not shot_records:
        raise RuntimeError("No shots found in the selected competitions/seasons.")

    # 3) Assemble DataFrame and merge minimal match context
    shots_df = pd.DataFrame([s.__dict__ for s in shot_records])

    # Ensure essential columns exist / types are reasonable
    validation.require_columns(
        shots_df,
        [
            "match_id",
            "team_id",
            "player_id",
            "x",
            "y",
            "distance_to_goal",
            "shot_angle",
            "is_goal",
        ],
    )
    # Basic non-negative check (x,y are in [0,120]x[0,80] in StatsBomb coordinates)
    validation.assert_bounds(shots_df, "x", min_val=0.0)
    validation.assert_bounds(shots_df, "y", min_val=0.0)
    # Distance to goal should be positive when calculated
    validation.assert_bounds(shots_df, "distance_to_goal", min_val=0.0)
    # Shot angle should be positive when calculated (radians)
    validation.assert_bounds(shots_df, "shot_angle", min_val=0.0)

    # 4) Write to silver
    target = out_path or io.shots_silver_path()
    io.write_table(shots_df, target, index=False)

    return shots_df


def parse_cli() -> argparse.Namespace:
    """Parse data from CLI arguments"""
    p = argparse.ArgumentParser(
        description="Build silver/shots.parquet from StatsBomb Open Data (bronze)."
    )
    p.add_argument(
        "--competitions",
        type=str,
        default=None,
        help="Comma-separated competition_ids to include (e.g., '2,9,43'). Defaults to all.",
    )
    p.add_argument(
        "--seasons",
        type=str,
        default=None,
        help="Comma-separated season_ids to include (e.g., '3,4,44'). Defaults to all.",
    )
    p.add_argument(
        "--competition-names",
        type=str,
        default=None,
        help="Comma-separated competition names (e.g., 'Premier League,La Liga'). Alternative to --competitions.",
    )
    p.add_argument(
        "--season-years",
        type=str,
        default=None,
        help="Comma-separated season years (e.g., '2022/2023,2023/2024'). Alternative to --seasons.",
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional custom output path (defaults to data/silver/shots.parquet)",
    )
    return p.parse_args()


def _parse_id_list(s: Optional[str]) -> Optional[List[int]]:
    if not s:
        return None
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_name_list(s: Optional[str]) -> Optional[List[str]]:
    if not s:
        return None
    return [x.strip() for x in s.split(",") if x.strip()]


def _resolve_competition_names_to_ids(comp_names: List[str]) -> List[int]:
    """Convert competition names to IDs using lookup."""
    comp_ids = []
    for name in comp_names:
        comp_id = lookup.get_competition_id(name)
        if comp_id is None:
            raise ValueError(f"Unknown competition name: '{name}'")
        comp_ids.append(comp_id)
    return comp_ids


def _resolve_season_names_to_ids(
    comp_ids: List[int], season_names: List[str]
) -> List[int]:
    """Convert season names to IDs using lookup."""
    season_ids = []
    for season_name in season_names:
        found = False
        for comp_id in comp_ids:
            season_id = lookup.get_season_id(comp_id, season_name)
            if season_id is not None:
                season_ids.append(season_id)
                found = True
                break
        if not found:
            raise ValueError(
                f"Unknown season name: '{season_name}' for competitions {comp_ids}"
            )
    return season_ids


if __name__ == "__main__":
    args = parse_cli()

    comps = _parse_id_list(args.competitions)
    comp_names = _parse_name_list(args.competition_names)

    if comps and comp_names:
        raise ValueError(
            "Cannot specify both --competitions and --competition-names. Choose one."
        )

    if comp_names:
        comps = _resolve_competition_names_to_ids(comp_names)

    seasons = _parse_id_list(args.seasons)
    season_years = _parse_name_list(args.season_years)

    if seasons and season_years:
        raise ValueError(
            "Cannot specify both --seasons and --season-years. Choose one."
        )

    if not (comps or comp_names):
        raise ValueError("Must specify either --competitions or --competition-names")

    if not (seasons or season_years):
        raise ValueError("Must specify either --seasons or --season-years")

    out_path = Path(args.out) if args.out else None

    df = build_shots(comps=comps, seasons=seasons, out_path=out_path)
    print(f"✅ Built shots: {len(df):,} rows → {out_path or io.shots_silver_path()}")
