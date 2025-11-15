from __future__ import annotations

import json
from pathlib import Path
from functools import lru_cache
from typing import Dict, Optional, List

from src.common.io import bronze

BRONZE_ROOT = bronze("statsbomb_open_data/data")


# -------------------------------
# Helpers
# -------------------------------


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# -------------------------------
# Competition / Season Lookups
# -------------------------------


@lru_cache(maxsize=1)
def _load_competitions() -> List[dict]:
    comp_path = BRONZE_ROOT / "competitions.json"
    return _read_json(comp_path)


def get_competition_id(name: str) -> Optional[int]:
    name = name.lower().strip()
    for c in _load_competitions():
        if c["competition_name"].lower() == name:
            return c["competition_id"]
    return None


def get_competition_name(comp_id: int) -> Optional[str]:
    for c in _load_competitions():
        if c["competition_id"] == comp_id:
            return c["competition_name"]
    return None


def get_seasons_for_competition(comp_id: int) -> List[dict]:
    return [c for c in _load_competitions() if c["competition_id"] == comp_id]


def get_season_id(comp_id: int, season_name: str) -> Optional[int]:
    season_name = season_name.lower().strip()
    for c in get_seasons_for_competition(comp_id):
        if c["season_name"].lower() == season_name:
            return c["season_id"]
    return None


def get_season_name(comp_id: int, season_id: int) -> Optional[str]:
    for c in get_seasons_for_competition(comp_id):
        if c["season_id"] == season_id:
            return c["season_name"]
    return None


# -------------------------------
# Team Lookups
# -------------------------------


@lru_cache(maxsize=1)
def _load_all_teams() -> Dict[int, str]:
    """
    Extract team_id → name by scanning all matches in the bronze data.
    """
    teams = {}
    matches_dir = BRONZE_ROOT / "matches"

    for comp_dir in matches_dir.glob("*"):
        if not comp_dir.is_dir():
            continue
        for season_file in comp_dir.glob("*.json"):
            for m in _read_json(season_file):
                home = m["home_team"]
                away = m["away_team"]
                teams[int(home["home_team_id"])] = home["home_team_name"]
                teams[int(away["away_team_id"])] = away["away_team_name"]

    return teams


def get_team_name(team_id: int) -> Optional[str]:
    return _load_all_teams().get(team_id)


def get_team_id(team_name: str) -> Optional[int]:
    team_name = team_name.lower().strip()
    for tid, name in _load_all_teams().items():
        if name.lower() == team_name:
            return tid
    return None


# -------------------------------
# Player Lookups
# -------------------------------


@lru_cache(maxsize=1)
def _load_all_players() -> Dict[int, str]:
    """
    Collect all player_id → name mappings from events and lineups.
    """
    players = {}

    events_dir = BRONZE_ROOT / "events"
    lineups_dir = BRONZE_ROOT / "lineups"

    # Scan lineups (more complete source of player → name)
    for lineup_file in lineups_dir.glob("*.json"):
        data = _read_json(lineup_file)
        for team_block in data:  # format: [{team, lineup:[...]}]
            for player in team_block.get("lineup", []):
                pid = player.get("player_id")
                pname = player.get("player_name")
                if pid and pname:
                    players[int(pid)] = pname

    # Fallback: scan events for any missing players
    for ev_file in events_dir.glob("*.json"):
        events = _read_json(ev_file)
        for ev in events:
            p = ev.get("player")
            if not p:
                continue
            pid = p.get("id")
            pname = p.get("name")
            if pid and pname:
                players[int(pid)] = pname

    return players


def get_player_name(player_id: int) -> Optional[str]:
    return _load_all_players().get(player_id)


def get_player_id(player_name: str) -> Optional[int]:
    player_name = player_name.lower().strip()
    for pid, name in _load_all_players().items():
        if name.lower() == player_name:
            return pid
    return None
