import json, pathlib, pandas as pd

RAW_DIR = pathlib.Path("data/bronze")
OUT = pathlib.Path("data/silver/shots.parquet")
OUT.parent.mkdir(parents=True, exist_ok=True)


def collect_shots():
    rows = []
    for p in RAW_DIR.rglob("*.json"):
        events = json.loads(p.read_text(encoding="utf-8"))
        for ev in events:
            if ev.get("type", {}).get("name") == "Shot":
                loc = ev.get("location", [None, None])
                rows.append(
                    {
                        "match_id": ev.get("match_id"),
                        "team": ev.get("team", {}).get("name"),
                        "player": ev.get("player", {}).get("name"),
                        "x": loc[0],
                        "y": loc[1],
                        "minute": ev.get("minute"),
                        "second": ev.get("second"),
                        "body_part": ev.get("shot", {})
                        .get("body_part", {})
                        .get("name"),
                        "technique": ev.get("shot", {})
                        .get("technique", {})
                        .get("name"),
                        "situation": ev.get("shot", {}).get("type", {}).get("name"),
                        "outcome": ev.get("shot", {}).get("outcome", {}).get("name"),
                        "is_goal": (
                            1
                            if ev.get("shot", {}).get("outcome", {}).get("name")
                            == "Goal"
                            else 0
                        ),
                    }
                )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = collect_shots()
    df.to_parquet(OUT, index=False)
    print(f"Wrote {len(df):,} shots → {OUT}")
