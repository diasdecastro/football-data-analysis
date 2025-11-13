# Build Shots Data

Processes StatsBomb Open Data to create shots dataset with geometric features.

## Usage

```bash
# All data
python -m src.tasks.xg.transform.build_shots

# Filter by competition (9=Bundesliga, 2=World Cup, 11=La Liga)
python -m src.tasks.xg.transform.build_shots --comps 9

# Filter by season
python -m src.tasks.xg.transform.build_shots --seasons 4

# Custom output
python -m src.tasks.xg.transform.build_shots --out my_shots.parquet
```

Creates `data/silver/shots.parquet` with shot data including `distance_to_goal` and `shot_angle` fields.