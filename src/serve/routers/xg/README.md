# xG (Expected Goals) API

FastAPI router that exposes the expected-goals service. All routes live under `/xg` and can be mounted from `src.serve.app`. This document acts as the canonical guide for consumers of the API.

## Structure

```
src/serve/routers/xg/
├── __init__.py
├── router.py     # FastAPI routes
└── helpers.py    # Feature building + viz utils
```

## Base URL

```
http://localhost:8000/xg
```

Swagger UI: http://localhost:8000/docs

---

## Score a Shot (`POST /xg/score`)

Calculate the expected goals probability for a single shot. Feature engineering (distance, angle, body-part one-hot) happens server-side so clients can submit clean, human-readable values.

**Query Parameters**

| Name | Type | Required | Description |
| --- | --- | --- | --- |
| `model_id` | string | No | Override active model using `name@version` or `name@stage` (e.g. `xG Bundesliga@5`). Otherwise the current selection is used. |

**Request Body**

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `x` | float | Yes | Shot X coordinate (0 own goal, 120 opponent goal). |
| `y` | float | Yes | Shot Y coordinate (0 bottom touchline, 80 top, 40 center). |
| `body_part` | string | No | `Right Foot`, `Left Foot`, `Head`, or `Other`. Defaults to `Right Foot`. |
| `is_open_play` | bool | No | Whether the shot was from open play (default `true`). |
| `one_on_one` | bool | No | Whether it was a 1v1 with the keeper (default `false`). |

> If future models add features, the router inspects `feature_names_in_` and attempts to map additional request fields automatically. Missing values default to 0.

**Response**

| Field | Type | Description |
| --- | --- | --- |
| `xG` | float | Probability of scoring (0-1). |
| `shot_distance` | float | Distance to goal in yards. |
| `shot_angle` | float | Shooting angle in degrees. |
| `quality` | string | Friendly label (`Excellent`, `Good`, `Average`, `Poor`). |

**Example**

```bash
curl -X POST "http://localhost:8000/xg/score?model_id=xG%20Bundesliga@2" \
  -H "Content-Type: application/json" \
  -d '{"x": 108.0, "y": 40.0, "body_part": "Right Foot", "is_open_play": true}'
```

```json
{
  "xG": 0.2456,
  "shot_distance": 12.0,
  "shot_angle": 18.43,
  "quality": "Good"
}
```

---

## Model Feature Discovery

Use these endpoints to inspect which raw fields a model expects. This helps when preparing datasets, validating new versions, or debugging inference drift.

### `GET /xg/models/features`

Lists every discovered model plus its ordered feature vector. Includes convenience metadata like available body-part encodings.

```bash
curl http://localhost:8000/xg/models/features
```

```json
{
  "models": [
    {
      "model_id": "xG Bundesliga@2",
      "display_name": "xG Bundesliga v2",
      "stage": "Production",
      "feature_count": 6,
      "features": [
        "shot_distance",
        "shot_angle",
        "body_part_Left Foot",
        "body_part_Right Foot",
        "body_part_Head",
        "body_part_Other"
      ],
      "body_part_options": [
        "Head",
        "Left Foot",
        "Other",
        "Right Foot"
      ]
    }
  ],
  "count": 1
}
```

### `GET /xg/models/{model_id}/features`

Fetch a single model's requirements. Remember to URL-encode the `@` symbol in the ID when calling from a browser.

```bash
curl http://localhost:8000/xg/models/xG%20Bundesliga@2/features
```

```json
{
  "model_id": "xG Bundesliga@2",
  "display_name": "xG Bundesliga v2",
  "stage": "Production",
  "run_id": "3a6c0d1b6eaa4a349a1f",
  "feature_count": 6,
  "features": [
    "shot_distance",
    "shot_angle",
    "body_part_Left Foot",
    "body_part_Right Foot",
    "body_part_Head",
    "body_part_Other"
  ],
  "body_part_options": [
    "Head",
    "Left Foot",
    "Other",
    "Right Foot"
  ]
}
```

---

## Other Endpoints

- `GET /xg/models`: list registered models with metadata and current selection.
- `POST /xg/models/select?model_id=...`: switch the active model used by `/score`.
- `GET /xg/heatmap`: transparent PNG of model predictions over the attacking half.
- `POST /xg/train`: trigger training (expects paths to local parquet files).
- `GET /xg/monitoring/drift`: Evidently report comparing live inference logs to the training reference.
- `GET /xg/health`: readiness probe.

---

## Router Usage

```python
from src.serve.routers import xg

app.include_router(xg.router)
```

---

## Helper Utilities

Key functions inside `helpers.py`:

- `build_features_for_model(model, **raw_features)` – matches the order expected by `model.feature_names_in_`.
- `build_features_from_request(model, shot)` – converts request payloads to model-ready DataFrames and returns geometry context.
- `generate_xg_heatmap(model, resolution)` – renders PNG overlay for data viz.
- `get_model_feature_names(model)` – surfaces the ordered feature vector even when the estimator does not expose `feature_names_in_`.
