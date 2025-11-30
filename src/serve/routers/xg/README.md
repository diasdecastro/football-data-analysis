# xG (Expected Goals) Module

This module provides xG prediction endpoints with multi-model support.

## Structure

```
xg/
├── __init__.py        # Module exports
├── router.py          # FastAPI route handlers
└── helpers.py         # Business logic and utilities
```

## Endpoints

- `GET /v1/xg/models` - List available xG models
- `POST /v1/xg/models/select` - Select active model
- `POST /v1/xg/score` - Calculate xG for a shot
- `GET /v1/xg/heatmap` - Generate xG heatmap overlay
- `POST /v1/xg/train` - Train a new xG model
- `GET /v1/xg/monitoring/drift` - View data drift report
- `GET /v1/xg/health` - Health check

## Usage

```python
from src.serve.routers import xg

app.include_router(xg.router)
```

## Helper Functions

The `helpers.py` module provides:

- `build_features_for_model()` - Dynamic feature building based on model requirements
- `interpret_xg()` - Convert xG probability to quality rating
- `calculate_shot_features()` - Calculate distance and angle from coordinates
- `generate_xg_heatmap()` - Generate xG heatmap visualization
