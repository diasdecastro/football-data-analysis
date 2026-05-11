# Football Data Analysis

---

## Overview

This project implements a **complete end-to-end machine learning system** for football analytics (**Expected Goals (xG)**). It covers all stages from raw data ingestion to model training and deployment via a REST API.

### Key Features

- **Complete ML Pipeline**: Data ingestion → Feature engineering → Model training → Serving  
- **Medallion Architecture**: Bronze/Silver/Gold data layers for data quality and governance  
- **MLflow Integration**: Experiment tracking, model registry, and version management
- **Comprehensive Evaluation**: Multiple metrics (accuracy, precision, recall, F1, ROC-AUC)
- **Scalable Design**: Modular codebase ready for extension

---

## Architecture

### Medallion Architecture

**Bronze Layer**: Raw data as-is from source (StatsBomb JSON)  
**Silver Layer**: Cleaned, validated, and enriched data (Parquet format)  
**Gold Layer**: Feature-engineered, ML-ready datasets optimized for specific models

---

## Project Structure

```
football-data-analysis/
├── data/                           # Data storage (gitignored except bronze)
│   ├── bronze/                     # Raw data from StatsBomb
│   │   └── statsbomb_open_data/    # Git submodule with event data
│   ├── silver/                     # Processed shot data
│   │   └── shots.parquet           # Cleaned shot events
│   └── gold/                       # ML-ready features
│       └── xg_features.parquet     # Training dataset
│
├── models/                         # Trained models (gitignored)
│   └── xg_model.joblib             # Serialized logistic regression
│
├── src/                            # Source code
│   ├── common/                     # Shared utilities
│   │   ├── geometry.py             # Shot distance & angle calculations
│   │   ├── io.py                   # Data reading/writing utilities
│   │   ├── lookup.py               # Competition/season/team lookups
│   │   └── validation.py           # Data validation functions
│   │
│   ├── tasks/                      # Data processing & ML tasks
│   │   ├── xg/                     # Expected Goals pipeline
│   │   │   ├── transform/          # Bronze → Silver
│   │   │   ├── features/           # Silver → Gold
│   │   │   └── train/              # Model training
│   │   └── some_other_task/        # E.g. Score prediction
│   │
│   └── serve/                      # API serving
│       ├── app.py                  # FastAPI application
│       ├── loaders.py              # Model loading utilities
│       ├── schemas.py              # Pydantic request/response models
│       └── routers/                # API route definitions   
│           └── xg_router.py        # xG prediction endpoints
│
├── tests/                          # Unit and integration tests
├── docker/                         # Docker configuration (WIP)
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── .gitignore                      # Git ignore rules
```

---

## Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager
- Git (for cloning repository and submodules)

### Setup Steps

1. **Clone the repository with submodules**

```bash
git clone --recurse-submodules https://github.com/yourusername/football-data-analysis.git
cd football-data-analysis
```

2. **Create virtual environment**

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

---

## Quick Start

### Step 1: Data Pipeline & Model Training

Run the complete pipeline from raw data to trained model:

```bash
# 1. Build shots data (Bronze → Silver)
python -m src.tasks.xg.transform.build_shots \
  --competition-names "La Liga" \
  --season-names "2020/2021"

# 2. Build features (Silver → Gold)
python -m src.tasks.xg.features.features_xg

# 3. Train model (Gold → Model)
python -m src.tasks.xg.train.train_xg --run-name "v1" --model-name "xG Bundesliga" # Optional run name for MLflow
```

### Step 2: Serve the API

**Option A: Local Development**

```bash
# Start API server
uvicorn src.serve.app:app --reload --host 0.0.0.0 --port 8000

# Start MLflow UI (in another terminal)
mlflow ui
```

**Services:**
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **MLflow UI**: http://localhost:5000

**Option B: Docker Deployment**

```bash
# Build and start all services
cd docker
docker compose up -d --build

# View logs
docker compose logs -f api
docker compose logs -f mlflow-ui

# Stop services
docker compose down
```

**Services:**
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **MLflow UI**: http://localhost:5001

**Persistent Data:**

The Docker setup mounts local directories, so models and experiments trained on your host machine are automatically available in containers:
- `mlruns/`: MLflow experiments and runs
- `models/`: Trained model artifacts
- `data/gold/` and `data/silver/`: Processed datasets (read-only)

### MLflow Tracking & Model Registry

All training runs are automatically logged to MLflow with metrics, parameters, and model artifacts.

**Start MLflow UI:**

```bash
mlflow ui
```

Then open your browser to: **http://localhost:5000**

---

## API Usage

The FastAPI service exposes all Expected Goals functionality under `http://localhost:8000/xg`. Common calls:

- `POST /xg/score` &mdash; predict the xG value for a single shot (optionally specify `model_id`).
- `GET /xg/models` &mdash; inspect registered MLflow versions and switch between them.
- `GET /xg/models/features` &mdash; discover the ordered feature vectors each model expects.

Swagger UI is available at **http://localhost:8000/docs** for quick experiments.

For the complete API reference (request/response tables, examples, and additional endpoints) see [src/serve/routers/xg/README.md](src/serve/routers/xg/README.md).

#### What is monitored?

Each call to `/xg/score` logs:

- `shot_distance`
- `shot_angle`
- predicted `xG`
- model version
- timestamp

These represent the **core features of the baseline model**, and changes in these distributions can indicate that the live data no longer matches the training data.