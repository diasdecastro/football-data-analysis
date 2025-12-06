# Football Data Analysis

> **End-to-End Machine Learning Project for Expected Goals (xG) Prediction**
>
> An MVP demonstrating a complete ML pipeline from data ingestion to model deployment. Includes MLflow for experiment tracking and model versioning. Future roadmap includes automated retraining, monitoring, and cloud deployment.

---

## Table of Contents

- [Overview](#overview)
- [Project Perspective](#project-perspective)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Usage](#api-usage)
- [Future Roadmap](#future-roadmap)

---

## Overview

This project implements a **complete end-to-end machine learning system** for football analytics (currently **Expected Goals (xG)** prediction). It covers all stages from raw data ingestion to model training and deployment via a REST API.

### Key Features

- **Complete ML Pipeline**: Data ingestion → Feature engineering → Model training → Serving  
- **Medallion Architecture**: Bronze/Silver/Gold data layers for data quality and governance  
- **MLflow Integration**: Experiment tracking, model registry, and version management
- **Comprehensive Evaluation**: Multiple metrics (accuracy, precision, recall, F1, ROC-AUC)  
- **Validation & Testing**: Data validation utilities and schema enforcement (Testing TBI)
- **Scalable Design**: Modular codebase ready for extension

---

## Project Perspective

### Purpose

- **End-to-end ML workflow**: From raw data to deployed predictions
- **Data engineering principles**: Medallion architecture for data quality
- **Software engineering best practices**: Modular design, type safety, validation
- **MLOps foundations**: Model persistence, API serving, reproducibility

### Current Stage: **Functional MVP**

This is a **working, deployable system** that can be used for Expected Goals (xG) prediction.

**What "MVP" means here:**
- **Complete pipeline**: Every stage from data ingestion to serving works
- **Production code quality**: Type hints, validation, error handling, documentation
- **Deployed model**: REST API ready for real-time predictions
- **Evaluated performance**: Comprehensive metrics and validation
- **Room for growth**: Intentional foundation for MLOps capabilities

### Learning Objectives

1. **Data Engineering**
   - Designed medallion architecture (Bronze/Silver/Gold layers)
   - Implemented ETL (extract, transform, load) pipelines with data validation

2. **Machine Learning**
   - Feature engineering from event data
   - Model training
   - Chose interpretable model (logistic regression) for baseline

3. **Software Engineering**
   - Modular, maintainable codebase with clear separation of concerns
   - Type hints and Pydantic schemas for type safety
   - Comprehensive documentation and examples

4. **API Development**
   - Built RESTful API with FastAPI
   - Request/response validation with Pydantic
   - Auto-generated interactive documentation (?)

5. **System Design**
   - Scalable architecture ready for growth
   - Clear data flow and dependencies
   - Reproducible experiments (random seeds, versioning) (?)
   - Thought about deployment and serving

---

### What Works Right Now

1. Clone the repo and install dependencies
2. Run the complete pipeline to build datasets
3. Train an xG model with custom parameters
4. Start the API server
5. Make predictions via REST API or interactive docs
6. Interactive visualization (very barebones - not the focus of this project)
7. Evaluate model performance with comprehensive metrics
8. Process new competitions/seasons as data becomes available (not automated)

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

Choose either local development or Docker deployment:

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
- **Drift Report (evidently)**: http://localhost:8000/xg/monitoring/drift

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

The MLflow UI provides:
- **Experiments**: View all training runs under "xG Training" experiment
- **Metrics Comparison**: Compare ROC-AUC, accuracy, and other metrics across runs
- **Model Registry**: Browse all registered "xG Bundesliga" model versions
- **Artifacts**: Download trained models, plots, and other artifacts
- **Parameters**: See hyperparameters used for each training run

**Model Versioning:**

Each training run automatically registers a new model version:
- **Baseline (v1)**: Distance + Angle only
- **Current (v2)**: Distance + Angle + Body Part

The API automatically discovers all registered model versions and allows switching between them via the `/xg/models` endpoint.

---

## API Usage

The FastAPI service exposes all Expected Goals functionality under `http://localhost:8000/xg`. Common calls:

- `POST /xg/score` &mdash; predict the xG value for a single shot (optionally specify `model_id`).
- `GET /xg/models` &mdash; inspect registered MLflow versions and switch between them.
- `GET /xg/models/features` &mdash; discover the ordered feature vectors each model expects.
- `GET /xg/monitoring/drift` &mdash; open the Evidently dashboard comparing training vs inference data.

Swagger UI is available at **http://localhost:8000/docs** for quick experiments.

👉 For the complete API reference (request/response tables, examples, and additional endpoints) see [src/serve/routers/xg/README.md](src/serve/routers/xg/README.md).

### Monitoring (Basic MVP)

This project includes a **very minimal monitoring component** to track how the deployed xG model behaves in real usage. Monitoring is still a new concept to me, so this implementation is intentionally simple and mainly for learning.

#### What is monitored?

Each call to `/xg/score` logs:

- `shot_distance`
- `shot_angle`
- predicted `xG`
- model version
- timestamp

These represent the **core features of the baseline model**, and changes in these distributions can indicate that the live data no longer matches the training data.

#### Drift detection

- **Reference data** → the gold training dataset  
- **Current data** → recent inference logs  
- Drift is computed using an Evidently **data drift report**.

You can view the drift dashboard via:
```bash
http://localhost:8000/xg/monitoring/drift
```

## Future Roadmap

The current system provides a solid foundation for evolution into a production-grade MLOps platform. Here's the planned growth path:

### Phase 2: Enhanced Features
*Improve model accuracy with additional context*

### Phase 3: Advanced Modeling
*Explore more sophisticated algorithms*

### Phase 4: MLOps Integration
*Transform into production ML platform*

- [x] **MLflow**: Experiment tracking and model registry
- [ ] **DVC**: Data version control
- [x] **Model monitoring** (): Drift detection and performance tracking (very basic MVP)
- [ ] **A/B testing**: Compare model versions in production
- [ ] **Automated retraining**: Scheduled pipelines with new data
- [ ] **CI/CD**: Automated testing and deployment

### Phase 5: Production Deployment
*Scale to production workloads*

- [x] **Docker containerization**
- [ ] Cloud deployment (AWS/GCP/Azure)

### Phase 6: Additional Models
*Expand analytics capabilities*

- [ ] Pass xG (expected threat from passes)
- [ ] Match outcome prediction
- [ ] Player form analysis
- [ ] Player rating system

---

## License

This project uses StatsBomb open data. Please see their [terms and conditions](https://github.com/statsbomb/open-data#license).

---
