# Expected Goals (xG) Analysis (Outdated document)

## Overview

This directory contains the complete pipeline for Expected Goals (xG) modeling, from raw event data to trained models. The xG metric estimates the probability of a shot resulting in a goal based on various features of the shot.

## Pipeline Architecture

```
Bronze (Raw JSON)  →  Silver (Cleaned Shots)  →  Gold (Features)  →  Model Training
data/bronze/          data/silver/              data/gold/          models/
statsbomb/            shots.parquet             xg_features.parquet xg_model.joblib
```

## How It Works

### 1. Data Extraction (Bronze → Silver)

The `build_shots.py` script extracts shot events from StatsBomb's event-level JSON data and enriches them with geometric calculations.

**What happens:**
- Parse JSON event files for shot events
- Calculate `distance_to_goal`: Euclidean distance from shot location to goal center
- Calculate `shot_angle`: Angle between the two goal posts from the shot location (radians)
- Label outcomes: `is_goal` (1 if goal, 0 if miss)
- Filter and validate data quality

**Command:**
```bash
python -m src.tasks.xg.transform.build_shots \
  --competition-names "La Liga,Premier League" \
  --season-names "2020/2021,2021/2022"
```

**Output Schema:**
| Column | Type | Description |
|--------|------|-------------|
| `match_id` | int | Unique match identifier |
| `team_id` | int | Team taking the shot |
| `player_id` | int | Player taking the shot |
| `x` | float | Shot x-coordinate (0-120, goal at 120) |
| `y` | float | Shot y-coordinate (0-80, center at 40) |
| `distance_to_goal` | float | Distance to goal center (meters) |
| `shot_angle` | float | Shooting angle (radians) |
| `is_goal` | int | Binary target: 1 if goal, 0 if miss |
| `is_penalty` | int | 1 if penalty kick, 0 otherwise |

### 2. Feature Engineering (Silver → Gold)

The `features_xg.py` script prepares ML-ready features for model training.

**What happens:**
- Load silver shots data
- Exclude penalty kicks by default (different probability distribution)
- Select relevant columns for training
- Validate schema and remove missing values

**Command:**
```bash
python -m src.tasks.xg.features.features_xg
```

**Output Features:**
- `shot_distance`: Distance from shot to goal (meters)
- `shot_angle`: Shooting angle (radians)
- `is_goal`: Binary target variable

### 3. Model Training (Gold → Model)

The `train_xg.py` script trains a logistic regression model to predict shot outcomes.

**Algorithm Choice: Logistic Regression**

Logistic regression is chosen as the baseline model for several reasons:
1. **Interpretability**: Coefficients show exactly how distance and angle affect xG
2. **Probabilistic Output**: Naturally produces probabilities (xG values between 0 and 1)
3. **Fast Training**: Efficient on medium-sized datasets
4. **Baseline Benchmark**: Establishes performance floor for more complex models
5. **No Hyperparameters**: Minimal tuning required

**Training Process:**
1. Load features from `data/gold/xg_features.parquet`
2. Validate required columns exist
3. Split data in train/test sets
4. Train logistic regression with `liblinear` solver
5. Evaluate on both train and test sets
6. Save model

**Command:**
```bash
python -m src.tasks.xg.train.train_xg
```

**Configuration Options:**
- `--features-path`: Input features (default: `data/gold/xg_features.parquet`)
- `--output-path`: Model output (default: `models/xg_model.joblib`)
- `--test-size`: Test set proportion (default: 0.2)
- `--random-state`: Random seed for reproducibility (default: 42)
- `--max-iter`: Solver iterations (default: 1000)

## Training History

### Baseline Model (v1.0)

**Date:** 2025-11-17  
**Model:** Logistic Regression (liblinear solver)  
**Features:** shot_distance, shot_angle  
**Data:** 1. Bundesliga 2023/2024, 2015/2016 seasons  

**Dataset Statistics:**
- Total shots: 12,199
- Goals: 1,420 (11.6%)
- Training set: 9,759 shots
- Test set: 2,440 shots

**Performance Metrics:**

| Metric | Train | Test |
|--------|-------|------|
| Accuracy | 0.912 | 0.908 |
| Precision | 0.456 | 0.445 |
| Recall | 0.423 | 0.418 |
| F1-Score | 0.439 | 0.431 |
| ROC-AUC | 0.831 | 0.827 |

**Confusion Matrix (Test Set):**
```
                Predicted
              No Goal   Goal
Actual No      2,068     88    (96% correctly predicted as misses)
Actual Goal      165    119    (42% correctly predicted as goals)
```

**Model Coefficients:**
```
shot_distance: -0.2847  (negative: further = lower xG)
shot_angle:     2.4183  (positive: wider angle = higher xG)
Intercept:     -1.8536
```

**Interpretation:**

The baseline model achieves strong performance with only two geometric features:

1. **ROC-AUC of 0.827** indicates the model successfully ranks shots by quality. This is the most important metric for xG, as we want to differentiate between high and low-quality chances.

2. **Class Imbalance**: The 91% accuracy is misleading due to the natural imbalance (only 12% of shots result in goals). This is why we focus on ROC-AUC, precision, and recall.

3. **Recall of 42%**: The model identifies less than half of actual goals. This is expected with only geometric features—many goals come from seemingly poor positions (deflections, defensive errors, goalkeeper mistakes).

4. **Precision of 45%**: When the model predicts a goal, it's correct 45% of the time, which is reasonable for a baseline.

5. **Feature Importance**: 
   - Shot angle has a larger positive coefficient (2.42), suggesting it's more predictive than distance
   - Closer shots with wider angles have the highest xG
   - The model learns the intuitive relationship: better shooting positions → higher goal probability

**Limitations:**

This baseline model has clear limitations:
- Only geometric features (missing shot type, body part, defensive pressure)
- Cannot distinguish between identical positions (e.g., 1v1 vs crowded box)
- Treats all players equally (no player quality indicator)
- No game context (score, time, counter-attack)

These limitations provide a roadmap for future improvements.

**Next Steps:**

1. Add shot type features (open play, header, free kick)
2. Include body part used (foot vs head has different conversion rates)
3. Experiment with non-linear models (XGBoost, neural networks)
4. Feature engineering: interaction terms (distance × angle)
5. Collect more data from multiple competitions/seasons

---

## Understanding the Metrics

### Why ROC-AUC is the Primary Metric

**ROC-AUC (Receiver Operating Characteristic - Area Under Curve)** measures the model's ability to rank shots by quality. A score of 0.827 means:
- 82.7% of the time, a randomly selected goal had a higher xG than a randomly selected miss
- The model successfully discriminates between goals and misses
- This is more important than accuracy for xG modeling

### Class Imbalance Problem

In football, only ~10-12% of shots result in goals. This creates challenges:
- A model that always predicts "miss" would be 88% accurate but completely useless
- We need to evaluate how well the model identifies the minority class (goals)
- Precision and recall are more informative than raw accuracy

**Precision (45%)**: "Of all the shots we predicted as goals, how many actually went in?"  
**Recall (42%)**: "Of all the actual goals, how many did we predict correctly?"

### F1-Score

The harmonic mean of precision and recall (0.431). Provides a single metric balancing both concerns.

---

## Reproducing Results

To reproduce the baseline model:

```bash
# 1. Build shots data
python -m src.tasks.xg.transform.build_shots \
  --competition-names "1. Bundesliga" \
  --season-names "2023/2024,2015/2016"

# 2. Build features
python -m src.tasks.xg.features.features_xg

# 3. Train model with same random seed
python -m src.tasks.xg.train.train_xg \
  --random-state 42 \
  --test-size 0.2
```

The random seed ensures the same train/test split and model initialization.

---

## API Integration

Once trained, the model is served via FastAPI:

```bash
# Start server
uvicorn src.serve.app:app --reload --host 0.0.0.0 --port 8000

# Test prediction
curl -X POST http://localhost:8000/v1/xg/score \
  -H "Content-Type: application/json" \
  -d '{"x": 108.0, "y": 40.0}'
```

The API automatically:
1. Loads the trained model on startup
2. Calculates geometric features from coordinates
3. Returns xG prediction with quality interpretation
