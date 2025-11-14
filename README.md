# Football Data Analysis

A project for analyzing football data using StatsBomb open data for learning purposes.

## Project Structure

```
├── data/                   # Data storage (bronze, silver, gold tiers)
├── docker/                 # Docker configuration files
├── src/                    # Source code
│   ├── common/             # Shared utilities and common functions
│   ├── serve/              # API serving components
│   └── tasks/              # Data processing tasks
│       └── xg/             # Expected Goals (xG) modeling
│           ├── features/   # Feature engineering
│           ├── train/      # Model training
│           └── transform/  # Data transformation
└── tests/                  # Test files
```

## Overview

This project implements a data pipeline for football analytics, focusing on Expected Goals (xG) modeling using StatsBomb data. The architecture follows a medallion data structure (bronze → silver → gold) for data processing and includes machine learning components for predictive modeling.

## Expected Goals (xG)

### Build shots data (silver)

**Example:**
```bash
python -m src.tasks.xg.transform.build_shots_data --competition_names "1. Bundesliga" --season_names "2020/2021"
```
**Options:**
```bash
--competitions: Comma-separated list of competition IDs
--seasons: Comma-separated list of season IDs
--competition_names: Comma-separated list of competition names
--season_names: Comma-separated list of season names
--out: Output path for the silver shots data
```

### Build xG features (gold)

```bash
python -m src.tasks.xg.features.build_xg_features
```

### train xG model (gold)

```bash
python -m src.tasks.xg.train.train_xg_model
```
**Options:**
```bash
--features-path: Path to the features data
--output-path: Path to save the trained model
--test-size: Proportion of data for testing
--random-state: Random seed for reproducibility
--max-iter: Maximum iterations for solver convergence
```

### Features
 - Shot distance to goal
 - Shot angle
 - Outcome (goal or no goal)
 - Shot type (e.g., open play, set piece) TBI
 - Body part used (e.g., foot, head) TBI
 - Defensive pressure TBI
 - Player position TBI
 - Is counterattack TBI
 - Is home team TBI
