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