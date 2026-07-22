# DGD-TabPA: Diffusion-Guided Dataset Distillation for Tabular Data with Privacy-Aware Evaluation

MSc Advanced Software Engineering Dissertation Project — University of Westminster

## Overview

DGD-TabPA is a framework that condenses large tabular datasets into small, high-utility synthetic summaries using diffusion-guided distillation, while providing measurable privacy guarantees through Differential Privacy (DP-SGD).

## Architecture

The system consists of five interconnected modules:

1. **Heterogeneous Data Ingestion & Preprocessing** — Gaussian quantile transforms for numerical features, one-hot encoding for categorical features
2. **Structure-Aware Latent Manifold Encoding** — Transformer encoder-decoder with a Conditioning Attention Mechanism
3. **Diffusion-Guided Distillation Loop** — Bi-level optimization condensing N real samples into M synthetic latent codes
4. **Privacy Guardrail & Sanitization** — DP-SGD via Opacus for configurable privacy budgets
5. **Multi-Dimensional Evaluation** — SynthEval-based pipeline measuring DCR, TSTR F1-score, and Wasserstein distance

## Setup

```bash
pip install -r requirements.txt
python scripts/download_data.py
```

## Usage

**Training the diffusion model:**
```bash
python scripts/train.py --config config/default.yaml
```

**Running the demo notebook:**
```bash
jupyter notebook notebooks/prototype_demo.ipynb
```

**Starting the API server:**
```bash
python -m src.api.app
```

## Benchmark Datasets

Download all prepared CSVs into `data/raw/`:

```bash
python scripts/download_data.py              # full suite
python scripts/download_data.py --dataset cvd  # single dataset
```

### 1. Generic Industrial Benchmarks

| Key | Dataset | Task | Target | Source |
|-----|---------|------|--------|--------|
| `adult` | Adult Income (Census Income) | Classification | `income` | [UCI](https://doi.org/10.24432/C5XW20) |
| `churn` | Churn Modelling | Classification | `Exited` | [Kaggle](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling) |
| `credit` | Default of Credit Card Clients | Classification | `default` | [Kaggle](https://www.kaggle.com/uciml/default-of-credit-card-clients-dataset) |
| `covertype` | Forest Cover Type | Classification | `Cover_Type` | [UCI](https://doi.org/10.24432/C50K5N) |

### 2. Specialized Medical Benchmarks

| Key | Dataset | Task | Target | Source |
|-----|---------|------|--------|--------|
| `cvd` | Cardiovascular Disease (CVD) | Classification | `cardio` | [Kaggle](https://www.kaggle.com/sulianova/cardiovascular-disease-dataset) |
| `hcv` | Hepatitis C Virus (Egyptian patients) | Classification | `histological_staging` | [UCI](https://archive.ics.uci.edu/dataset/503) |
| `ilpd` | Indian Liver Patient (ILP) | Classification | `is_patient` | [UCI](https://archive.ics.uci.edu/ml/datasets/ILPD+(Indian+Liver+Patient+Dataset)) |
| `diabetes` | Pima Indians Diabetes | Classification | `Outcome` | [Kaggle](https://kaggle.com/uciml/pima-indians-diabetes-database) |

### 3. Regression Benchmarks

| Key | Dataset | Task | Target | Source |
|-----|---------|------|--------|--------|
| `california_housing` | California Housing | Regression | `MedHouseVal` | [Kaggle](https://www.kaggle.com/camnugent/california-housing-prices) |
| `king_county` | House Sales in King County | Regression | `price` | [Kaggle](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction) |

Train a specific dataset:

```bash
python scripts/train.py --config config/default.yaml --dataset diabetes --epochs 50
```

> Note: the current training / distillation loop is classification-oriented (label embeddings + TSTR F1). Regression CSVs are included for evaluation and future regression support.

## Technology Stack

- Python 3.8+, PyTorch, scikit-learn
- Opacus (DP-SGD), SynthEval (evaluation)
- Flask (API), XGBoost/CatBoost (downstream classifiers)
