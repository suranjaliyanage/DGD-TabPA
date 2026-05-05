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

- Adult Income (UCI) — Binary classification, 32.5K rows, 14 features
- Indian Liver Patient (UCI) — Binary classification, 583 rows, 10 features

## Technology Stack

- Python 3.8+, PyTorch, scikit-learn
- Opacus (DP-SGD), SynthEval (evaluation)
- Flask (API), XGBoost/CatBoost (downstream classifiers)
