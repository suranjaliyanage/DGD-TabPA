# DGD-TabPA: Diffusion-Guided Dataset Distillation for Tabular Data with Privacy-Aware Evaluation

MSc Advanced Software Engineering Dissertation Project — University of Westminster

## Overview

DGD-TabPA condenses large tabular datasets into small, high-utility synthetic summaries using diffusion-guided distillation, with measurable privacy guarantees via Differential Privacy (DP-SGD).

## Architecture

1. **Heterogeneous Data Ingestion & Preprocessing** — Gaussian quantile transforms + one-hot encoding  
2. **Structure-Aware Latent Manifold Encoding** — Transformer encoder-decoder with Conditioning Attention  
3. **Diffusion-Guided Distillation Loop** — Bi-level optimisation (N real → M synthetic)  
4. **Privacy Guardrail & Sanitization** — DP-SGD (Opacus) with configurable ε  
5. **Multi-Dimensional Evaluation** — Fidelity (WD / JSD / PCD), TSTR (F1 / ROC-AUC), privacy (DCR / MIA)

---

## Setup (once)

```bash
pip install -r requirements.txt
python scripts/download_data.py
```

Single dataset only:

```bash
python scripts/download_data.py --dataset diabetes
```

---

## Full process sequence

Run these **in order**. Each step appends to `outputs/experiments/results_master.csv` and writes figures under `outputs/experiments/<run_id>/figures/`.

### Recommended epoch settings

| Purpose | `--epochs` | `--distill-epochs` |
|---------|------------|--------------------|
| Smoke / debug | 2–5 | 10 |
| Draft evaluation numbers | 30–50 | 50 |
| Stronger final numbers | 100+ | 100+ |

### Step 1 — Single end-to-end DGD run (minimum complete pipeline)

Train → distill → evaluate → save metrics + figures for one dataset:

```bash
python scripts/run_experiment.py ^
  --config config/default.yaml ^
  --dataset diabetes ^
  --method dgd_tabpa ^
  --epochs 50 ^
  --distill-epochs 50 ^
  --run-id diabetes_dgd
```

(Linux/macOS: replace `^` with `\`.)

### Step 2 — SMOTE baseline (same TSTR / DCR protocol)

```bash
python scripts/run_experiment.py ^
  --dataset diabetes ^
  --method smote ^
  --run-id diabetes_smote
```

### Step 3 — Multi-dataset core quantitative table

Fast subset (diabetes, ilpd, churn, adult):

```bash
python scripts/run_batch_experiments.py --suite core_fast --epochs 50
```

Full classification suite:

```bash
python scripts/run_batch_experiments.py --suite core --epochs 50
```

### Step 4 — SMOTE across datasets

```bash
python scripts/run_batch_experiments.py --suite smote
```

### Step 5 — Ablation studies

```bash
python scripts/run_batch_experiments.py --suite ablations --dataset diabetes --epochs 30
```

Or one ablation at a time:

```bash
python scripts/run_experiment.py --dataset diabetes --ablation mlp_denoiser --epochs 30 --distill-epochs 50
python scripts/run_experiment.py --dataset diabetes --ablation no_attention --epochs 30 --distill-epochs 50
python scripts/run_experiment.py --dataset diabetes --ablation minmax --epochs 30 --distill-epochs 50
python scripts/run_experiment.py --dataset diabetes --ablation raw_space --epochs 30 --distill-epochs 50
```

| Ablation flag | What it tests |
|---------------|---------------|
| `mlp_denoiser` | Transformer vs MLP denoiser |
| `no_attention` | Conditioning Attention on/off |
| `minmax` | Quantile vs min-max preprocessing |
| `raw_space` | Latent vs raw-space distillation |

### Step 6 — Privacy–utility sweep

Non-private + ε ∈ {1, 4, 8, 100}:

```bash
python scripts/run_batch_experiments.py --suite privacy --dataset diabetes --epochs 30
```

Single private run:

```bash
python scripts/run_experiment.py --dataset diabetes --privacy --epsilon 4.0 --epochs 30 --distill-epochs 50 --run-id diabetes_eps4
```

### Step 7 — Collect evaluation artefacts

After the suites finish:

| Artefact | Location | Use |
|----------|----------|-----|
| Master comparison table | `outputs/experiments/results_master.csv` | Numeric comparison tables |
| Per-run metrics | `outputs/experiments/<run_id>/metrics.json` | Detailed reporting |
| Per-run summary row | `outputs/experiments/<run_id>/summary_row.json` | Quick paste into tables |
| Figures | `outputs/experiments/<run_id>/figures/*.png` | Qualitative plots + ROC / DCR / loss |
| Privacy trade-off plot | `outputs/experiments/privacy_utility_<dataset>.png` | Privacy–utility curves |
| Synthetic CSV | `outputs/experiments/<run_id>/synthetic_*.csv` | Appendix / inspection |

Figure types produced per successful DGD run:

- `train_loss_*.png`, `distill_loss_*.png`
- `marginals_*.png`, `categorical_*.png`
- `correlation_*.png`
- `manifold_*.png` (PCA / t-SNE)
- `roc_*.png`, `dcr_*.png`

### One-shot “everything” (long-running)

```bash
python scripts/run_batch_experiments.py --suite all --dataset diabetes --epochs 30
```

This runs core + smote + privacy + ablations. Prefer the stepped sequence above so you can inspect outputs between stages.

---

## Notebooks

| Notebook | Purpose |
|----------|---------|
| [`notebooks/full_experiments.ipynb`](notebooks/full_experiments.ipynb) | Guided full experiment sequence (local) |
| [`notebooks/colab_full_experiments.ipynb`](notebooks/colab_full_experiments.ipynb) | **Google Colab** full experiment sequence (GPU) |
| [`notebooks/prototype_demo.ipynb`](notebooks/prototype_demo.ipynb) | Module / API demo & positive-negative test cases |

### Local

```bash
jupyter notebook notebooks/full_experiments.ipynb
```

### Google Colab

1. Open [Google Colab](https://colab.research.google.com/)  
2. **File → Upload notebook** and select `notebooks/colab_full_experiments.ipynb`  
   — or open from GitHub: `suranjaliyanage/DGD-TabPA` → `notebooks/colab_full_experiments.ipynb`  
3. **Runtime → Change runtime type → GPU**  
4. Set `EPOCHS` / `DISTILL_EPOCHS` (and `RUN_HEAVY` if needed) in the config cell  
5. **Runtime → Run all**  
6. Download `dgd_tabpa_outputs_*.zip` when the final cell finishes  

If the GitHub repo is private, set `GITHUB_TOKEN` in the config cell, or upload the project to Drive and set `USE_DRIVE = True`.

In the local notebook:

1. Set `EPOCHS` / `DISTILL_EPOCHS`  
2. Run Steps 1–4 for a complete single-dataset evidence pack  
3. Set `RUN_HEAVY = True` to execute multi-dataset / ablation / privacy suites  
4. The final step builds a cleaned `results_table.csv` for reporting  

---

## Other scripts (optional)

**Train only** (no distill/eval):

```bash
python scripts/train.py --config config/default.yaml --dataset diabetes --epochs 50
```

**Inspect an existing checkpoint:**

```bash
python scripts/inspect_outputs.py --dataset adult --n-samples 200
```

**API server:**

```bash
python -m src.api.app
```

---

## Checklist — process complete?

- [ ] Data downloaded into `data/raw/`
- [ ] At least one DGD run finished (`metrics.json` + `figures/`)
- [ ] SMOTE baseline run for comparison
- [ ] Core table has multiple datasets (or `core_fast`)
- [ ] Ablations and/or privacy sweep done for discussion
- [ ] `results_master.csv` reviewed and figures ready for reporting

---

## Benchmark datasets

### 1. Generic industrial

| Key | Dataset | Task | Target |
|-----|---------|------|--------|
| `adult` | Adult Income | Classification | `income` |
| `churn` | Churn Modelling | Classification | `Exited` |
| `credit` | Credit Card Default | Classification | `default` |
| `covertype` | Forest Cover Type | Classification | `Cover_Type` |

### 2. Medical

| Key | Dataset | Task | Target |
|-----|---------|------|--------|
| `cvd` | Cardiovascular Disease | Classification | `cardio` |
| `hcv` | Hepatitis C (Egyptian) | Classification | `histological_staging` |
| `ilpd` | Indian Liver Patient | Classification | `is_patient` |
| `diabetes` | Pima Indians Diabetes | Classification | `Outcome` |

### 3. Regression (data available; the TSTR experiment runner is classification-oriented)

| Key | Dataset | Target |
|-----|---------|--------|
| `california_housing` | California Housing | `MedHouseVal` |
| `king_county` | King County House Sales | `price` |

---

## Technology stack

- Python 3.8+, PyTorch, scikit-learn  
- Opacus (DP-SGD), SynthEval (optional)  
- XGBoost / CatBoost, imbalanced-learn (SMOTE)  
- Flask API, matplotlib / seaborn for figures  
