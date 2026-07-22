"""
SMOTE baseline for comparative evaluation against DGD-TabPA.

Generates synthetic tabular samples via oversampling / interpolation so that
DGD-TabPA can be compared under the same TSTR + DCR protocol.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_smote_synthetic(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    target_col: str,
    n_synthetic: int = 500,
    random_state: int = 42,
    task: str = "classification",
) -> pd.DataFrame:
    """
    Produce a synthetic DataFrame of size n_synthetic.
    Classification: SMOTE (or NN interpolation fallback).
    Regression: bootstrap + Gaussian noise on numeric features.
    """
    y_train = np.asarray(y_train)
    if task == "regression":
        return _regression_noise_baseline(
            X_train, y_train, target_col, n_synthetic, random_state
        )

    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X_train.columns if c not in num_cols]

    if not num_cols:
        raise ValueError("SMOTE baseline requires at least one numeric feature")

    X_num = X_train[num_cols].fillna(X_train[num_cols].median()).values.astype(float)

    try:
        from imblearn.over_sampling import SMOTE

        # SMOTE needs enough samples per class; fall back if not
        classes, counts = np.unique(y_train, return_counts=True)
        k = int(min(5, counts.min() - 1)) if counts.min() > 1 else 1
        if k < 1:
            raise ValueError("Not enough samples per class for SMOTE")

        # Oversample until we have at least n_synthetic rows, then subsample
        # Target: equal class balance with total >= n_synthetic
        n_per = max(int(np.ceil(n_synthetic / len(classes))), int(counts.max()))
        sampling_strategy = {c: max(n_per, int(counts[i])) for i, c in enumerate(classes)}
        smote = SMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=k,
            random_state=random_state,
        )
        X_res, y_res = smote.fit_resample(X_num, y_train)
    except Exception:
        X_res, y_res = _nn_interpolate(X_num, y_train, n_synthetic, random_state)

    rng = np.random.RandomState(random_state)
    if len(X_res) > n_synthetic:
        idx = rng.choice(len(X_res), n_synthetic, replace=False)
        X_res, y_res = X_res[idx], y_res[idx]
    elif len(X_res) < n_synthetic:
        # pad with NN interpolation
        extra_X, extra_y = _nn_interpolate(
            X_num, y_train, n_synthetic - len(X_res), random_state + 1
        )
        X_res = np.vstack([X_res, extra_X])
        y_res = np.concatenate([y_res, extra_y])

    syn = pd.DataFrame(X_res, columns=num_cols)

    # Sample categorical columns from real class-conditional empirical frequencies
    for col in cat_cols:
        syn[col] = _sample_categorical(X_train[col], y_train, y_res, random_state)

    syn[target_col] = y_res
    # Preserve column order: features then target
    ordered = [c for c in X_train.columns if c in syn.columns] + [target_col]
    return syn[ordered]


def _regression_noise_baseline(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    target_col: str,
    n_synthetic: int,
    random_state: int,
) -> pd.DataFrame:
    """Bootstrap real rows and add small Gaussian noise to numeric features."""
    rng = np.random.RandomState(random_state)
    idx = rng.choice(len(X_train), n_synthetic, replace=True)
    syn = X_train.iloc[idx].reset_index(drop=True).copy()
    num_cols = syn.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        std = float(X_train[col].std() or 1.0)
        syn[col] = syn[col].astype(float) + rng.normal(0, 0.05 * std, size=len(syn))
    syn[target_col] = np.asarray(y_train, dtype=float)[idx]
    ordered = [c for c in X_train.columns if c in syn.columns] + [target_col]
    return syn[ordered]


def _nn_interpolate(
    X: np.ndarray,
    y: np.ndarray,
    n: int,
    random_state: int,
) -> tuple:
    """Simple SMOTE-like interpolation fallback without imblearn."""
    rng = np.random.RandomState(random_state)
    classes = np.unique(y)
    out_X, out_y = [], []
    per_class = max(1, n // max(len(classes), 1))

    for c in classes:
        xc = X[y == c]
        if len(xc) == 0:
            continue
        for _ in range(per_class):
            i = rng.randint(0, len(xc))
            j = rng.randint(0, len(xc))
            alpha = rng.rand()
            out_X.append(xc[i] + alpha * (xc[j] - xc[i]))
            out_y.append(c)

    if not out_X:
        idx = rng.choice(len(X), n, replace=True)
        return X[idx].copy(), y[idx].copy()

    X_out = np.vstack(out_X)[:n]
    y_out = np.array(out_y)[:n]
    return X_out, y_out


def _sample_categorical(
    series: pd.Series,
    y_real: np.ndarray,
    y_syn: np.ndarray,
    random_state: int,
) -> np.ndarray:
    rng = np.random.RandomState(random_state)
    out = []
    for c in y_syn:
        mask = y_real == c
        pool = series[mask] if mask.any() else series
        out.append(rng.choice(pool.astype(str).values))
    return np.array(out)
