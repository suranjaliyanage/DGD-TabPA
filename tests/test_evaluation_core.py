"""Focused unit tests for evaluation schema, stats, and resume helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation.schema import RESULT_SCHEMA, normalize_result_row, validate_row, empty_result_row
from src.evaluation.stats import mean_std_ci, paired_comparison
from src.evaluation.reporting import append_result_row, load_master_csv
from src.evaluation.evaluate import Evaluator


def test_schema_normalize_aliases():
    row = normalize_result_row(
        {"dataset": "diabetes", "method": "smote", "task": "classification", "n_exact_copies": 3}
    )
    assert row["task_type"] == "classification"
    assert row["exact_copy_count"] == 3
    assert set(RESULT_SCHEMA).issubset(set(row.keys())) or all(k in row for k in RESULT_SCHEMA[:5])


def test_validate_row_requires_ids():
    errs = validate_row({})
    assert any("run_id" in e for e in errs)


def test_mean_std_ci():
    s = mean_std_ci([1.0, 2.0, 3.0])
    assert s["n"] == 3
    assert abs(s["mean"] - 2.0) < 1e-9


def test_paired_comparison_insufficient():
    r = paired_comparison([1.0], [0.5], metric="f1")
    assert r["n_paired"] == 1


def test_append_result_row_upsert(tmp_path: Path):
    master = tmp_path / "results_master.csv"
    row1 = empty_result_row(run_id="a", dataset="diabetes", method="smote", status="success")
    row2 = empty_result_row(run_id="a", dataset="diabetes", method="smote", status="success", mean_tstr_f1=0.9)
    append_result_row(row1, master)
    append_result_row(row2, master)
    df = load_master_csv(master)
    assert len(df) == 1
    assert float(df.iloc[0]["mean_tstr_f1"]) == 0.9


def test_dcr_exact_copy_detection():
    rng = np.random.RandomState(0)
    real = pd.DataFrame({"x": rng.randn(50), "y": rng.randn(50)})
    syn = real.copy()  # exact copies
    syn2 = real + 1.0
    ev = Evaluator(random_state=0)
    # privacy method expects train/test/syn
    out = ev._evaluate_privacy(real, real.iloc[:10], syn)
    assert out["exact_copy_count"] > 0
    assert out["exact_copy_rate"] > 0
    out2 = ev._evaluate_privacy(real, real.iloc[:10], syn2)
    assert out2["exact_copy_count"] == 0


def test_classification_vs_regression_metrics():
    rng = np.random.RandomState(1)
    real_train = pd.DataFrame({"a": rng.randn(40), "b": rng.randn(40)})
    real_test = pd.DataFrame({"a": rng.randn(20), "b": rng.randn(20)})
    syn = pd.DataFrame({"a": rng.randn(30), "b": rng.randn(30)})
    ytr = (real_train["a"] > 0).astype(int).values
    yte = (real_test["a"] > 0).astype(int).values
    ysyn = (syn["a"] > 0).astype(int).values
    ev = Evaluator(random_state=1)
    out = ev.evaluate_all(
        real_train, real_test, syn, "target",
        real_train_labels=ytr, real_test_labels=yte, synthetic_labels=ysyn,
        task="classification",
    )
    assert "utility" in out
    assert any("tstr_f1" in (v or {}) for k, v in out["utility"].items() if isinstance(v, dict))

    ytr_r = real_train["a"].values
    yte_r = real_test["a"].values
    ysyn_r = syn["a"].values
    out_r = ev.evaluate_all(
        real_train, real_test, syn, "target",
        real_train_labels=ytr_r, real_test_labels=yte_r, synthetic_labels=ysyn_r,
        task="regression",
    )
    assert any("tstr_r2" in (v or {}) for k, v in out_r["utility"].items() if isinstance(v, dict))
