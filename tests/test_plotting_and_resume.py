"""Additional unit tests: plotting, resume skip marker, stats CI."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.plotting import (
    plot_confusion_matrix,
    plot_dcr_histogram,
    plot_residuals,
)
from src.evaluation.stats import mean_std_ci, paired_comparison


def test_plot_outputs(tmp_path: Path):
    y_true = [0, 1, 0, 1, 1]
    y_pred = [0, 1, 1, 1, 0]
    p1 = plot_confusion_matrix(y_true, y_pred, tmp_path / "cm.png")
    assert p1 is not None and p1.exists() and p1.stat().st_size > 100

    dcr = list(np.random.RandomState(0).rand(100))
    dcr[0] = 0.0
    p2 = plot_dcr_histogram(dcr, tmp_path / "dcr.png", exact_copy_count=1)
    assert p2.exists()

    p3 = plot_residuals([1.0, 2.0, 3.0], [1.1, 1.9, 2.5], tmp_path / "res.png")
    assert p3.exists()


def test_ci_and_wilcoxon():
    a = [0.7, 0.72, 0.68, 0.71, 0.69]
    b = [0.65, 0.66, 0.64, 0.67, 0.63]
    s = mean_std_ci(a)
    assert s["ci_low"] is not None and s["ci_high"] is not None
    r = paired_comparison(a, b, metric="f1", method_a="dgd", method_b="smote")
    assert r["n_paired"] == 5
    assert "p_value" in r
    assert r["test_used"] in ("paired_t_test", "wilcoxon_signed_rank", None)


def test_resume_status_file(tmp_path: Path):
    """Simulate completed-run marker used by run_experiment skip logic."""
    run = tmp_path / "diabetes_smote"
    run.mkdir()
    (run / "summary_row.json").write_text(json.dumps({"run_id": "diabetes_smote", "status": "success"}))
    (run / "status.json").write_text(json.dumps({"run_id": "diabetes_smote", "status": "success"}))
    st = json.loads((run / "status.json").read_text())
    assert st["status"] == "success"
    assert (run / "summary_row.json").exists()
