"""
Stable experiment result schema for reproducible evaluation reporting.

All runs write the same column set; non-applicable fields are None/NaN.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# Fixed master CSV / summary_row column order
RESULT_SCHEMA: List[str] = [
    # Identification
    "run_id",
    "timestamp",
    "random_seed",
    "dataset",
    "method",
    "ablation",
    "task_type",
    "status",
    "number_of_real_rows",
    "number_of_synthetic_rows",
    "compression_ratio",
    "target_column",
    "train_split_size",
    "test_split_size",
    # Utility — classification
    "accuracy",
    "balanced_accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "f1_weighted",
    "roc_auc",
    "pr_auc",
    "mean_tstr_f1",
    "mean_tstr_auc",
    "mean_f1_gap",
    "trtr_f1",
    "trtr_auc",
    # Utility — regression
    "r2",
    "rmse",
    "mae",
    "mean_tstr_r2",
    "mean_r2_gap",
    # Fidelity
    "wasserstein_mean",
    "wasserstein_std",
    "jsd_mean",
    "jsd_std",
    "ks_mean",
    "pcd",
    "correlation_frobenius_norm",
    "mmd_final",
    # Privacy
    "dcr_median",
    "dcr_mean",
    "dcr_5th_percentile",
    "exact_copy_count",
    "exact_copy_rate",
    "mia_auc",
    "mia_accuracy",
    "privacy_rating",
    "epsilon",
    "target_epsilon",
    "delta",
    "noise_multiplier",
    "max_grad_norm",
    # Runtime / resources
    "training_time_seconds",
    "distillation_time_seconds",
    "sampling_time_seconds",
    "evaluation_time_seconds",
    "total_runtime_seconds",
    "peak_cpu_ram_mb",
    "peak_gpu_memory_mb",
    "device_name",
    "number_of_trainable_parameters",
    # Reproducibility
    "python_version",
    "torch_version",
    "cuda_version",
    "git_commit",
]


def empty_result_row(**overrides: Any) -> Dict[str, Any]:
    """Return a full-schema row with None defaults, then apply overrides."""
    row: Dict[str, Any] = {k: None for k in RESULT_SCHEMA}
    for k, v in overrides.items():
        if k in row:
            row[k] = v
        else:
            row[k] = v  # allow extra keys; master writer still uses schema order
    return row


def normalize_result_row(partial: Dict[str, Any]) -> Dict[str, Any]:
    """Merge a partial summary into the fixed schema."""
    row = empty_result_row()
    for k, v in partial.items():
        row[k] = v
    # Backward-compatible aliases from older summary_row fields
    if row.get("task_type") is None and partial.get("task") is not None:
        row["task_type"] = partial["task"]
    if row.get("exact_copy_count") is None and partial.get("n_exact_copies") is not None:
        row["exact_copy_count"] = partial["n_exact_copies"]
    if row.get("roc_auc") is None and partial.get("mean_tstr_auc") is not None:
        row["roc_auc"] = partial["mean_tstr_auc"]
    if row.get("f1_weighted") is None and partial.get("mean_tstr_f1") is not None:
        row["f1_weighted"] = partial["mean_tstr_f1"]
    if row.get("r2") is None and partial.get("mean_tstr_r2") is not None:
        row["r2"] = partial["mean_tstr_r2"]
    return row


def validate_row(row: Dict[str, Any], require_run_id: bool = True) -> List[str]:
    """Return list of critical validation errors (empty if OK)."""
    errors: List[str] = []
    if require_run_id and not row.get("run_id"):
        errors.append("missing run_id")
    if not row.get("dataset"):
        errors.append("missing dataset")
    if not row.get("method"):
        errors.append("missing method")
    status = row.get("status") or "success"
    if status == "success":
        for key in ("wasserstein_mean", "dcr_median"):
            v = row.get(key)
            if v is None:
                continue
            try:
                import math

                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    errors.append(f"non-finite {key}")
            except Exception:
                pass
    return errors
