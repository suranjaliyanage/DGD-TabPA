"""
Build a full schema-compliant result row from evaluator outputs + run metadata.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .schema import normalize_result_row


def build_extended_summary(
    *,
    run_id: str,
    dataset: str,
    method: str,
    task_type: str,
    evaluator_summary: Dict[str, Any],
    evaluator_results: Dict[str, Any],
    random_seed: int,
    target_column: str,
    n_train: int,
    n_test: int,
    n_real: int,
    n_synthetic: int,
    ablation: str = "none",
    status: str = "success",
    runtime: Optional[Dict[str, Any]] = None,
    privacy_report: Optional[Dict[str, Any]] = None,
    mmd_final: Optional[float] = None,
    env_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    res = evaluator_results.get("resemblance", {}) or {}
    util = evaluator_results.get("utility", {}) or {}
    priv = evaluator_results.get("privacy", {}) or {}
    agg = util.get("_aggregates", {}) or {}

    compression = (
        float(n_synthetic) / float(n_real) if n_real and n_synthetic is not None else None
    )

    row: Dict[str, Any] = {
        **evaluator_summary,
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "random_seed": random_seed,
        "dataset": dataset,
        "method": method,
        "ablation": ablation,
        "task_type": task_type,
        "status": status,
        "number_of_real_rows": n_real,
        "number_of_synthetic_rows": n_synthetic,
        "compression_ratio": compression,
        "target_column": target_column,
        "train_split_size": n_train,
        "test_split_size": n_test,
        # utility aggregates
        "accuracy": agg.get("accuracy"),
        "balanced_accuracy": agg.get("balanced_accuracy"),
        "precision_macro": agg.get("precision_macro"),
        "recall_macro": agg.get("recall_macro"),
        "f1_macro": agg.get("f1_macro"),
        "f1_weighted": agg.get("f1_weighted", evaluator_summary.get("mean_tstr_f1")),
        "roc_auc": agg.get("roc_auc", evaluator_summary.get("mean_tstr_auc")),
        "pr_auc": agg.get("pr_auc"),
        "trtr_f1": agg.get("trtr_f1"),
        "trtr_auc": agg.get("trtr_auc"),
        "r2": evaluator_summary.get("mean_tstr_r2"),
        # fidelity extras
        "wasserstein_mean": res.get("wasserstein_mean"),
        "wasserstein_std": res.get("wasserstein_std"),
        "jsd_mean": res.get("jsd_mean"),
        "jsd_std": res.get("jsd_std"),
        "ks_mean": res.get("ks_mean"),
        "pcd": res.get("pcd"),
        "correlation_frobenius_norm": res.get("correlation_frobenius_norm"),
        "mmd_final": mmd_final,
        # privacy
        "dcr_median": priv.get("dcr_median"),
        "dcr_mean": priv.get("dcr_mean"),
        "dcr_5th_percentile": priv.get("dcr_5th_percentile"),
        "exact_copy_count": priv.get("exact_copy_count", priv.get("n_exact_copies")),
        "exact_copy_rate": priv.get("exact_copy_rate"),
        "mia_auc": priv.get("mia_auc"),
        "mia_accuracy": priv.get("mia_accuracy"),
        "privacy_rating": priv.get("privacy_rating"),
    }

    if privacy_report:
        row["epsilon"] = privacy_report.get("current_epsilon")
        row["target_epsilon"] = privacy_report.get("target_epsilon")
        row["delta"] = privacy_report.get("target_delta")
        row["max_grad_norm"] = privacy_report.get("max_grad_norm")
        row["noise_multiplier"] = privacy_report.get("noise_multiplier")

    if runtime:
        row.update(runtime)

    if env_meta:
        row["python_version"] = env_meta.get("python_version")
        row["torch_version"] = env_meta.get("torch_version")
        row["cuda_version"] = env_meta.get("cuda_version")
        row["git_commit"] = env_meta.get("git_commit")

    # Regression headline MAE/RMSE from preferred model
    if task_type == "regression":
        prefer = util.get("xgboost") or next(
            (m for k, m in util.items() if isinstance(m, dict) and "tstr_r2" in m),
            {},
        )
        if isinstance(prefer, dict):
            row["mae"] = prefer.get("tstr_mae")
            row["rmse"] = prefer.get("tstr_rmse")
            row["r2"] = prefer.get("tstr_r2", row.get("r2"))

    return normalize_result_row(row)
