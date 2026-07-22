"""
Evaluation reporting: summary tables, aggregated figures, Markdown report.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .schema import RESULT_SCHEMA, normalize_result_row
from .stats import mean_std_ci, paired_comparison


def load_master_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=RESULT_SCHEMA)
    try:
        df = pd.read_csv(path)
    except Exception:
        # Recover from historically mixed schemas
        try:
            df = pd.read_csv(path, engine="python", on_bad_lines="skip")
        except Exception:
            bak = path.with_suffix(".csv.bak")
            try:
                path.replace(bak)
            except Exception:
                pass
            return pd.DataFrame(columns=RESULT_SCHEMA)
    rows = [normalize_result_row(r) for r in df.to_dict(orient="records")]
    return pd.DataFrame(rows)


def write_master_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [c for c in RESULT_SCHEMA if c in df.columns] + [
        c for c in df.columns if c not in RESULT_SCHEMA
    ]
    out = df.reindex(columns=cols)
    # Atomic write
    tmp = path.with_suffix(".csv.tmp")
    out.to_csv(tmp, index=False)
    tmp.replace(path)
    return path


def append_result_row(row: Dict[str, Any], master_csv: Path) -> Path:
    """Atomically append one normalized row with stable schema."""
    master_csv.parent.mkdir(parents=True, exist_ok=True)
    norm = normalize_result_row(row)
    df_new = pd.DataFrame([norm])
    if master_csv.exists():
        try:
            df_old = pd.read_csv(master_csv)
        except Exception:
            try:
                df_old = pd.read_csv(master_csv, engine="python", on_bad_lines="skip")
            except Exception:
                bak = master_csv.with_suffix(".csv.bak")
                try:
                    master_csv.replace(bak)
                except Exception:
                    pass
                df_old = pd.DataFrame()
        # Drop prior row with same run_id if present
        if "run_id" in df_old.columns and norm.get("run_id"):
            df_old = df_old[df_old["run_id"] != norm["run_id"]]
        # Align columns via normalize
        old_rows = [normalize_result_row(r) for r in df_old.to_dict(orient="records")]
        df = pd.DataFrame(old_rows + [norm])
    else:
        df = df_new
    return write_master_csv(df, master_csv)


KEY_TABLE_COLS = [
    "run_id",
    "dataset",
    "method",
    "ablation",
    "task_type",
    "status",
    "mean_tstr_f1",
    "mean_tstr_auc",
    "mean_f1_gap",
    "mean_tstr_r2",
    "r2",
    "rmse",
    "mae",
    "wasserstein_mean",
    "jsd_mean",
    "pcd",
    "dcr_median",
    "dcr_5th_percentile",
    "exact_copy_count",
    "mia_auc",
    "privacy_rating",
    "epsilon",
    "total_runtime_seconds",
    "number_of_synthetic_rows",
    "compression_ratio",
]


def build_all_tables(
    master: pd.DataFrame,
    tables_dir: Path,
    experiments_dir: Path,
) -> Dict[str, Path]:
    """Generate evaluation summary tables under tables_dir."""
    tables_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}

    full = master.copy()
    p_full = tables_dir / "results_master_full.csv"
    write_master_csv(full, p_full)
    paths["results_master_full.csv"] = p_full

    # Excel if openpyxl/xlsxwriter available
    p_xlsx = tables_dir / "results_master_full.xlsx"
    try:
        full.to_excel(p_xlsx, index=False)
        paths["results_master_full.xlsx"] = p_xlsx
    except Exception:
        pass

    # One row per dataset/method (prefer successful primary runs)
    t_all = full.copy()
    if "status" in t_all.columns:
        t_all = t_all[t_all["status"].fillna("success") != "failed"]
    cols = [c for c in KEY_TABLE_COLS if c in t_all.columns]
    p_all = tables_dir / "table_all_datasets.csv"
    t_all[cols].to_csv(p_all, index=False)
    paths["table_all_datasets"] = p_all

    # Method averages (classification vs regression separately)
    rows_avg = []
    if "task_type" not in full.columns:
        full = full.copy()
        full["task_type"] = "unknown"
    for task, grp in full.groupby("task_type", dropna=False):
        if "method" not in grp.columns:
            continue
        for method, g in grp.groupby("method"):
            if str(method).startswith("unavailable"):
                continue
            rec = {"task_type": task, "method": method, "n_runs": len(g)}
            metric_cols = [
                "mean_tstr_f1",
                "mean_tstr_auc",
                "mean_tstr_r2",
                "wasserstein_mean",
                "pcd",
                "dcr_median",
                "mia_auc",
                "total_runtime_seconds",
            ]
            for m in metric_cols:
                if m not in g.columns:
                    continue
                stats = mean_std_ci(g[m].tolist())
                rec[f"{m}_mean"] = stats["mean"]
                rec[f"{m}_std"] = stats["std"]
            rows_avg.append(rec)
    p_avg = tables_dir / "table_method_averages.csv"
    pd.DataFrame(rows_avg).to_csv(p_avg, index=False)
    paths["table_method_averages"] = p_avg

    # Baseline comparison (dgd vs smote and any available baselines)
    methods = ["dgd_tabpa", "smote", "tvae", "ctab_gan_plus", "tabddpm", "mtabgen"]
    t_base = full[full["method"].isin(methods)] if "method" in full.columns else full
    p_base = tables_dir / "table_baseline_comparison.csv"
    t_base[[c for c in KEY_TABLE_COLS if c in t_base.columns]].to_csv(p_base, index=False)
    paths["table_baseline_comparison"] = p_base

    # Ablations
    if "ablation" in full.columns:
        t_abl = full[full["ablation"].fillna("none") != "none"]
    else:
        t_abl = full.iloc[0:0]
    p_abl = tables_dir / "table_ablation_results.csv"
    t_abl[[c for c in KEY_TABLE_COLS if c in t_abl.columns]].to_csv(p_abl, index=False)
    paths["table_ablation_results"] = p_abl

    # Privacy sweep
    if "epsilon" in full.columns or "target_epsilon" in full.columns:
        mask = full.get("epsilon").notna() if "epsilon" in full.columns else False
        if "run_id" in full.columns:
            mask = mask | full["run_id"].astype(str).str.contains("eps|privacy|noprivacy", case=False, na=False)
        t_priv = full[mask] if isinstance(mask, pd.Series) else full.iloc[0:0]
    else:
        t_priv = full.iloc[0:0]
    p_priv = tables_dir / "table_privacy_sweep.csv"
    t_priv[[c for c in KEY_TABLE_COLS if c in t_priv.columns]].to_csv(p_priv, index=False)
    paths["table_privacy_sweep"] = p_priv

    # Runtime
    rt_cols = [
        c
        for c in [
            "run_id",
            "dataset",
            "method",
            "training_time_seconds",
            "distillation_time_seconds",
            "evaluation_time_seconds",
            "total_runtime_seconds",
            "peak_cpu_ram_mb",
            "peak_gpu_memory_mb",
            "device_name",
            "number_of_trainable_parameters",
        ]
        if c in full.columns
    ]
    p_rt = tables_dir / "table_runtime_results.csv"
    full[rt_cols].to_csv(p_rt, index=False)
    paths["table_runtime_results"] = p_rt

    # Statistical tests (if paired dgd vs smote exists)
    p_stats = tables_dir / "table_statistical_tests.csv"
    tests = _build_statistical_tests(full)
    pd.DataFrame(tests).to_csv(p_stats, index=False)
    paths["table_statistical_tests"] = p_stats

    return paths


def _build_statistical_tests(full: pd.DataFrame) -> List[Dict]:
    tests: List[Dict] = []
    if full.empty or "method" not in full.columns:
        return tests
    for dataset, g in full.groupby("dataset"):
        dgd = g[g["method"] == "dgd_tabpa"]
        smt = g[g["method"] == "smote"]
        for metric in ("mean_tstr_f1", "mean_tstr_r2", "wasserstein_mean", "dcr_median", "mia_auc"):
            if metric not in g.columns:
                continue
            a = dgd[metric].dropna().tolist()
            b = smt[metric].dropna().tolist()
            n = min(len(a), len(b))
            if n < 2:
                # still record single-run note
                if n == 1:
                    tests.append(
                        {
                            "metric": metric,
                            "methods_compared": "dgd_tabpa vs smote",
                            "dataset": dataset,
                            "n_paired": 1,
                            "test_used": "none",
                            "statistic": None,
                            "p_value": None,
                            "note": "need repeated seeds for paired significance testing",
                        }
                    )
                continue
            tests.append(
                paired_comparison(
                    a[:n],
                    b[:n],
                    method_a="dgd_tabpa",
                    method_b="smote",
                    metric=metric,
                    dataset=str(dataset),
                )
            )
    return tests


def plot_method_comparisons(
    master: pd.DataFrame,
    out_dir: Path,
    metrics: Optional[Sequence[str]] = None,
) -> List[Path]:
    """Bar charts comparing methods across datasets for selected metrics."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = list(
        metrics
        or [
            "mean_tstr_f1",
            "mean_tstr_auc",
            "wasserstein_mean",
            "pcd",
            "dcr_median",
            "mia_auc",
            "total_runtime_seconds",
        ]
    )
    paths: List[Path] = []
    df = master[master.get("status", "success").fillna("success") != "failed"].copy()
    if df.empty:
        return paths

    for metric in metrics:
        if metric not in df.columns:
            continue
        pivot = df.pivot_table(
            index="dataset", columns="method", values=metric, aggfunc="mean"
        )
        if pivot.empty:
            continue
        fig, ax = plt.subplots(figsize=(10, 4.5))
        pivot.plot(kind="bar", ax=ax)
        ax.set_ylabel(metric)
        ax.set_title(f"Comparison: {metric} by dataset")
        ax.legend(title="method", fontsize=8)
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        name = {
            "mean_tstr_f1": "comparison_f1_by_dataset.png",
            "mean_tstr_auc": "comparison_auc_by_dataset.png",
            "wasserstein_mean": "comparison_wasserstein_by_dataset.png",
            "pcd": "comparison_pcd_by_dataset.png",
            "dcr_median": "comparison_dcr_by_dataset.png",
            "mia_auc": "comparison_mia_by_dataset.png",
            "total_runtime_seconds": "comparison_runtime_by_dataset.png",
        }.get(metric, f"comparison_{metric}_by_dataset.png")
        out = out_dir / name
        fig.savefig(out, dpi=300)
        plt.close(fig)
        paths.append(out)
    return paths


def plot_privacy_suite_figures(master: pd.DataFrame, dataset: str, out_dir: Path) -> List[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    g = master[master["dataset"] == dataset].copy()
    if g.empty:
        return []
    # Prefer rows with epsilon or eps in run_id
    if "target_epsilon" in g.columns:
        gp = g[g["target_epsilon"].notna() | g["run_id"].astype(str).str.contains("noprivacy|eps", na=False)]
    else:
        gp = g[g["run_id"].astype(str).str.contains("noprivacy|eps", na=False)]
    if gp.empty:
        gp = g
    # Sort by epsilon
    eps = gp.get("target_epsilon", gp.get("epsilon"))
    if eps is not None:
        gp = gp.assign(_eps=eps.fillna(1e6)).sort_values("_eps")
    paths = []
    metric_map = {
        "mean_tstr_f1": "privacy_utility_f1",
        "mean_tstr_auc": "privacy_utility_auc",
        "mean_tstr_r2": "privacy_utility_r2",
        "dcr_median": "privacy_utility_dcr",
        "mia_auc": "privacy_utility_mia",
        "wasserstein_mean": "privacy_utility_fidelity",
    }
    for col, prefix in metric_map.items():
        if col not in gp.columns or gp[col].isna().all():
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        x = gp["_eps"] if "_eps" in gp.columns else range(len(gp))
        ax.plot(x, gp[col], "o-", linewidth=1.8)
        ax.set_xlabel("Privacy budget (target epsilon; 1e6 = no DP)")
        ax.set_ylabel(col)
        ax.set_title(f"Privacy–utility: {col} ({dataset})")
        ax.set_xscale("log")
        fig.tight_layout()
        out = out_dir / f"{prefix}_{dataset}.png"
        fig.savefig(out, dpi=300)
        plt.close(fig)
        paths.append(out)
    return paths


def plot_ablation_figures(master: pd.DataFrame, dataset: str, out_dir: Path) -> List[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    g = master[(master["dataset"] == dataset)].copy()
    if "ablation" not in g.columns:
        return []
    # Include full model (ablation none / empty) + variants
    g["ablation"] = g["ablation"].fillna("none")
    paths = []
    for col, prefix in [
        ("mean_tstr_f1", "ablation_f1"),
        ("mean_tstr_auc", "ablation_auc"),
        ("wasserstein_mean", "ablation_fidelity"),
        ("dcr_median", "ablation_privacy"),
        ("total_runtime_seconds", "ablation_runtime"),
    ]:
        if col not in g.columns or g[col].isna().all():
            continue
        agg = g.groupby("ablation")[col].mean()
        fig, ax = plt.subplots(figsize=(8, 4))
        agg.plot(kind="bar", ax=ax, color="steelblue")
        ax.set_ylabel(col)
        ax.set_title(f"Ablation: {col} ({dataset})")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        out = out_dir / f"{prefix}_{dataset}.png"
        fig.savefig(out, dpi=300)
        plt.close(fig)
        paths.append(out)
    return paths


def generate_evaluation_report(
    master: pd.DataFrame,
    report_dir: Path,
    tables_dir: Path,
    figure_paths: Optional[List[Path]] = None,
    baseline_status: Optional[Dict] = None,
    config_snapshot: Optional[Dict] = None,
) -> Path:
    """Write evidence-driven evaluation_report.md (no unsupported claims)."""
    report_dir.mkdir(parents=True, exist_ok=True)
    report = report_dir / "evaluation_report.md"
    figure_paths = figure_paths or []
    baseline_status = baseline_status or {}

    success = master[master.get("status", "success").fillna("success") == "success"] if not master.empty else master
    failed = master[master.get("status") == "failed"] if "status" in master.columns else pd.DataFrame()
    methods = sorted(success["method"].dropna().unique()) if "method" in success.columns else []
    datasets = sorted(success["dataset"].dropna().unique()) if "dataset" in success.columns else []

    lines = [
        "# Evaluation Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## 1. Experiment configuration",
        "",
        "See per-run `config_snapshot.yaml` under each run directory and the frozen project config.",
        "",
    ]
    if config_snapshot:
        lines.append("```yaml")
        lines.append(json.dumps(config_snapshot, indent=2, default=str)[:4000])
        lines.append("```")
        lines.append("")

    lines += [
        "## 2. Dataset summary",
        "",
        f"- Datasets with at least one successful run: {', '.join(datasets) if datasets else '(none)'}",
        f"- Number of successful result rows: {len(success)}",
        f"- Number of failed result rows: {len(failed)}",
        "",
        "## 3. Methods successfully executed",
        "",
    ]
    for m in methods:
        lines.append(f"- `{m}`")
    lines.append("")
    lines.append("### Baseline availability")
    lines.append("")
    for name, st in sorted(baseline_status.items()):
        avail = st.get("available")
        reason = st.get("reason")
        if avail:
            lines.append(f"- `{name}`: available")
        else:
            lines.append(f"- `{name}`: **unavailable** — {reason}")
    lines.append("")
    lines.append("Methods listed as unavailable were **not evaluated** and have no fabricated metrics.")
    lines.append("")

    lines += [
        "## 4. Utility results",
        "",
        "See `tables/table_all_datasets.csv` and `tables/table_method_averages.csv`.",
        "",
        "Classification metrics (F1 / AUC) and regression metrics (R² / RMSE / MAE) are reported separately.",
        "",
        "## 5. Statistical fidelity results",
        "",
        "Primary fidelity fields: `wasserstein_mean`, `jsd_mean`, `pcd`, `ks_mean`.",
        "",
        "## 6. Privacy results",
        "",
        "Primary privacy fields: `dcr_median`, `dcr_5th_percentile`, `exact_copy_count`, `mia_auc`, `epsilon`.",
        "",
        "Epsilon alone does not imply regulatory compliance; observed leakage metrics are reported neutrally.",
        "",
        "## 7. Runtime and scalability results",
        "",
        "See `tables/table_runtime_results.csv`.",
        "",
        "## 8. Ablation findings",
        "",
        "See `tables/table_ablation_results.csv` and ablation figures under this folder.",
        "",
        "## 9. Privacy–utility findings",
        "",
        "See `tables/table_privacy_sweep.csv` and `privacy_utility_*` figures.",
        "",
        "## 10. Statistical significance findings",
        "",
        "See `tables/table_statistical_tests.csv`.",
        "",
        "p-values below 0.05 indicate a statistically significant difference under the stated test;",
        "they are not interpreted here as proof of superiority.",
        "",
        "## 11. Limitations and failed runs",
        "",
    ]
    if failed.empty:
        lines.append("- No failed runs recorded in the master table.")
    else:
        for _, r in failed.iterrows():
            lines.append(
                f"- `{r.get('run_id')}` dataset={r.get('dataset')} method={r.get('method')}"
            )
    lines.append("")
    lines.append("### Figures")
    lines.append("")
    for fp in figure_paths:
        lines.append(f"- `{fp.as_posix()}`")
    lines.append("")
    lines.append("### Tables")
    lines.append("")
    for p in sorted(tables_dir.glob("*.csv")):
        lines.append(f"- `{p.as_posix()}`")
    lines.append("")

    report.write_text("\n".join(lines), encoding="utf-8")

    # Figure captions
    captions = report_dir / "figure_captions.md"
    cap_lines = ["# Figure captions", ""]
    for i, fp in enumerate(figure_paths, start=1):
        cap_lines += [
            f"## Figure {i}: {fp.name}",
            "",
            f"**File:** `{fp.as_posix()}`",
            "",
            "Objective description: Experimental visualisation generated from recorded run artefacts.",
            "",
            "Measured observations: Refer to the linked plot and the corresponding rows in `results_master.csv`.",
            "",
            "Interpretation is limited to quantities shown in the figure; no unsupported claims are made.",
            "",
        ]
    captions.write_text("\n".join(cap_lines), encoding="utf-8")
    return report


def build_paired_method_figures(
    experiments_dir: Path,
    dataset: str,
    out_dir: Path,
    method_a: str = "dgd_tabpa",
    method_b: str = "smote",
) -> List[Path]:
    """
    Build paired comparison figures from two completed run directories when present.

    Looks for ``{dataset}_dgd`` and ``{dataset}_smote`` (or method-suffixed) artefacts.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates_a = [
        experiments_dir / f"{dataset}_dgd",
        experiments_dir / f"{dataset}_{method_a}",
    ]
    candidates_b = [
        experiments_dir / f"{dataset}_smote",
        experiments_dir / f"{dataset}_{method_b}",
    ]
    dir_a = next((p for p in candidates_a if p.exists()), None)
    dir_b = next((p for p in candidates_b if p.exists()), None)
    if dir_a is None or dir_b is None:
        return []

    paths: List[Path] = []
    pairs = [
        ("correlation", "real_vs_smote_vs_dgd_correlation.png"),
        ("dcr", "smote_vs_dgd_dcr.png"),
        ("manifold", "smote_vs_dgd_manifold.png"),
        ("marginals", "smote_vs_dgd_marginals.png"),
        ("roc", "smote_vs_dgd_roc.png"),
    ]
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    for key, out_name in pairs:
        fa = (
            list((dir_a / "figures").glob(f"{key}_*.png"))
            if (dir_a / "figures").exists()
            else []
        )
        fb = (
            list((dir_b / "figures").glob(f"{key}_*.png"))
            if (dir_b / "figures").exists()
            else []
        )
        if not fa or not fb:
            continue
        try:
            img_a = mpimg.imread(str(fa[0]))
            img_b = mpimg.imread(str(fb[0]))
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes[0].imshow(img_a)
            axes[0].set_title(f"{method_a}")
            axes[0].axis("off")
            axes[1].imshow(img_b)
            axes[1].set_title(f"{method_b}")
            axes[1].axis("off")
            fig.suptitle(f"{dataset}: {key} ({method_a} vs {method_b})")
            fig.tight_layout()
            out = out_dir / out_name
            fig.savefig(out, dpi=300)
            plt.close(fig)
            paths.append(out)
        except Exception:
            continue
    return paths
