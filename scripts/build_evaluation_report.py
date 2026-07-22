"""
Build evaluation tables, aggregated figures, and Markdown report.

Usage:
    python scripts/build_evaluation_report.py
    python scripts/build_evaluation_report.py --experiments-dir outputs/experiments --dataset diabetes
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baselines.registry import list_baseline_status
from src.evaluation.reporting import (
    build_all_tables,
    build_paired_method_figures,
    generate_evaluation_report,
    load_master_csv,
    plot_ablation_figures,
    plot_method_comparisons,
    plot_privacy_suite_figures,
)


def main():
    parser = argparse.ArgumentParser(description="Build evaluation report artefacts")
    parser.add_argument(
        "--experiments-dir",
        type=str,
        default="outputs/experiments",
    )
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Primary dataset for privacy/ablation aggregated figures",
    )
    args = parser.parse_args()

    exp_dir = Path(args.experiments_dir)
    master_path = exp_dir / "results_master.csv"
    tables_dir = exp_dir / "tables"
    report_dir = exp_dir / "report"

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    reporting_cfg = cfg.get("reporting", {})
    primary = args.dataset or reporting_cfg.get("primary_dataset", "diabetes")

    master = load_master_csv(master_path)
    print(f"Loaded master rows: {len(master)} from {master_path}")

    table_paths = build_all_tables(master, tables_dir, exp_dir)
    print(f"Wrote {len(table_paths)} tables under {tables_dir}")

    fig_paths = []
    fig_paths += plot_method_comparisons(master, report_dir)
    fig_paths += plot_privacy_suite_figures(master, primary, report_dir)
    fig_paths += plot_ablation_figures(master, primary, report_dir)
    fig_paths += build_paired_method_figures(exp_dir, primary, report_dir)

    report = generate_evaluation_report(
        master=master,
        report_dir=report_dir,
        tables_dir=tables_dir,
        figure_paths=fig_paths,
        baseline_status=list_baseline_status(),
        config_snapshot={
            "reporting": reporting_cfg,
            "experiments": cfg.get("experiments", {}),
            "privacy": cfg.get("privacy", {}),
        },
    )
    print(f"Report: {report}")
    print(f"Captions: {report_dir / 'figure_captions.md'}")
    print(f"Aggregated figures: {len(fig_paths)}")


if __name__ == "__main__":
    main()
