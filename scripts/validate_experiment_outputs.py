"""
Validate experiment outputs for completeness and schema consistency.

Usage:
    python scripts/validate_experiment_outputs.py
    python scripts/validate_experiment_outputs.py --require-all-ten
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.reporting import load_master_csv
from src.evaluation.schema import RESULT_SCHEMA, validate_row

ALL_TEN = [
    "adult",
    "churn",
    "credit",
    "covertype",
    "cvd",
    "hcv",
    "ilpd",
    "diabetes",
    "california_housing",
    "king_county",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate experiment outputs")
    parser.add_argument("--experiments-dir", type=str, default="outputs/experiments")
    parser.add_argument("--require-all-ten", action="store_true")
    args = parser.parse_args()

    exp = Path(args.experiments_dir)
    master_path = exp / "results_master.csv"
    errors = []
    warnings = []

    if not master_path.exists():
        print(f"CRITICAL: missing {master_path}")
        return 2

    df = load_master_csv(master_path)
    print(f"Master rows: {len(df)}")

    missing_cols = [c for c in RESULT_SCHEMA if c not in df.columns]
    if missing_cols:
        warnings.append(f"schema columns missing from CSV (will be null): {missing_cols[:10]}...")

    if "run_id" in df.columns:
        dups = df["run_id"].dropna()[df["run_id"].duplicated()].unique().tolist()
        if dups:
            errors.append(f"duplicate run_ids: {dups[:5]}")

    for _, row in df.iterrows():
        errs = validate_row(row.to_dict())
        for e in errs:
            errors.append(f"{row.get('run_id')}: {e}")

        if row.get("status") == "success" and row.get("exact_copy_count") not in (None, 0):
            warnings.append(
                f"{row.get('run_id')}: exact_copy_count={row.get('exact_copy_count')} "
                "(not ignored — flagged for privacy discussion)"
            )

    # Per-run directories
    for run_dir in sorted([p for p in exp.iterdir() if p.is_dir()]):
        if run_dir.name in ("tables", "report"):
            continue
        for req in ("summary_row.json", "status.json"):
            if not (run_dir / req).exists():
                # only warn if metrics exist (partial run)
                if (run_dir / "metrics.json").exists():
                    warnings.append(f"{run_dir.name}: missing {req}")
        fig_dir = run_dir / "figures"
        if fig_dir.exists():
            for png in fig_dir.glob("*.png"):
                if png.stat().st_size < 100:
                    warnings.append(f"tiny figure: {png}")

    if args.require_all_ten and "dataset" in df.columns:
        present = set(df["dataset"].dropna().unique())
        missing = [d for d in ALL_TEN if d not in present]
        if missing:
            errors.append(f"missing datasets for all_ten: {missing}")

    failed_csv = exp / "failed_runs.csv"
    if failed_csv.exists():
        print(f"Failed runs file present: {failed_csv}")

    print("\n=== Validation summary ===")
    print(f"Errors: {len(errors)}")
    for e in errors[:30]:
        print("  ERROR:", e)
    print(f"Warnings: {len(warnings)}")
    for w in warnings[:30]:
        print("  WARN:", w)

    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
