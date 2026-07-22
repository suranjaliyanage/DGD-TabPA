"""
Batch runner for multi-dataset / ablation / privacy sweeps.

Examples:
    # All 10 benchmark datasets
    python scripts/run_batch_experiments.py --suite all_ten --epochs 30

    # Classification-only core table
    python scripts/run_batch_experiments.py --suite core --epochs 20

    # Privacy–utility sweep on one dataset
    python scripts/run_batch_experiments.py --suite privacy --dataset diabetes --epochs 15

    # Ablation suite on one dataset
    python scripts/run_batch_experiments.py --suite ablations --dataset diabetes --epochs 15

    # SMOTE baseline across datasets
    python scripts/run_batch_experiments.py --suite smote
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# All 10 benchmarks from config/default.yaml
ALL_TEN_DATASETS = [
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

CLASSIFICATION_DATASETS = [
    "adult",
    "churn",
    "credit",
    "covertype",
    "cvd",
    "hcv",
    "ilpd",
    "diabetes",
]

REGRESSION_DATASETS = [
    "california_housing",
    "king_county",
]

# Smaller / faster set for quick evaluation iterations
CORE_FAST = ["diabetes", "ilpd", "churn", "adult"]

PRIVACY_EPSILONS = [1.0, 4.0, 8.0, 100.0]

ABLATIONS = ["none", "mlp_denoiser", "no_attention", "minmax", "raw_space"]


def run_one(cmd: list) -> int:
    print("\n" + "=" * 72)
    print(" ".join(cmd))
    print("=" * 72)
    result = subprocess.run(cmd, cwd=str(ROOT))
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Batch DGD-TabPA experiments")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument(
        "--suite",
        type=str,
        required=True,
        choices=[
            "all_ten",
            "core",
            "core_fast",
            "privacy",
            "ablations",
            "smote",
            "all",
        ],
        help="all_ten = train/eval on all 10 datasets; core = 8 classification; "
        "all = all_ten DGD + smote + privacy + ablations",
    )
    parser.add_argument("--dataset", type=str, default="diabetes")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--distill-epochs", type=int, default=None)
    parser.add_argument("--datasets", type=str, nargs="*", default=None)
    args = parser.parse_args()

    py = sys.executable
    script = str(ROOT / "scripts" / "run_experiment.py")
    base = [py, script, "--config", args.config, "--epochs", str(args.epochs)]
    if args.distill_epochs is not None:
        base += ["--distill-epochs", str(args.distill_epochs)]

    datasets = args.datasets
    if datasets is None:
        if args.suite in ("all_ten", "all"):
            datasets = ALL_TEN_DATASETS
        elif args.suite == "core":
            datasets = CLASSIFICATION_DATASETS
        elif args.suite == "core_fast":
            datasets = CORE_FAST
        elif args.suite == "smote":
            datasets = ALL_TEN_DATASETS
        else:
            datasets = [args.dataset]

    codes = []

    if args.suite in ("all_ten", "core", "core_fast", "all"):
        print(f"\nRunning DGD on {len(datasets)} datasets: {datasets}")
        for ds in datasets:
            codes.append(
                run_one(
                    base
                    + ["--dataset", ds, "--method", "dgd_tabpa", "--run-id", f"{ds}_dgd"]
                )
            )

    if args.suite in ("smote", "all"):
        for ds in datasets:
            codes.append(
                run_one(
                    base
                    + [
                        "--dataset",
                        ds,
                        "--method",
                        "smote",
                        "--run-id",
                        f"{ds}_smote",
                    ]
                )
            )

    if args.suite in ("privacy", "all"):
        ds = args.dataset
        codes.append(
            run_one(
                base
                + [
                    "--dataset",
                    ds,
                    "--method",
                    "dgd_tabpa",
                    "--run-id",
                    f"{ds}_noprivacy",
                ]
            )
        )
        for eps in PRIVACY_EPSILONS:
            codes.append(
                run_one(
                    base
                    + [
                        "--dataset",
                        ds,
                        "--method",
                        "dgd_tabpa",
                        "--privacy",
                        "--epsilon",
                        str(eps),
                        "--run-id",
                        f"{ds}_eps{eps}",
                    ]
                )
            )

    if args.suite in ("ablations", "all"):
        ds = args.dataset
        for abl in ABLATIONS:
            if abl == "none":
                continue
            codes.append(
                run_one(
                    base
                    + [
                        "--dataset",
                        ds,
                        "--method",
                        "dgd_tabpa",
                        "--ablation",
                        abl,
                        "--run-id",
                        f"{ds}_{abl}",
                    ]
                )
            )

    failed = sum(1 for c in codes if c != 0)
    print(f"\nBatch finished: {len(codes) - failed}/{len(codes)} succeeded.")
    if failed:
        sys.exit(1)

    if args.suite in ("privacy", "all"):
        _try_privacy_plot(args.dataset)


def _try_privacy_plot(dataset: str):
    try:
        import pandas as pd

        from src.evaluation.plotting import plot_privacy_utility

        master = ROOT / "outputs" / "experiments" / "results_master.csv"
        if not master.exists():
            return
        df = pd.read_csv(master)
        df = df[df["dataset"] == dataset]
        points = []
        for _, row in df.iterrows():
            method = str(row.get("method", ""))
            eps = row.get("epsilon")
            if pd.isna(eps):
                if "noprivacy" in method or method == "dgd_tabpa":
                    eps = 1e6
                else:
                    continue
            f1 = row.get("mean_tstr_f1")
            if pd.isna(f1):
                f1 = row.get("mean_tstr_r2")
            dcr = row.get("dcr_median")
            if pd.isna(f1):
                continue
            points.append(
                {
                    "epsilon": float(eps) if float(eps) < 1e5 else 1000.0,
                    "tstr_f1": float(f1),
                    "dcr_median": float(dcr) if not pd.isna(dcr) else None,
                    "label": "no-DP" if float(eps) >= 1e5 else f"eps={eps}",
                }
            )
        if points:
            out = (
                ROOT
                / "outputs"
                / "experiments"
                / f"privacy_utility_{dataset}.png"
            )
            plot_privacy_utility(
                points,
                out,
                title=f"Privacy-Utility Trade-off ({dataset})",
            )
            print(f"Privacy-utility figure: {out}")
    except Exception as e:
        print(f"[warn] Could not build privacy plot: {e}")


if __name__ == "__main__":
    main()
