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
import time
from pathlib import Path
from typing import List, Sequence

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

# Requested variants that are not cleanly separable in the current architecture
# (documented rather than simulated):
UNSUPPORTED_ABLATIONS = {
    "no_dynamic_masking": (
        "Training uses a fixed --mask-ratio; there is no separate dynamic-masking "
        "module to disable independently without changing the diffusion objective."
    ),
    "no_dp_sgd": (
        "Covered by the default non-private run (privacy suite noprivacy / privacy.enabled=false)."
    ),
    "no_fast_sampler": (
        "Sampling steps are controlled by diffusion.sampling_steps; a distinct "
        "fast-sampler abstraction is not present as a toggleable component."
    ),
    "full_model": "Equivalent to ablation=none (default dgd_tabpa run).",
    "no_transformer_or_mlp_denoiser": "Use --ablation mlp_denoiser (MLP instead of Transformer).",
    "no_conditioning_attention": "Use --ablation no_attention.",
    "no_gaussian_quantile": "Use --ablation minmax (MinMax instead of quantile-normal).",
}


def run_one(cmd: list) -> int:
    print("\n" + "=" * 72)
    print(" ".join(cmd))
    print("=" * 72, flush=True)
    result = subprocess.run(cmd, cwd=str(ROOT))
    return result.returncode


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def print_suite_progress(
    *,
    index: int,
    total: int,
    current_label: str,
    elapsed_times: Sequence[float],
    suite_started: float,
) -> None:
    """Print current item, percent complete, and ETA for a batch suite."""
    pct = 100.0 * (index - 1) / total if total else 0.0
    done = index - 1
    remaining = total - done
    if elapsed_times:
        avg = sum(elapsed_times) / len(elapsed_times)
        eta = avg * remaining
        avg_txt = _format_duration(avg)
        eta_txt = _format_duration(eta)
    else:
        avg_txt = "n/a"
        eta_txt = "estimating..."
    elapsed_suite = time.perf_counter() - suite_started
    bar_len = 20
    filled = int(bar_len * (index - 1) / total) if total else 0
    bar = "#" * filled + "-" * (bar_len - filled)
    print("\n" + "-" * 72, flush=True)
    print(
        f"Progress [{bar}] {pct:5.1f}%  "
        f"({done}/{total} finished, starting {index}/{total})",
        flush=True,
    )
    print(f"Current:  {current_label}", flush=True)
    print(
        f"Elapsed:  {_format_duration(elapsed_suite)}  |  "
        f"Avg/run: {avg_txt}  |  ETA remaining: {eta_txt}",
        flush=True,
    )
    print("-" * 72, flush=True)


def run_jobs_with_progress(
    jobs: List[tuple],
    record,
) -> None:
    """
    Run (tag, cmd) jobs with live progress.

    ``record`` is called as ``record(cmd, returncode, tag)``.
    """
    total = len(jobs)
    if total == 0:
        return
    suite_started = time.perf_counter()
    elapsed_times: List[float] = []
    for i, (tag, cmd) in enumerate(jobs, start=1):
        print_suite_progress(
            index=i,
            total=total,
            current_label=tag,
            elapsed_times=elapsed_times,
            suite_started=suite_started,
        )
        t0 = time.perf_counter()
        code = run_one(cmd)
        dt = time.perf_counter() - t0
        elapsed_times.append(dt)
        record(cmd, code, tag)
        print(
            f"Finished {tag} in {_format_duration(dt)} "
            f"[{i}/{total} = {100.0 * i / total:.1f}%]",
            flush=True,
        )
        remaining = total - i
        if remaining and elapsed_times:
            eta = (sum(elapsed_times) / len(elapsed_times)) * remaining
            print(f"ETA for remaining {remaining} run(s): {_format_duration(eta)}", flush=True)


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
            "seeds",
        ],
        help="all_ten = train/eval on all 10 datasets; core = 8 classification; "
        "seeds = repeated-seed runs for primary dataset; "
        "all = all_ten DGD + smote + privacy + ablations",
    )
    parser.add_argument("--dataset", type=str, default="diabetes")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--distill-epochs", type=int, default=None)
    parser.add_argument("--datasets", type=str, nargs="*", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        default=True,
        help="Continue batch after a failed dataset (default True)",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop batch on first failure",
    )
    args = parser.parse_args()

    import yaml

    with open(ROOT / args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    privacy_eps = cfg.get("experiments", {}).get(
        "privacy_epsilon_grid", PRIVACY_EPSILONS
    )
    seed_list = cfg.get("experiments", {}).get("seeds", [42, 52, 62, 72, 82])

    py = sys.executable
    script = str(ROOT / "scripts" / "run_experiment.py")
    base = [py, script, "--config", args.config, "--epochs", str(args.epochs)]
    if args.distill_epochs is not None:
        base += ["--distill-epochs", str(args.distill_epochs)]
    if args.force:
        base += ["--force"]

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
    successes, skipped, failures = [], [], []
    failed_rows = []

    def record(cmd, code, tag):
        codes.append(code)
        if code == 0:
            successes.append(tag)
        else:
            failures.append(tag)
            failed_rows.append({"run_tag": tag, "exit_code": code, "cmd": " ".join(cmd)})
            if args.fail_fast:
                raise SystemExit(1)

    try:
        if args.suite in ("all_ten", "core", "core_fast", "all"):
            print(f"\nRunning DGD on {len(datasets)} datasets: {datasets}", flush=True)
            jobs = [
                (
                    f"{ds}_dgd",
                    base + ["--dataset", ds, "--method", "dgd_tabpa", "--run-id", f"{ds}_dgd"],
                )
                for ds in datasets
            ]
            run_jobs_with_progress(jobs, record)

        if args.suite in ("smote", "all"):
            jobs = [
                (
                    f"{ds}_smote",
                    base + ["--dataset", ds, "--method", "smote", "--run-id", f"{ds}_smote"],
                )
                for ds in datasets
            ]
            run_jobs_with_progress(jobs, record)

        if args.suite in ("privacy", "all"):
            ds = args.dataset
            jobs = [
                (
                    f"{ds}_noprivacy",
                    base + ["--dataset", ds, "--method", "dgd_tabpa", "--run-id", f"{ds}_noprivacy"],
                )
            ]
            for eps in privacy_eps:
                jobs.append(
                    (
                        f"{ds}_eps{eps}",
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
                        ],
                    )
                )
            run_jobs_with_progress(jobs, record)

        if args.suite in ("ablations", "all"):
            ds = args.dataset
            print("\nSupported ablations:", [a for a in ABLATIONS if a != "none"], flush=True)
            print("Unsupported / aliased ablations (not simulated):", flush=True)
            for k, v in UNSUPPORTED_ABLATIONS.items():
                print(f"  - {k}: {v}", flush=True)
            jobs = []
            for abl in ABLATIONS:
                if abl == "none":
                    continue
                jobs.append(
                    (
                        f"{ds}_{abl}",
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
                        ],
                    )
                )
            run_jobs_with_progress(jobs, record)

        if args.suite == "seeds":
            ds = args.dataset
            jobs = []
            for seed in seed_list:
                for method in ("dgd_tabpa", "smote"):
                    tag = f"{ds}_{method}_seed{seed}"
                    jobs.append(
                        (
                            tag,
                            base
                            + [
                                "--dataset",
                                ds,
                                "--method",
                                method,
                                "--seed",
                                str(seed),
                                "--run-id",
                                tag,
                                "--force",
                            ],
                        )
                    )
            run_jobs_with_progress(jobs, record)

    except SystemExit:
        pass

    # Write failed_runs.csv
    import csv as csvmod

    failed_path = ROOT / "outputs" / "experiments" / "failed_runs.csv"
    failed_path.parent.mkdir(parents=True, exist_ok=True)
    with open(failed_path, "w", newline="", encoding="utf-8") as f:
        w = csvmod.DictWriter(f, fieldnames=["run_tag", "exit_code", "cmd"])
        w.writeheader()
        for row in failed_rows:
            w.writerow(row)

    print("\n=== Batch summary ===")
    print(f"Successful: {len(successes)}")
    print(f"Failed:     {len(failures)}")
    for t in failures:
        print(f"  FAIL {t}")
    print(f"Failed log: {failed_path}")
    print(f"Master CSV: {ROOT / 'outputs' / 'experiments' / 'results_master.csv'}")

    if args.suite in ("privacy", "all"):
        _try_privacy_plot(args.dataset)

    if failures:
        sys.exit(1)


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
