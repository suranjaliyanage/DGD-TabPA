"""
Statistical helpers for repeated-seed evaluation and paired method comparisons.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def mean_std_ci(
    values: Sequence[float], alpha: float = 0.05
) -> Dict[str, Optional[float]]:
    """Mean, std, and approximate 95% CI (normal) for a list of values."""
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if len(arr) == 0:
        return {"mean": None, "std": None, "ci_low": None, "ci_high": None, "n": 0}
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    if len(arr) > 1:
        # t critical approx via normal for simplicity when scipy unavailable
        try:
            from scipy import stats

            tcrit = float(stats.t.ppf(1 - alpha / 2, df=len(arr) - 1))
        except Exception:
            tcrit = 1.96
        se = std / np.sqrt(len(arr))
        ci_low, ci_high = mean - tcrit * se, mean + tcrit * se
    else:
        ci_low = ci_high = mean
    return {
        "mean": mean,
        "std": std,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "n": int(len(arr)),
    }


def paired_comparison(
    a: Sequence[float],
    b: Sequence[float],
    method_a: str = "a",
    method_b: str = "b",
    metric: str = "metric",
    dataset: str = "pooled",
) -> Dict:
    """
    Paired comparison of two methods on identical seeds.
    Uses Wilcoxon signed-rank when available; falls back to paired t-test.
    """
    pairs = [
        (x, y)
        for x, y in zip(a, b)
        if x is not None and y is not None and np.isfinite(x) and np.isfinite(y)
    ]
    result = {
        "metric": metric,
        "method_a": method_a,
        "method_b": method_b,
        "dataset": dataset,
        "n_paired": len(pairs),
        "test_used": None,
        "statistic": None,
        "p_value": None,
        "mean_diff": None,
        "effect_size": None,
        "note": "p < 0.05 indicates a statistically significant difference, not proof of superiority",
    }
    if len(pairs) < 2:
        result["note"] = "insufficient paired observations"
        return result

    xa = np.array([p[0] for p in pairs], dtype=float)
    xb = np.array([p[1] for p in pairs], dtype=float)
    diff = xa - xb
    result["mean_diff"] = float(diff.mean())
    # Cohen's d for paired differences
    if diff.std(ddof=1) > 0:
        result["effect_size"] = float(diff.mean() / diff.std(ddof=1))

    try:
        from scipy import stats

        # Prefer Wilcoxon when n is small or non-normal
        try:
            stat, p = stats.wilcoxon(xa, xb)
            result["test_used"] = "wilcoxon_signed_rank"
            result["statistic"] = float(stat)
            result["p_value"] = float(p)
        except ValueError:
            stat, p = stats.ttest_rel(xa, xb)
            result["test_used"] = "paired_t_test"
            result["statistic"] = float(stat)
            result["p_value"] = float(p)
    except Exception as e:
        result["note"] = f"stats unavailable: {e}"
    return result
