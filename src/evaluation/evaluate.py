"""
Module 5 (Part 1): Multi-Dimensional Evaluation Pipeline

Evaluates distilled/synthetic data across three dimensions:
  1. Resemblance: statistical similarity (Wasserstein, KS-test, PCD)
  2. Utility: TSTR (Train Synthetic, Test Real) with XGBoost/CatBoost/MLP
  3. Privacy: Distance to Closest Record (DCR) and leakage sanity checks

Integrates SynthEval when available, with fallback to custom implementations.
"""

import numpy as np
import pandas as pd
import torch
from scipy.stats import ks_2samp, wasserstein_distance
from scipy.spatial.distance import cdist
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from typing import Optional


class Evaluator:
    """
    Orchestrated evaluation pipeline for synthetic tabular data.

    Computes resemblance, utility (TSTR), and privacy metrics
    between real and synthetic datasets.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.results = {}

    def evaluate_all(
        self,
        real_train: pd.DataFrame,
        real_test: pd.DataFrame,
        synthetic: pd.DataFrame,
        target_col: str,
        real_train_labels: np.ndarray = None,
        real_test_labels: np.ndarray = None,
        synthetic_labels: np.ndarray = None,
    ) -> dict:
        """
        Run the complete evaluation pipeline.

        Args:
            real_train: real training data (original feature space)
            real_test: real test data (original feature space)
            synthetic: synthetic/distilled data (original feature space)
            target_col: name of the target column
            real_train_labels: labels for real training data
            real_test_labels: labels for real test data
            synthetic_labels: labels for synthetic data
        Returns:
            dict with resemblance, utility, and privacy metrics
        """
        self.results = {
            "resemblance": self._evaluate_resemblance(real_train, synthetic),
            "utility": self._evaluate_utility(
                real_train, real_test, synthetic, target_col,
                real_train_labels, real_test_labels, synthetic_labels
            ),
            "privacy": self._evaluate_privacy(real_train, synthetic),
        }
        return self.results

    def _evaluate_resemblance(
        self, real: pd.DataFrame, synthetic: pd.DataFrame
    ) -> dict:
        """Compute statistical resemblance metrics."""
        results = {}

        # Per-column Wasserstein distance (numerical columns only)
        num_cols = real.select_dtypes(include=[np.number]).columns
        common_num = [c for c in num_cols if c in synthetic.columns]

        wasserstein_scores = {}
        ks_scores = {}

        for col in common_num:
            real_vals = real[col].dropna().values.astype(float)
            syn_vals = synthetic[col].dropna().values.astype(float)

            if len(real_vals) > 0 and len(syn_vals) > 0:
                wasserstein_scores[col] = float(wasserstein_distance(real_vals, syn_vals))
                ks_stat, ks_pval = ks_2samp(real_vals, syn_vals)
                ks_scores[col] = {"statistic": float(ks_stat), "p_value": float(ks_pval)}

        results["wasserstein_per_column"] = wasserstein_scores
        results["wasserstein_mean"] = float(np.mean(list(wasserstein_scores.values()))) if wasserstein_scores else None
        results["ks_test_per_column"] = ks_scores

        # Pairwise Correlation Difference (PCD)
        if len(common_num) > 1:
            real_corr = real[common_num].corr().values
            syn_corr = synthetic[common_num].corr().values
            pcd = np.abs(real_corr - syn_corr).mean()
            results["pcd"] = float(pcd)
        else:
            results["pcd"] = None

        return results

    def _evaluate_utility(
        self,
        real_train: pd.DataFrame,
        real_test: pd.DataFrame,
        synthetic: pd.DataFrame,
        target_col: str,
        real_train_labels: np.ndarray = None,
        real_test_labels: np.ndarray = None,
        synthetic_labels: np.ndarray = None,
    ) -> dict:
        """
        TSTR (Train on Synthetic, Test on Real) evaluation.
        Also computes TRTR (Train on Real, Test on Real) as baseline.
        """
        results = {}

        # Determine feature columns
        feature_cols = [c for c in synthetic.columns if c != target_col]
        common_features = [c for c in feature_cols if c in real_train.columns and c in real_test.columns]

        if not common_features:
            return {"error": "No common feature columns found"}

        # Prepare numeric feature matrices
        X_real_train = real_train[common_features].select_dtypes(include=[np.number]).values
        X_real_test = real_test[common_features].select_dtypes(include=[np.number]).values
        X_syn = synthetic[common_features].select_dtypes(include=[np.number]).values

        if real_train_labels is None:
            if target_col in real_train.columns:
                y_real_train = real_train[target_col].values
            else:
                return {"error": f"Target column '{target_col}' not found"}
        else:
            y_real_train = real_train_labels

        if real_test_labels is None:
            if target_col in real_test.columns:
                y_real_test = real_test[target_col].values
            else:
                return {"error": f"Target column '{target_col}' not found in test"}
        else:
            y_real_test = real_test_labels

        if synthetic_labels is None:
            if target_col in synthetic.columns:
                y_syn = synthetic[target_col].values
            else:
                return {"error": f"Target column '{target_col}' not in synthetic data"}
        else:
            y_syn = synthetic_labels

        # Handle NaN
        mask_train = ~np.isnan(X_real_train).any(axis=1)
        mask_test = ~np.isnan(X_real_test).any(axis=1)
        mask_syn = ~np.isnan(X_syn).any(axis=1)

        X_real_train = X_real_train[mask_train]
        y_real_train = y_real_train[mask_train]
        X_real_test = X_real_test[mask_test]
        y_real_test = y_real_test[mask_test]
        X_syn = X_syn[mask_syn]
        y_syn = y_syn[mask_syn]

        models = {}

        # XGBoost
        try:
            from xgboost import XGBClassifier
            models["xgboost"] = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=self.random_state,
                eval_metric="logloss",
                use_label_encoder=False,
            )
        except ImportError:
            pass

        # CatBoost
        try:
            from catboost import CatBoostClassifier
            models["catboost"] = CatBoostClassifier(
                iterations=100,
                depth=6,
                random_seed=self.random_state,
                verbose=0,
            )
        except ImportError:
            pass

        # MLP
        models["mlp"] = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=200,
            random_state=self.random_state,
        )

        for name, model in models.items():
            try:
                # TRTR: Train Real, Test Real (baseline)
                model_trtr = _clone_model(model)
                model_trtr.fit(X_real_train, y_real_train)
                y_pred_trtr = model_trtr.predict(X_real_test)

                trtr_f1 = float(f1_score(y_real_test, y_pred_trtr, average="weighted"))
                trtr_acc = float(accuracy_score(y_real_test, y_pred_trtr))

                # TSTR: Train Synthetic, Test Real
                model_tstr = _clone_model(model)
                model_tstr.fit(X_syn, y_syn)
                y_pred_tstr = model_tstr.predict(X_real_test)

                tstr_f1 = float(f1_score(y_real_test, y_pred_tstr, average="weighted"))
                tstr_acc = float(accuracy_score(y_real_test, y_pred_tstr))

                results[name] = {
                    "trtr_f1": trtr_f1,
                    "trtr_accuracy": trtr_acc,
                    "tstr_f1": tstr_f1,
                    "tstr_accuracy": tstr_acc,
                    "f1_gap": round(trtr_f1 - tstr_f1, 4),
                }
            except Exception as e:
                results[name] = {"error": str(e)}

        return results

    def _evaluate_privacy(
        self, real: pd.DataFrame, synthetic: pd.DataFrame
    ) -> dict:
        """
        Compute privacy risk metrics.

        DCR (Distance to Closest Record): Euclidean distance between
        each synthetic record and its nearest real neighbor.
        """
        num_cols = real.select_dtypes(include=[np.number]).columns
        common_num = [c for c in num_cols if c in synthetic.columns]

        if not common_num:
            return {"error": "No common numerical columns for DCR computation"}

        real_arr = real[common_num].dropna().values.astype(float)
        syn_arr = synthetic[common_num].dropna().values.astype(float)

        # Compute pairwise distances
        dists = cdist(syn_arr, real_arr, metric="euclidean")
        dcr_values = dists.min(axis=1)

        # Sanity check: no exact copies
        n_exact_copies = int((dcr_values == 0.0).sum())

        results = {
            "dcr_median": float(np.median(dcr_values)),
            "dcr_mean": float(np.mean(dcr_values)),
            "dcr_min": float(np.min(dcr_values)),
            "dcr_5th_percentile": float(np.percentile(dcr_values, 5)),
            "n_exact_copies": n_exact_copies,
            "sanity_check_passed": n_exact_copies == 0,
        }

        # Privacy rating based on DCR distribution
        if results["dcr_5th_percentile"] > 0.5:
            results["privacy_rating"] = "Excellent"
        elif results["dcr_5th_percentile"] > 0.1:
            results["privacy_rating"] = "Good"
        elif results["dcr_5th_percentile"] > 0.01:
            results["privacy_rating"] = "Moderate"
        else:
            results["privacy_rating"] = "Poor"

        return results

    def evaluate_with_syntheval(
        self,
        real_df: pd.DataFrame,
        synthetic_df: pd.DataFrame,
        target_col: str,
        cat_cols: list = None,
    ) -> dict:
        """
        Run SynthEval evaluation if the library is available.
        Falls back to custom metrics if not installed.
        """
        try:
            from syntheval import SynthEval

            evaluator = SynthEval(real_df, target=target_col, cat_cols=cat_cols)
            results = evaluator.evaluate(synthetic_df)
            return {"syntheval": results, "source": "syntheval_library"}
        except ImportError:
            print("[Evaluation] SynthEval not installed. Using custom metrics.")
            return {"source": "custom", "note": "Install syntheval for full evaluation"}
        except Exception as e:
            print(f"[Evaluation] SynthEval error: {e}. Using custom metrics.")
            return {"source": "custom", "error": str(e)}

    def print_report(self):
        """Print a formatted evaluation report."""
        if not self.results:
            print("No results available. Run evaluate_all() first.")
            return

        print("\n" + "=" * 60)
        print("  DGD-TabPA EVALUATION REPORT")
        print("=" * 60)

        # Resemblance
        res = self.results.get("resemblance", {})
        print("\n--- RESEMBLANCE (Statistical Similarity) ---")
        if res.get("wasserstein_mean") is not None:
            print(f"  Mean Wasserstein Distance: {res['wasserstein_mean']:.4f}")
        if res.get("pcd") is not None:
            print(f"  Pairwise Correlation Diff: {res['pcd']:.4f}")

        # Utility
        util = self.results.get("utility", {})
        print("\n--- UTILITY (Train Synthetic, Test Real) ---")
        for model_name, metrics in util.items():
            if isinstance(metrics, dict) and "tstr_f1" in metrics:
                print(f"  [{model_name}]")
                print(f"    TRTR F1: {metrics['trtr_f1']:.4f} (baseline)")
                print(f"    TSTR F1: {metrics['tstr_f1']:.4f}")
                print(f"    F1 Gap:  {metrics['f1_gap']:.4f}")

        # Privacy
        priv = self.results.get("privacy", {})
        print("\n--- PRIVACY (Distance to Closest Record) ---")
        if "dcr_median" in priv:
            print(f"  DCR Median: {priv['dcr_median']:.4f}")
            print(f"  DCR 5th Percentile: {priv['dcr_5th_percentile']:.4f}")
            print(f"  Exact Copies: {priv['n_exact_copies']}")
            print(f"  Sanity Check: {'PASSED' if priv['sanity_check_passed'] else 'FAILED'}")
            print(f"  Privacy Rating: {priv['privacy_rating']}")

        print("\n" + "=" * 60)


def _clone_model(model):
    """Create a fresh clone of a sklearn-compatible model."""
    from sklearn.base import clone
    try:
        return clone(model)
    except Exception:
        return model.__class__(**model.get_params())
