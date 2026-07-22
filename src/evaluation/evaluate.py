"""
Module 5 (Part 1): Multi-Dimensional Evaluation Pipeline

Evaluates distilled/synthetic data across three dimensions:
  1. Resemblance: statistical similarity (Wasserstein, KS-test, PCD, JSD)
  2. Utility: TSTR (Train Synthetic, Test Real) with XGBoost/CatBoost/MLP + ROC-AUC
  3. Privacy: Distance to Closest Record (DCR) and distance-based Membership Inference (MIA)

Integrates SynthEval when available, with fallback to custom implementations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, jensenshannon
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor


class Evaluator:
    """
    Orchestrated evaluation pipeline for synthetic tabular data.

    Computes resemblance, utility (TSTR), and privacy metrics
    between real and synthetic datasets.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.results = {}
        self._roc_curves = {}

    def evaluate_all(
        self,
        real_train: pd.DataFrame,
        real_test: pd.DataFrame,
        synthetic: pd.DataFrame,
        target_col: str,
        real_train_labels: np.ndarray = None,
        real_test_labels: np.ndarray = None,
        synthetic_labels: np.ndarray = None,
        cat_cols: list = None,
        task: str = "classification",
    ) -> dict:
        """
        Run the complete evaluation pipeline.

        Returns:
            dict with resemblance, utility, and privacy metrics
        """
        self.results = {
            "task": task,
            "resemblance": self._evaluate_resemblance(
                real_train, synthetic, cat_cols=cat_cols
            ),
            "utility": self._evaluate_utility(
                real_train,
                real_test,
                synthetic,
                target_col,
                real_train_labels,
                real_test_labels,
                synthetic_labels,
                task=task,
            ),
            "privacy": self._evaluate_privacy(
                real_train,
                real_test,
                synthetic,
                real_train_labels=real_train_labels,
                real_test_labels=real_test_labels,
            ),
        }
        return self.results

    def _evaluate_resemblance(
        self,
        real: pd.DataFrame,
        synthetic: pd.DataFrame,
        cat_cols: list = None,
    ) -> dict:
        """Compute statistical resemblance metrics."""
        results = {}

        num_cols = real.select_dtypes(include=[np.number]).columns
        common_num = [c for c in num_cols if c in synthetic.columns]

        wasserstein_scores = {}
        ks_scores = {}

        for col in common_num:
            real_vals = real[col].dropna().values.astype(float)
            syn_vals = synthetic[col].dropna().values.astype(float)

            if len(real_vals) > 0 and len(syn_vals) > 0:
                wasserstein_scores[col] = float(
                    wasserstein_distance(real_vals, syn_vals)
                )
                ks_stat, ks_pval = ks_2samp(real_vals, syn_vals)
                ks_scores[col] = {
                    "statistic": float(ks_stat),
                    "p_value": float(ks_pval),
                }

        results["wasserstein_per_column"] = wasserstein_scores
        results["wasserstein_mean"] = (
            float(np.mean(list(wasserstein_scores.values())))
            if wasserstein_scores
            else None
        )
        results["wasserstein_std"] = (
            float(np.std(list(wasserstein_scores.values())))
            if wasserstein_scores
            else None
        )
        results["ks_test_per_column"] = ks_scores
        results["ks_mean"] = (
            float(np.mean([v["statistic"] for v in ks_scores.values()]))
            if ks_scores
            else None
        )

        # Pairwise Correlation Difference (PCD)
        if len(common_num) > 1:
            real_corr = real[common_num].corr().values
            syn_corr = synthetic[common_num].corr().values
            # Guard NaNs from constant columns
            real_corr = np.nan_to_num(real_corr, nan=0.0)
            syn_corr = np.nan_to_num(syn_corr, nan=0.0)
            pcd = np.abs(real_corr - syn_corr).mean()
            results["pcd"] = float(pcd)
            results["correlation_frobenius_norm"] = float(
                np.linalg.norm(real_corr - syn_corr, ord="fro")
            )
            results["real_corr"] = real_corr.tolist()
            results["syn_corr"] = syn_corr.tolist()
            results["corr_columns"] = list(common_num)
        else:
            results["pcd"] = None
            results["correlation_frobenius_norm"] = None

        # Jensen–Shannon Distance for categorical columns
        if cat_cols is None:
            cat_cols = [
                c
                for c in real.columns
                if c in synthetic.columns
                and (
                    real[c].dtype == object
                    or str(real[c].dtype) == "category"
                    or real[c].nunique() <= 20
                )
                and c not in common_num
            ]
            # Also include low-cardinality numerics that look categorical
            cat_cols = list(
                dict.fromkeys(
                    list(cat_cols)
                    + [
                        c
                        for c in real.columns
                        if c in synthetic.columns
                        and c not in common_num
                        and real[c].dtype == object
                    ]
                )
            )

        jsd_scores = {}
        for col in cat_cols:
            if col not in real.columns or col not in synthetic.columns:
                continue
            jsd = _jsd_categorical(real[col], synthetic[col])
            if jsd is not None:
                jsd_scores[col] = jsd

        results["jsd_per_column"] = jsd_scores
        results["jsd_mean"] = (
            float(np.mean(list(jsd_scores.values()))) if jsd_scores else None
        )
        results["jsd_std"] = (
            float(np.std(list(jsd_scores.values()))) if jsd_scores else None
        )

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
        task: str = "classification",
    ) -> dict:
        """
        TSTR evaluation for classification (F1/AUC) or regression (R2/MAE/RMSE).
        """
        if task == "regression":
            return self._evaluate_utility_regression(
                real_train,
                real_test,
                synthetic,
                target_col,
                real_train_labels,
                real_test_labels,
                synthetic_labels,
            )

        results = {}
        self._roc_curves = {}

        feature_cols = [c for c in synthetic.columns if c != target_col]
        common_features = [
            c
            for c in feature_cols
            if c in real_train.columns and c in real_test.columns
        ]

        if not common_features:
            return {"error": "No common feature columns found"}

        X_real_train = (
            real_train[common_features].select_dtypes(include=[np.number]).values
        )
        X_real_test = (
            real_test[common_features].select_dtypes(include=[np.number]).values
        )
        X_syn = synthetic[common_features].select_dtypes(include=[np.number]).values
        used_features = list(
            real_train[common_features].select_dtypes(include=[np.number]).columns
        )

        if X_real_train.size == 0 or X_syn.size == 0:
            return {"error": "No numeric feature columns for TSTR"}

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

        y_real_train = np.asarray(y_real_train)
        y_real_test = np.asarray(y_real_test)
        y_syn = np.asarray(y_syn)

        mask_train = ~np.isnan(X_real_train).any(axis=1)
        mask_test = ~np.isnan(X_real_test).any(axis=1)
        mask_syn = ~np.isnan(X_syn).any(axis=1)

        X_real_train = X_real_train[mask_train]
        y_real_train = y_real_train[mask_train]
        X_real_test = X_real_test[mask_test]
        y_real_test = y_real_test[mask_test]
        X_syn = X_syn[mask_syn]
        y_syn = y_syn[mask_syn]

        # Align label spaces via string cast then factorize jointly where needed
        y_real_train, y_real_test, y_syn = _align_labels(
            y_real_train, y_real_test, y_syn
        )

        models = {}

        try:
            from xgboost import XGBClassifier

            models["xgboost"] = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=self.random_state,
                eval_metric="logloss",
            )
        except ImportError:
            pass

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

        models["mlp"] = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=200,
            random_state=self.random_state,
        )

        n_classes = len(np.unique(np.concatenate([y_real_train, y_real_test, y_syn])))

        for name, model in models.items():
            try:
                model_trtr = _clone_model(model)
                model_trtr.fit(X_real_train, y_real_train)
                y_pred_trtr = model_trtr.predict(X_real_test)

                trtr_f1 = float(
                    f1_score(y_real_test, y_pred_trtr, average="weighted")
                )
                trtr_acc = float(accuracy_score(y_real_test, y_pred_trtr))
                trtr_auc = _safe_auc(model_trtr, X_real_test, y_real_test, n_classes)

                model_tstr = _clone_model(model)
                model_tstr.fit(X_syn, y_syn)
                y_pred_tstr = model_tstr.predict(X_real_test)

                tstr_f1 = float(
                    f1_score(y_real_test, y_pred_tstr, average="weighted")
                )
                tstr_acc = float(accuracy_score(y_real_test, y_pred_tstr))
                tstr_auc = _safe_auc(model_tstr, X_real_test, y_real_test, n_classes)

                results[name] = {
                    "trtr_f1": trtr_f1,
                    "trtr_accuracy": trtr_acc,
                    "trtr_auc": trtr_auc,
                    "tstr_f1": tstr_f1,
                    "tstr_accuracy": tstr_acc,
                    "tstr_auc": tstr_auc,
                    "f1_gap": round(trtr_f1 - tstr_f1, 4),
                    "tstr_balanced_accuracy": float(
                        balanced_accuracy_score(y_real_test, y_pred_tstr)
                    ),
                    "tstr_precision_macro": float(
                        precision_score(
                            y_real_test, y_pred_tstr, average="macro", zero_division=0
                        )
                    ),
                    "tstr_recall_macro": float(
                        recall_score(
                            y_real_test, y_pred_tstr, average="macro", zero_division=0
                        )
                    ),
                    "tstr_f1_macro": float(
                        f1_score(y_real_test, y_pred_tstr, average="macro")
                    ),
                    "tstr_pr_auc": _safe_pr_auc(
                        model_tstr, X_real_test, y_real_test, n_classes
                    ),
                }

                if not hasattr(self, "_confusion_data"):
                    self._confusion_data = {}
                self._confusion_data[name] = {
                    "y_true": np.asarray(y_real_test).tolist(),
                    "y_pred": np.asarray(y_pred_tstr).tolist(),
                }

                # Store ROC for binary TSTR
                if n_classes == 2 and hasattr(model_tstr, "predict_proba"):
                    proba = model_tstr.predict_proba(X_real_test)
                    if proba.shape[1] == 2:
                        fpr, tpr, _ = roc_curve(y_real_test, proba[:, 1])
                        self._roc_curves[name] = {
                            "fpr": fpr.tolist(),
                            "tpr": tpr.tolist(),
                            "auc": tstr_auc,
                        }
                        prec, rec, _ = precision_recall_curve(y_real_test, proba[:, 1])
                        if not hasattr(self, "_pr_curves"):
                            self._pr_curves = {}
                        self._pr_curves[name] = {
                            "precision": prec.tolist(),
                            "recall": rec.tolist(),
                            "ap": results[name]["tstr_pr_auc"],
                        }
            except Exception as e:
                results[name] = {"error": str(e)}

        # Aggregate convenience fields across models
        f1s = [
            m["tstr_f1"]
            for m in results.values()
            if isinstance(m, dict) and "tstr_f1" in m
        ]
        if f1s:
            # Prefer xgboost for single headline metrics when present
            prefer = results.get("xgboost") or next(
                (m for m in results.values() if isinstance(m, dict) and "tstr_f1" in m),
                {},
            )
            results["_aggregates"] = {
                "accuracy": prefer.get("tstr_accuracy"),
                "balanced_accuracy": prefer.get("tstr_balanced_accuracy"),
                "precision_macro": prefer.get("tstr_precision_macro"),
                "recall_macro": prefer.get("tstr_recall_macro"),
                "f1_macro": prefer.get("tstr_f1_macro"),
                "f1_weighted": prefer.get("tstr_f1"),
                "roc_auc": prefer.get("tstr_auc"),
                "pr_auc": prefer.get("tstr_pr_auc"),
                "trtr_f1": prefer.get("trtr_f1"),
                "trtr_auc": prefer.get("trtr_auc"),
            }

        results["_meta"] = {
            "n_features_used": len(used_features),
            "n_train": int(len(X_real_train)),
            "n_test": int(len(X_real_test)),
            "n_synthetic": int(len(X_syn)),
            "n_classes": int(n_classes),
            "task": "classification",
        }
        return results

    def _evaluate_utility_regression(
        self,
        real_train: pd.DataFrame,
        real_test: pd.DataFrame,
        synthetic: pd.DataFrame,
        target_col: str,
        real_train_labels: np.ndarray = None,
        real_test_labels: np.ndarray = None,
        synthetic_labels: np.ndarray = None,
    ) -> dict:
        """TSTR regression utility: R2, MAE, RMSE."""
        results = {}
        self._roc_curves = {}

        feature_cols = [c for c in synthetic.columns if c != target_col]
        common_features = [
            c
            for c in feature_cols
            if c in real_train.columns and c in real_test.columns
        ]
        if not common_features:
            return {"error": "No common feature columns found"}

        X_real_train = (
            real_train[common_features].select_dtypes(include=[np.number]).values
        )
        X_real_test = (
            real_test[common_features].select_dtypes(include=[np.number]).values
        )
        X_syn = synthetic[common_features].select_dtypes(include=[np.number]).values

        y_real_train = (
            np.asarray(real_train_labels, dtype=float)
            if real_train_labels is not None
            else real_train[target_col].astype(float).values
        )
        y_real_test = (
            np.asarray(real_test_labels, dtype=float)
            if real_test_labels is not None
            else real_test[target_col].astype(float).values
        )
        y_syn = (
            np.asarray(synthetic_labels, dtype=float)
            if synthetic_labels is not None
            else synthetic[target_col].astype(float).values
        )

        mask_train = ~np.isnan(X_real_train).any(axis=1) & ~np.isnan(y_real_train)
        mask_test = ~np.isnan(X_real_test).any(axis=1) & ~np.isnan(y_real_test)
        mask_syn = ~np.isnan(X_syn).any(axis=1) & ~np.isnan(y_syn)
        X_real_train, y_real_train = X_real_train[mask_train], y_real_train[mask_train]
        X_real_test, y_real_test = X_real_test[mask_test], y_real_test[mask_test]
        X_syn, y_syn = X_syn[mask_syn], y_syn[mask_syn]

        models = {}
        try:
            from xgboost import XGBRegressor

            models["xgboost"] = XGBRegressor(
                n_estimators=100, max_depth=6, random_state=self.random_state
            )
        except ImportError:
            pass
        try:
            from catboost import CatBoostRegressor

            models["catboost"] = CatBoostRegressor(
                iterations=100, depth=6, random_seed=self.random_state, verbose=0
            )
        except ImportError:
            pass
        models["mlp"] = MLPRegressor(
            hidden_layer_sizes=(128, 64),
            max_iter=300,
            random_state=self.random_state,
        )

        for name, model in models.items():
            try:
                m_trtr = _clone_model(model)
                m_trtr.fit(X_real_train, y_real_train)
                pred_trtr = m_trtr.predict(X_real_test)

                m_tstr = _clone_model(model)
                m_tstr.fit(X_syn, y_syn)
                pred_tstr = m_tstr.predict(X_real_test)

                trtr_r2 = float(r2_score(y_real_test, pred_trtr))
                tstr_r2 = float(r2_score(y_real_test, pred_tstr))
                results[name] = {
                    "trtr_r2": trtr_r2,
                    "tstr_r2": tstr_r2,
                    "r2_gap": round(trtr_r2 - tstr_r2, 4),
                    "trtr_mae": float(mean_absolute_error(y_real_test, pred_trtr)),
                    "tstr_mae": float(mean_absolute_error(y_real_test, pred_tstr)),
                    "trtr_rmse": float(
                        mean_squared_error(y_real_test, pred_trtr) ** 0.5
                    ),
                    "tstr_rmse": float(
                        mean_squared_error(y_real_test, pred_tstr) ** 0.5
                    ),
                }
                if name == "xgboost" or not getattr(self, "_residual_data", None):
                    self._residual_data = {
                        "y_true": np.asarray(y_real_test).tolist(),
                        "y_pred": np.asarray(pred_tstr).tolist(),
                        "model": name,
                    }
            except Exception as e:
                results[name] = {"error": str(e)}

        results["_meta"] = {
            "n_train": int(len(X_real_train)),
            "n_test": int(len(X_real_test)),
            "n_synthetic": int(len(X_syn)),
            "task": "regression",
        }
        return results

    def _evaluate_privacy(
        self,
        real_train: pd.DataFrame,
        real_test: pd.DataFrame,
        synthetic: pd.DataFrame,
        real_train_labels: np.ndarray = None,
        real_test_labels: np.ndarray = None,
    ) -> dict:
        """
        Privacy risk metrics: DCR + distance-based Membership Inference Attack (MIA).
        """
        num_cols = real_train.select_dtypes(include=[np.number]).columns
        common_num = [c for c in num_cols if c in synthetic.columns]

        if not common_num:
            return {"error": "No common numerical columns for DCR computation"}

        real_arr = real_train[common_num].dropna().values.astype(float)
        syn_arr = synthetic[common_num].dropna().values.astype(float)

        if len(real_arr) == 0 or len(syn_arr) == 0:
            return {"error": "Empty arrays for DCR"}

        dists = cdist(syn_arr, real_arr, metric="euclidean")
        dcr_values = dists.min(axis=1)
        n_exact_copies = int((dcr_values == 0.0).sum())

        results = {
            "dcr_median": float(np.median(dcr_values)),
            "dcr_mean": float(np.mean(dcr_values)),
            "dcr_min": float(np.min(dcr_values)),
            "dcr_5th_percentile": float(np.percentile(dcr_values, 5)),
            "n_exact_copies": n_exact_copies,
            "exact_copy_count": n_exact_copies,
            "exact_copy_rate": float(n_exact_copies / max(len(dcr_values), 1)),
            "sanity_check_passed": n_exact_copies == 0,
            "dcr_values": dcr_values.tolist(),
        }

        if results["dcr_5th_percentile"] > 0.5:
            results["privacy_rating"] = "Excellent"
        elif results["dcr_5th_percentile"] > 0.1:
            results["privacy_rating"] = "Good"
        elif results["dcr_5th_percentile"] > 0.01:
            results["privacy_rating"] = "Moderate"
        else:
            results["privacy_rating"] = "Poor"

        # Distance-based MIA: members (train) vs non-members (test)
        test_common = [c for c in common_num if c in real_test.columns]
        if test_common and len(real_test) > 0:
            member = real_train[test_common].dropna().values.astype(float)
            non_member = real_test[test_common].dropna().values.astype(float)
            syn_m = synthetic[test_common].dropna().values.astype(float)
            if len(member) and len(non_member) and len(syn_m):
                rng = np.random.RandomState(self.random_state)
                max_n = 2000
                if len(member) > max_n:
                    member = member[rng.choice(len(member), max_n, replace=False)]
                if len(non_member) > max_n:
                    non_member = non_member[
                        rng.choice(len(non_member), max_n, replace=False)
                    ]

                d_mem = cdist(member, syn_m, metric="euclidean").min(axis=1)
                d_non = cdist(non_member, syn_m, metric="euclidean").min(axis=1)
                scores = np.concatenate([-d_mem, -d_non])
                labels_mia = np.concatenate(
                    [np.ones(len(d_mem)), np.zeros(len(d_non))]
                )
                try:
                    mia_auc = float(roc_auc_score(labels_mia, scores))
                except ValueError:
                    mia_auc = None
                results["mia_auc"] = mia_auc
                results["mia_member_mean_dist"] = float(np.mean(d_mem))
                results["mia_nonmember_mean_dist"] = float(np.mean(d_non))
                results["mia_near_random"] = (
                    bool(abs(mia_auc - 0.5) < 0.05) if mia_auc is not None else None
                )
                # Threshold accuracy at score median
                thr = float(np.median(scores))
                pred = (scores >= thr).astype(int)
                results["mia_accuracy"] = float((pred == labels_mia).mean())

        return results

    def get_roc_curves(self) -> dict:
        return self._roc_curves

    def get_pr_curves(self) -> dict:
        return getattr(self, "_pr_curves", {})

    def get_confusion_data(self) -> dict:
        return getattr(self, "_confusion_data", {})

    def get_residual_data(self) -> Optional[dict]:
        return getattr(self, "_residual_data", None)

    def summary_row(self, dataset: str, method: str = "dgd_tabpa") -> dict:
        """Flat row suitable for CSV / summary tables."""
        res = self.results.get("resemblance", {})
        util = self.results.get("utility", {})
        priv = self.results.get("privacy", {})

        row = {
            "dataset": dataset,
            "method": method,
            "task": self.results.get("task", util.get("_meta", {}).get("task", "classification")),
            "wasserstein_mean": res.get("wasserstein_mean"),
            "pcd": res.get("pcd"),
            "jsd_mean": res.get("jsd_mean"),
            "dcr_median": priv.get("dcr_median"),
            "dcr_5th_percentile": priv.get("dcr_5th_percentile"),
            "n_exact_copies": priv.get("n_exact_copies"),
            "mia_auc": priv.get("mia_auc"),
            "privacy_rating": priv.get("privacy_rating"),
        }

        f1s, gaps, aucs, r2s, r2_gaps = [], [], [], [], []
        for model_name, metrics in util.items():
            if model_name.startswith("_"):
                continue
            if not isinstance(metrics, dict):
                continue
            if "tstr_f1" in metrics:
                row[f"{model_name}_trtr_f1"] = metrics["trtr_f1"]
                row[f"{model_name}_tstr_f1"] = metrics["tstr_f1"]
                row[f"{model_name}_f1_gap"] = metrics["f1_gap"]
                row[f"{model_name}_tstr_auc"] = metrics.get("tstr_auc")
                f1s.append(metrics["tstr_f1"])
                gaps.append(metrics["f1_gap"])
                if metrics.get("tstr_auc") is not None:
                    aucs.append(metrics["tstr_auc"])
            if "tstr_r2" in metrics:
                row[f"{model_name}_trtr_r2"] = metrics["trtr_r2"]
                row[f"{model_name}_tstr_r2"] = metrics["tstr_r2"]
                row[f"{model_name}_r2_gap"] = metrics["r2_gap"]
                row[f"{model_name}_tstr_mae"] = metrics.get("tstr_mae")
                r2s.append(metrics["tstr_r2"])
                r2_gaps.append(metrics["r2_gap"])

        row["mean_tstr_f1"] = float(np.mean(f1s)) if f1s else None
        row["mean_f1_gap"] = float(np.mean(gaps)) if gaps else None
        row["mean_tstr_auc"] = float(np.mean(aucs)) if aucs else None
        row["mean_tstr_r2"] = float(np.mean(r2s)) if r2s else None
        row["mean_r2_gap"] = float(np.mean(r2_gaps)) if r2_gaps else None
        return row

    def save_results(self, path: str | Path, extra: dict = None) -> Path:
        """Persist full results JSON (large arrays truncated for DCR)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "results": _json_safe(self.results),
            "roc_curves": self._roc_curves,
        }
        if extra:
            payload["meta"] = _json_safe(extra)
        # Truncate bulky DCR list in saved copy
        priv = payload["results"].get("privacy", {})
        if isinstance(priv, dict) and "dcr_values" in priv:
            vals = priv["dcr_values"]
            if isinstance(vals, list) and len(vals) > 500:
                priv["dcr_values_sample"] = vals[:500]
                priv["dcr_values_n"] = len(vals)
                del priv["dcr_values"]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return path

    def evaluate_with_syntheval(
        self,
        real_df: pd.DataFrame,
        synthetic_df: pd.DataFrame,
        target_col: str,
        cat_cols: list = None,
    ) -> dict:
        """Run SynthEval evaluation if the library is available."""
        try:
            from syntheval import SynthEval

            evaluator = SynthEval(real_df, target=target_col, cat_cols=cat_cols)
            results = evaluator.evaluate(synthetic_df)
            return {"syntheval": results, "source": "syntheval_library"}
        except ImportError:
            print("[Evaluation] SynthEval not installed. Using custom metrics.")
            return {
                "source": "custom",
                "note": "Install syntheval for full evaluation",
            }
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

        res = self.results.get("resemblance", {})
        print("\n--- RESEMBLANCE (Statistical Similarity) ---")
        if res.get("wasserstein_mean") is not None:
            print(f"  Mean Wasserstein Distance: {res['wasserstein_mean']:.4f}")
        if res.get("pcd") is not None:
            print(f"  Pairwise Correlation Diff: {res['pcd']:.4f}")
        if res.get("jsd_mean") is not None:
            print(f"  Mean Jensen-Shannon Dist:  {res['jsd_mean']:.4f}")

        util = self.results.get("utility", {})
        print("\n--- UTILITY (Train Synthetic, Test Real) ---")
        for model_name, metrics in util.items():
            if model_name.startswith("_"):
                continue
            if not isinstance(metrics, dict):
                continue
            if "tstr_f1" in metrics:
                print(f"  [{model_name}]")
                print(f"    TRTR F1:  {metrics['trtr_f1']:.4f} (baseline)")
                print(f"    TSTR F1:  {metrics['tstr_f1']:.4f}")
                print(f"    F1 Gap:   {metrics['f1_gap']:.4f}")
                if metrics.get("tstr_auc") is not None:
                    print(f"    TSTR AUC: {metrics['tstr_auc']:.4f}")
            elif "tstr_r2" in metrics:
                print(f"  [{model_name}]")
                print(f"    TRTR R2:  {metrics['trtr_r2']:.4f} (baseline)")
                print(f"    TSTR R2:  {metrics['tstr_r2']:.4f}")
                print(f"    R2 Gap:   {metrics['r2_gap']:.4f}")
                print(f"    TSTR MAE: {metrics['tstr_mae']:.4f}")

        priv = self.results.get("privacy", {})
        print("\n--- PRIVACY (DCR + MIA) ---")
        if "dcr_median" in priv:
            print(f"  DCR Median: {priv['dcr_median']:.4f}")
            print(f"  DCR 5th Percentile: {priv['dcr_5th_percentile']:.4f}")
            print(f"  Exact Copies: {priv['n_exact_copies']}")
            print(
                f"  Sanity Check: "
                f"{'PASSED' if priv['sanity_check_passed'] else 'FAILED'}"
            )
            print(f"  Privacy Rating: {priv['privacy_rating']}")
        if priv.get("mia_auc") is not None:
            print(f"  MIA AUC: {priv['mia_auc']:.4f} (0.5 ~ random)")
            print(f"  MIA near-random: {priv.get('mia_near_random')}")

        print("\n" + "=" * 60)


def _jsd_categorical(real_series: pd.Series, syn_series: pd.Series) -> Optional[float]:
    """Jensen–Shannon distance between categorical marginals (base-2, in [0, 1])."""
    real_counts = real_series.astype(str).value_counts(normalize=True)
    syn_counts = syn_series.astype(str).value_counts(normalize=True)
    cats = sorted(set(real_counts.index) | set(syn_counts.index))
    if len(cats) < 2:
        return None
    p = np.array([real_counts.get(c, 0.0) for c in cats], dtype=float)
    q = np.array([syn_counts.get(c, 0.0) for c in cats], dtype=float)
    p = p / p.sum()
    q = q / q.sum()
    return float(jensenshannon(p, q, base=2.0))


def _align_labels(y_train, y_test, y_syn):
    """Map heterogeneous label values to a shared integer encoding."""
    all_vals = pd.Series(
        list(np.asarray(y_train))
        + list(np.asarray(y_test))
        + list(np.asarray(y_syn))
    ).astype(str)
    codes, _ = pd.factorize(all_vals)
    n_tr, n_te = len(y_train), len(y_test)
    return (
        codes[:n_tr].astype(np.int64),
        codes[n_tr : n_tr + n_te].astype(np.int64),
        codes[n_tr + n_te :].astype(np.int64),
    )


def _safe_pr_auc(model, X, y, n_classes) -> Optional[float]:
    try:
        if not hasattr(model, "predict_proba") or n_classes != 2:
            return None
        proba = model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] == 2:
            return float(average_precision_score(y, proba[:, 1]))
        return float(average_precision_score(y, proba))
    except Exception:
        return None


def _safe_auc(model, X, y, n_classes) -> Optional[float]:
    try:
        if not hasattr(model, "predict_proba"):
            return None
        proba = model.predict_proba(X)
        if n_classes == 2:
            if proba.shape[1] == 2:
                return float(roc_auc_score(y, proba[:, 1]))
            return float(roc_auc_score(y, proba))
        return float(
            roc_auc_score(y, proba, multi_class="ovr", average="weighted")
        )
    except Exception:
        return None


def _clone_model(model):
    """Create a fresh clone of a sklearn-compatible model."""
    from sklearn.base import clone

    try:
        return clone(model)
    except Exception:
        return model.__class__(**model.get_params())


def _json_safe(obj):
    """Convert numpy types for JSON serialization."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj
