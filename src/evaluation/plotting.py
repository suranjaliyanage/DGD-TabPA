"""
Figure generation for DGD-TabPA experiments.

Produces thesis-ready PNGs: loss curves, marginals, categorical bars,
correlation heatmaps, PCA/t-SNE, ROC curves, DCR histograms, privacy–utility.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def _ensure_dir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_loss_curve(
    loss_history: list,
    out_path: Path,
    title: str = "Training Loss",
    ylabel: str = "Loss",
    label: str = None,
):
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(
        range(1, len(loss_history) + 1),
        loss_history,
        color="steelblue",
        linewidth=1.5,
        label=label,
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if label:
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_loss_comparison(
    histories: dict,
    out_path: Path,
    title: str = "Loss Comparison",
):
    """histories: {label: list[float]}"""
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, hist in histories.items():
        ax.plot(range(1, len(hist) + 1), hist, linewidth=1.5, label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_marginal_distributions(
    real_df: pd.DataFrame,
    syn_df: pd.DataFrame,
    columns: list,
    out_path: Path,
    title: str = "Real vs Synthetic Marginal Distributions",
    max_cols: int = 6,
):
    cols = [c for c in columns if c in real_df.columns and c in syn_df.columns][
        :max_cols
    ]
    if not cols:
        return None

    n = len(cols)
    fig, axes = plt.subplots(2, n, figsize=(3.5 * n, 6))
    if n == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    fig.suptitle(title, fontsize=13)

    for i, col in enumerate(cols):
        axes[0, i].hist(
            real_df[col].dropna().astype(float),
            bins=40,
            density=True,
            alpha=0.75,
            color="steelblue",
        )
        axes[0, i].set_title(f"{col} (Real)")
        axes[1, i].hist(
            syn_df[col].dropna().astype(float),
            bins=40,
            density=True,
            alpha=0.75,
            color="coral",
        )
        axes[1, i].set_title(f"{col} (Synthetic)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_categorical_frequencies(
    real_df: pd.DataFrame,
    syn_df: pd.DataFrame,
    columns: list,
    out_path: Path,
    title: str = "Categorical Frequency Alignment",
    max_cols: int = 4,
):
    cols = [c for c in columns if c in real_df.columns and c in syn_df.columns][
        :max_cols
    ]
    if not cols:
        return None

    n = len(cols)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=13)

    for ax, col in zip(axes, cols):
        real_p = real_df[col].astype(str).value_counts(normalize=True)
        syn_p = syn_df[col].astype(str).value_counts(normalize=True)
        cats = list(dict.fromkeys(list(real_p.index) + list(syn_p.index)))[:12]
        x = np.arange(len(cats))
        w = 0.35
        ax.bar(
            x - w / 2,
            [real_p.get(c, 0) for c in cats],
            w,
            label="Real",
            color="steelblue",
        )
        ax.bar(
            x + w / 2,
            [syn_p.get(c, 0) for c in cats],
            w,
            label="Synthetic",
            color="coral",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(cats, rotation=45, ha="right", fontsize=8)
        ax.set_title(col)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_correlation_heatmaps(
    real_df: pd.DataFrame,
    syn_df: pd.DataFrame,
    columns: list,
    out_path: Path,
    title: str = "Correlation Structure (Real | Synthetic | Absolute Diff)",
):
    cols = [c for c in columns if c in real_df.columns and c in syn_df.columns]
    if len(cols) < 2:
        return None

    real_corr = real_df[cols].corr().fillna(0)
    syn_corr = syn_df[cols].corr().fillna(0)
    diff = (real_corr - syn_corr).abs()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle(title, fontsize=13)
    for ax, mat, name, cmap in zip(
        axes,
        [real_corr, syn_corr, diff],
        ["Real", "Synthetic", "|Diff| (PCD map)"],
        ["coolwarm", "coolwarm", "YlOrRd"],
    ):
        sns.heatmap(
            mat,
            ax=ax,
            cmap=cmap,
            center=0 if name != "|Diff| (PCD map)" else None,
            vmin=-1 if name != "|Diff| (PCD map)" else 0,
            vmax=1,
            square=True,
            cbar_kws={"shrink": 0.7},
            xticklabels=True,
            yticklabels=True,
        )
        ax.set_title(name)
        ax.tick_params(labelsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_manifold(
    real_X: np.ndarray,
    syn_X: np.ndarray,
    real_y: np.ndarray,
    syn_y: np.ndarray,
    out_path: Path,
    title: str = "Manifold Alignment (PCA / t-SNE)",
    max_points: int = 1500,
    seed: int = 42,
):
    """Joint PCA and t-SNE of real vs synthetic feature space."""
    rng = np.random.RandomState(seed)

    def _sub(X, y):
        if len(X) > max_points:
            idx = rng.choice(len(X), max_points, replace=False)
            return X[idx], np.asarray(y)[idx]
        return X, np.asarray(y)

    real_X, real_y = _sub(np.asarray(real_X, dtype=float), real_y)
    syn_X, syn_y = _sub(np.asarray(syn_X, dtype=float), syn_y)

    # Align dims
    d = min(real_X.shape[1], syn_X.shape[1])
    real_X, syn_X = real_X[:, :d], syn_X[:, :d]

    scaler = StandardScaler()
    joint = np.vstack([real_X, syn_X])
    joint = np.nan_to_num(joint, nan=0.0)
    joint_s = scaler.fit_transform(joint)
    n_real = len(real_X)

    pca = PCA(n_components=2, random_state=seed)
    emb_pca = pca.fit_transform(joint_s)

    # t-SNE can be slow; use PCA init on a capped set
    tsne = TSNE(
        n_components=2,
        perplexity=min(30, max(5, len(joint_s) // 10)),
        random_state=seed,
        init="pca",
        learning_rate="auto",
    )
    emb_tsne = tsne.fit_transform(joint_s)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=13)

    for ax, emb, name in zip(axes, [emb_pca, emb_tsne], ["PCA", "t-SNE"]):
        ax.scatter(
            emb[:n_real, 0],
            emb[:n_real, 1],
            c=real_y,
            cmap="tab10",
            s=12,
            alpha=0.5,
            marker="o",
            label="Real",
        )
        ax.scatter(
            emb[n_real:, 0],
            emb[n_real:, 1],
            c=syn_y,
            cmap="tab10",
            s=18,
            alpha=0.8,
            marker="x",
            label="Synthetic",
        )
        ax.set_title(name)
        ax.legend(markerscale=1.5, fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_roc_curves(
    roc_curves: dict,
    out_path: Path,
    title: str = "TSTR ROC Curves",
):
    if not roc_curves:
        return None
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, curve in roc_curves.items():
        ax.plot(
            curve["fpr"],
            curve["tpr"],
            linewidth=1.8,
            label=f"{name} (AUC={curve.get('auc', float('nan')):.3f})",
        )
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_dcr_histogram(
    dcr_values: list,
    out_path: Path,
    title: str = "Distance to Closest Record (DCR)",
):
    if not dcr_values:
        return None
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(dcr_values, bins=40, color="teal", alpha=0.8, edgecolor="white")
    ax.axvline(
        np.median(dcr_values),
        color="crimson",
        linestyle="--",
        label=f"Median={np.median(dcr_values):.3f}",
    )
    ax.set_xlabel("Euclidean DCR")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_privacy_utility(
    points: list,
    out_path: Path,
    title: str = "Privacy–Utility Trade-off",
):
    """
    points: list of dicts with keys epsilon, tstr_f1, dcr_median, label (optional)
    """
    if not points:
        return None
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(title, fontsize=13)

    eps = [p["epsilon"] for p in points]
    f1 = [p["tstr_f1"] for p in points]
    dcr = [p.get("dcr_median") for p in points]
    labels = [p.get("label", f"ε={e}") for p, e in zip(points, eps)]

    axes[0].plot(eps, f1, "o-", color="steelblue", linewidth=1.8)
    for x, y, lab in zip(eps, f1, labels):
        axes[0].annotate(lab, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7)
    axes[0].set_xlabel("Privacy budget ε")
    axes[0].set_ylabel("Mean TSTR F1")
    axes[0].set_title("Utility vs ε")

    axes[1].plot(eps, dcr, "s-", color="teal", linewidth=1.8)
    axes[1].set_xlabel("Privacy budget ε")
    axes[1].set_ylabel("DCR Median")
    axes[1].set_title("Privacy (DCR) vs ε")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def generate_all_figures(
    out_dir: Path,
    dataset: str,
    real_train: pd.DataFrame,
    syn_df: pd.DataFrame,
    target_col: str,
    num_features: list,
    cat_features: list,
    train_loss: list = None,
    distill_loss: list = None,
    roc_curves: dict = None,
    dcr_values: list = None,
    real_y: np.ndarray = None,
    syn_y: np.ndarray = None,
    real_X_num: np.ndarray = None,
    syn_X_num: np.ndarray = None,
) -> dict:
    """Generate the full evaluation figure set for one experiment run."""
    fig_dir = _ensure_dir(Path(out_dir) / "figures")
    paths = {}

    if train_loss:
        paths["train_loss"] = str(
            plot_loss_curve(
                train_loss,
                fig_dir / f"train_loss_{dataset}.png",
                title=f"Diffusion Training Loss ({dataset})",
            )
        )
    if distill_loss:
        paths["distill_loss"] = str(
            plot_loss_curve(
                distill_loss,
                fig_dir / f"distill_loss_{dataset}.png",
                title=f"Distillation MMD Loss ({dataset})",
                ylabel="MMD Loss",
            )
        )

    if num_features:
        p = plot_marginal_distributions(
            real_train,
            syn_df,
            num_features,
            fig_dir / f"marginals_{dataset}.png",
            title=f"Marginal Distributions ({dataset})",
        )
        if p:
            paths["marginals"] = str(p)

    if cat_features:
        p = plot_categorical_frequencies(
            real_train,
            syn_df,
            cat_features,
            fig_dir / f"categorical_{dataset}.png",
            title=f"Categorical Frequencies ({dataset})",
        )
        if p:
            paths["categorical"] = str(p)

    if num_features and len(num_features) >= 2:
        p = plot_correlation_heatmaps(
            real_train,
            syn_df,
            num_features[:12],
            fig_dir / f"correlation_{dataset}.png",
            title=f"Correlation Heatmaps ({dataset})",
        )
        if p:
            paths["correlation"] = str(p)

    if real_X_num is not None and syn_X_num is not None:
        p = plot_manifold(
            real_X_num,
            syn_X_num,
            real_y if real_y is not None else np.zeros(len(real_X_num)),
            syn_y if syn_y is not None else np.zeros(len(syn_X_num)),
            fig_dir / f"manifold_{dataset}.png",
            title=f"PCA / t-SNE Manifold Alignment ({dataset})",
        )
        if p:
            paths["manifold"] = str(p)

    if roc_curves:
        p = plot_roc_curves(
            roc_curves,
            fig_dir / f"roc_{dataset}.png",
            title=f"TSTR ROC Curves ({dataset})",
        )
        if p:
            paths["roc"] = str(p)

    if dcr_values:
        p = plot_dcr_histogram(
            dcr_values,
            fig_dir / f"dcr_{dataset}.png",
            title=f"DCR Distribution ({dataset})",
        )
        if p:
            paths["dcr"] = str(p)

    return paths
