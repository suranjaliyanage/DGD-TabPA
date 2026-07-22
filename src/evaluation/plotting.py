"""
Figure generation for DGD-TabPA experiments.

Produces evaluation PNGs: loss curves, marginals, categorical bars,
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
    fig.savefig(out_path, dpi=300)
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
    fig.savefig(out_path, dpi=300)
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
    fig.savefig(out_path, dpi=300)
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
    fig.savefig(out_path, dpi=300)
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
    fig.savefig(out_path, dpi=300)
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
    fig.savefig(out_path, dpi=300)
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
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_dcr_histogram(
    dcr_values: list,
    out_path: Path,
    title: str = "Distance to Closest Record (DCR)",
    exact_copy_count: int = None,
):
    if not dcr_values:
        return None
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(7, 4))
    vals = np.asarray(dcr_values, dtype=float)
    ax.hist(vals, bins=40, color="teal", alpha=0.8, edgecolor="white")
    med = float(np.median(vals))
    p5 = float(np.percentile(vals, 5))
    ax.axvline(med, color="crimson", linestyle="--", label=f"Median={med:.3f}")
    ax.axvline(p5, color="darkorange", linestyle=":", label=f"5th pct={p5:.3f}")
    n_exact = (
        int(exact_copy_count)
        if exact_copy_count is not None
        else int((vals == 0.0).sum())
    )
    ax.set_xlabel("Euclidean DCR")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend(title=f"Exact copies={n_exact}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_pr_curves(pr_curves: dict, out_path: Path, title: str = "TSTR Precision-Recall"):
    if not pr_curves:
        return None
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    for name, curve in pr_curves.items():
        ax.plot(
            curve["recall"],
            curve["precision"],
            linewidth=1.8,
            label=f"{name} (AP={curve.get('ap', float('nan')):.3f})",
        )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_class_distribution(
    y_real,
    y_syn,
    out_path: Path,
    title: str = "Class / target distribution",
):
    import matplotlib.pyplot as plt
    import pandas as pd

    fig, ax = plt.subplots(figsize=(7, 4))
    real_c = pd.Series(y_real).astype(str).value_counts(normalize=True)
    syn_c = pd.Series(y_syn).astype(str).value_counts(normalize=True)
    cats = list(dict.fromkeys(list(real_c.index) + list(syn_c.index)))[:15]
    x = np.arange(len(cats))
    w = 0.35
    ax.bar(x - w / 2, [real_c.get(c, 0) for c in cats], w, label="Real", color="steelblue")
    ax.bar(x + w / 2, [syn_c.get(c, 0) for c in cats], w, label="Synthetic", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_residuals(
    y_true,
    y_pred,
    out_path: Path,
    title: str = "TSTR Residuals",
):
    import matplotlib.pyplot as plt

    resid = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_pred, resid, s=12, alpha=0.6, color="teal")
    ax.axhline(0, color="crimson", linestyle="--")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual (true - pred)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_confusion_matrix(
    y_true,
    y_pred,
    out_path: Path,
    title: str = "TSTR Confusion Matrix",
):
    """Save a normalised confusion matrix heatmap."""
    from sklearn.metrics import confusion_matrix

    import matplotlib.pyplot as plt

    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    labels = sorted(set(yt.tolist()) | set(yp.tolist()))
    cm = confusion_matrix(yt, yp, labels=labels, normalize="true")
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_privacy_utility(
    points: list,
    out_path: Path,
    title: str = "Privacy-Utility Trade-off",
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
    labels = [p.get("label", f"eps={e}") for p, e in zip(points, eps)]

    axes[0].plot(eps, f1, "o-", color="steelblue", linewidth=1.8)
    for x, y, lab in zip(eps, f1, labels):
        axes[0].annotate(lab, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7)
    axes[0].set_xlabel("Privacy budget eps")
    axes[0].set_ylabel("Mean TSTR F1")
    axes[0].set_title("Utility vs eps")

    axes[1].plot(eps, dcr, "s-", color="teal", linewidth=1.8)
    axes[1].set_xlabel("Privacy budget ε")
    axes[1].set_ylabel("DCR Median")
    axes[1].set_title("Privacy (DCR) vs ε")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
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
    method: str = "dgd_tabpa",
    pr_curves: dict = None,
    confusion_data: dict = None,
    residual_data: dict = None,
    exact_copy_count: int = None,
    task: str = "classification",
) -> dict:
    """Generate the full evaluation figure set for one experiment run."""
    fig_dir = _ensure_dir(Path(out_dir) / "figures")
    paths = {}
    tag = f"{dataset}_{method}"

    if train_loss:
        paths["train_loss"] = str(
            plot_loss_curve(
                train_loss,
                fig_dir / f"training_loss_{tag}.png",
                title=f"Diffusion Training Loss ({dataset}, {method})",
            )
        )
    if distill_loss:
        paths["distill_loss"] = str(
            plot_loss_curve(
                distill_loss,
                fig_dir / f"distillation_loss_{tag}.png",
                title=f"Distillation MMD Loss ({dataset}, {method})",
                ylabel="MMD Loss",
            )
        )

    # Choose 3–6 representative numerical features
    num_sel = list(num_features or [])[:6]
    if num_sel:
        p = plot_marginal_distributions(
            real_train,
            syn_df,
            num_sel,
            fig_dir / f"marginals_{tag}.png",
            title=f"Marginal Distributions ({dataset}, {method})",
        )
        if p:
            paths["marginals"] = str(p)

    if cat_features:
        p = plot_categorical_frequencies(
            real_train,
            syn_df,
            list(cat_features)[:4],
            fig_dir / f"categorical_{tag}.png",
            title=f"Categorical Frequencies ({dataset}, {method})",
        )
        if p:
            paths["categorical"] = str(p)

    if num_features and len(num_features) >= 2:
        p = plot_correlation_heatmaps(
            real_train,
            syn_df,
            num_features[:12],
            fig_dir / f"correlation_{tag}.png",
            title=f"Correlation Heatmaps ({dataset}, {method})",
        )
        if p:
            paths["correlation"] = str(p)

    if real_X_num is not None and syn_X_num is not None:
        p = plot_manifold(
            real_X_num,
            syn_X_num,
            real_y if real_y is not None else np.zeros(len(real_X_num)),
            syn_y if syn_y is not None else np.zeros(len(syn_X_num)),
            fig_dir / f"manifold_{tag}.png",
            title=f"PCA / t-SNE Manifold Alignment ({dataset}, {method})",
        )
        if p:
            paths["manifold"] = str(p)

    if roc_curves and task == "classification":
        p = plot_roc_curves(
            roc_curves,
            fig_dir / f"roc_{tag}.png",
            title=f"TSTR ROC Curves ({dataset}, {method})",
        )
        if p:
            paths["roc"] = str(p)

    if pr_curves and task == "classification":
        p = plot_pr_curves(
            pr_curves,
            fig_dir / f"pr_curve_{tag}.png",
            title=f"TSTR PR Curves ({dataset}, {method})",
        )
        if p:
            paths["pr_curve"] = str(p)

    if confusion_data and task == "classification":
        for model_name, data in confusion_data.items():
            p = plot_confusion_matrix(
                data["y_true"],
                data["y_pred"],
                fig_dir / f"confusion_matrix_{dataset}_{model_name}_{method}.png",
                title=f"TSTR Confusion ({dataset}, {model_name}, {method})",
            )
            if p:
                paths[f"confusion_{model_name}"] = str(p)

    if residual_data and task == "regression":
        p = plot_residuals(
            residual_data["y_true"],
            residual_data["y_pred"],
            fig_dir / f"residuals_{tag}.png",
            title=f"TSTR Residuals ({dataset}, {method})",
        )
        if p:
            paths["residuals"] = str(p)

    if dcr_values:
        p = plot_dcr_histogram(
            dcr_values,
            fig_dir / f"dcr_{tag}.png",
            title=f"DCR Distribution ({dataset}, {method})",
            exact_copy_count=exact_copy_count,
        )
        if p:
            paths["dcr"] = str(p)

    if real_y is not None and syn_y is not None:
        try:
            p = plot_class_distribution(
                real_y,
                syn_y,
                fig_dir / f"class_distribution_{tag}.png",
                title=f"Class/target distribution ({dataset}, {method})",
            )
            if p:
                paths["class_distribution"] = str(p)
        except Exception:
            pass

    return paths
