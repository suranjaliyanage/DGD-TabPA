"""
End-to-end experiment runner for DGD-TabPA evaluation results.

Pipeline: load data → preprocess → train diffusion → distill (or SMOTE) →
evaluate (fidelity / TSTR / privacy) → save metrics JSON/CSV + figures.

Usage:
    python scripts/run_experiment.py --dataset diabetes --epochs 30
    python scripts/run_experiment.py --dataset adult --method smote
    python scripts/run_experiment.py --dataset diabetes --ablation mlp_denoiser
    python scripts/run_experiment.py --dataset diabetes --privacy --epsilon 4.0
    python scripts/run_experiment.py --dataset diabetes --ablation minmax
    python scripts/run_experiment.py --dataset diabetes --ablation no_attention
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baselines import generate_smote_synthetic
from src.distillation import DistillationLoop
from src.evaluation import Evaluator
from src.evaluation.plotting import generate_all_figures
from src.models import GaussianDiffusion, MLPDenoiser, TabularTransformerDenoiser
from src.preprocessing import TabularPreprocessor
from src.privacy.dp_sgd import DPSGDWrapper


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(config_device: str) -> torch.device:
    if config_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(config_device)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_denoiser(cfg: dict, info, ablation: str):
    model_cfg = cfg["model"]
    common = dict(
        total_dim=info.total_dim,
        num_classes=info.num_classes,
        d_model=model_cfg["d_model"],
        n_heads=model_cfg["n_heads"],
        n_encoder_layers=model_cfg["n_encoder_layers"],
        n_decoder_layers=model_cfg["n_decoder_layers"],
        d_ff=model_cfg["d_ff"],
        dropout=model_cfg["dropout"],
    )
    if ablation == "mlp_denoiser":
        return MLPDenoiser(**common)
    use_attn = ablation != "no_attention"
    return TabularTransformerDenoiser(
        **common, use_conditioning_attention=use_attn
    )


def train_diffusion(
    cfg: dict,
    info,
    X_train_t: torch.Tensor,
    y_train_t: torch.Tensor,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    mask_ratio: float,
    ablation: str,
    privacy_enabled: bool,
    epsilon: float,
    save_dir: Path,
    dataset_name: str,
):
    denoiser = build_denoiser(cfg, info, ablation)
    diff_cfg = cfg["diffusion"]
    diffusion = GaussianDiffusion(
        denoiser=denoiser,
        num_timesteps=diff_cfg["num_timesteps"],
        beta_start=diff_cfg["beta_start"],
        beta_end=diff_cfg["beta_end"],
        beta_schedule=diff_cfg["beta_schedule"],
        loss_type=diff_cfg["loss_type"],
    ).to(device)

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True
    )

    optimizer = torch.optim.AdamW(
        diffusion.parameters(),
        lr=lr,
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    privacy_cfg = dict(cfg.get("privacy", {}))
    privacy_cfg["enabled"] = privacy_enabled
    privacy_cfg["epsilon"] = epsilon
    dp_wrapper = DPSGDWrapper.from_config({"privacy": privacy_cfg})
    diffusion, optimizer, train_loader = dp_wrapper.attach(
        model=diffusion,
        optimizer=optimizer,
        data_loader=train_loader,
        epochs=epochs,
    )

    print(
        f"\nTraining diffusion ({ablation or 'transformer'}) for {epochs} epochs "
        f"(mask_ratio={mask_ratio}, privacy={privacy_enabled}, eps={epsilon})..."
    )
    best_loss = float("inf")
    loss_history = []
    epsilon_history = []

    for epoch in range(1, epochs + 1):
        diffusion.train()
        epoch_loss = 0.0
        n_batches = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for X_batch, y_batch in pbar:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            t = torch.randint(
                0, diff_cfg["num_timesteps"], (X_batch.size(0),), device=device
            )
            loss = diffusion.compute_loss(
                x_start=X_batch, t=t, labels=y_batch, mask_ratio=mask_ratio
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                diffusion.parameters(), cfg["training"]["gradient_clip"]
            )
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        loss_history.append(avg_loss)

        eps_now = dp_wrapper.get_epsilon()
        if eps_now is not None:
            epsilon_history.append(float(eps_now))

        if epoch % cfg["training"]["log_interval"] == 0 or epoch == 1:
            msg = f"  Epoch {epoch}/{epochs} | Loss: {avg_loss:.6f}"
            if eps_now is not None:
                msg += f" | eps: {eps_now:.4f}"
            print(msg)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": diffusion.state_dict(),
                    "loss": best_loss,
                    "config": cfg,
                    "ablation": ablation,
                    "dataset_info": {
                        "name": info.name,
                        "total_dim": info.total_dim,
                        "num_classes": info.num_classes,
                        "num_features": info.num_features,
                        "cat_features": info.cat_features,
                        "target_col": info.target_col,
                    },
                },
                save_dir / f"best_model_{dataset_name}.pt",
            )

    torch.save(
        {
            "epoch": epochs,
            "model_state_dict": diffusion.state_dict(),
            "loss_history": loss_history,
            "epsilon_history": epsilon_history,
            "config": cfg,
            "ablation": ablation,
        },
        save_dir / f"final_model_{dataset_name}.pt",
    )

    privacy_report = dp_wrapper.get_privacy_report()
    return diffusion, loss_history, privacy_report, epsilon_history


def run_distillation(
    cfg: dict,
    diffusion,
    X_train_t: torch.Tensor,
    y_train_t: torch.Tensor,
    device: torch.device,
    info,
    raw_space: bool,
    X_train_df: pd.DataFrame,
    y_train_raw: np.ndarray,
):
    """
    Distill in latent/transformed space (default) or raw numeric space.
    """
    d_cfg = cfg["distillation"]

    if raw_space:
        print("\n[Ablation] Distilling in RAW numeric feature space...")
        num_cols = X_train_df.select_dtypes(include=[np.number]).columns.tolist()
        X_raw = torch.tensor(
            X_train_df[num_cols].fillna(0).values.astype(np.float32)
        )
        # Map labels
        from sklearn.preprocessing import LabelEncoder

        le = LabelEncoder()
        y_enc = torch.tensor(le.fit_transform(y_train_raw), dtype=torch.long)
        loop = DistillationLoop(
            diffusion_model=diffusion,
            total_dim=X_raw.shape[1],
            num_classes=int(y_enc.unique().numel()),
            num_synthetic=d_cfg["num_synthetic"],
            distill_lr=d_cfg["distill_lr"],
            distill_epochs=d_cfg["distill_epochs"],
            inner_steps=d_cfg["inner_steps"],
            device=device,
        )
        syn_data, syn_labels, distill_loss = loop.distill(
            X_raw, y_enc, batch_size=cfg["training"]["batch_size"]
        )
        syn_df = pd.DataFrame(syn_data.cpu().numpy(), columns=num_cols)
        # Sample categoricals from real empirical
        for col in X_train_df.columns:
            if col not in syn_df.columns:
                syn_df[col] = (
                    X_train_df[col]
                    .sample(n=len(syn_df), replace=True, random_state=42)
                    .values
                )
        syn_label_names = le.inverse_transform(syn_labels.cpu().numpy())
        return syn_df, syn_label_names, distill_loss

    loop = DistillationLoop(
        diffusion_model=diffusion,
        total_dim=info.total_dim,
        num_classes=info.num_classes,
        num_synthetic=d_cfg["num_synthetic"],
        distill_lr=d_cfg["distill_lr"],
        distill_epochs=d_cfg["distill_epochs"],
        inner_steps=d_cfg["inner_steps"],
        device=device,
    )
    syn_data, syn_labels, distill_loss = loop.distill(
        X_train_t, y_train_t, batch_size=cfg["training"]["batch_size"]
    )
    return syn_data, syn_labels, distill_loss


def evaluate_and_plot(
    cfg: dict,
    dataset_name: str,
    method: str,
    preprocessor: TabularPreprocessor,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train,
    y_test,
    syn_df: pd.DataFrame,
    syn_labels,
    target_col: str,
    out_dir: Path,
    train_loss=None,
    distill_loss=None,
    privacy_report=None,
    extra_meta=None,
    task: str = "classification",
):
    # Ensure target column present with original label values
    syn_eval = syn_df.copy()
    if target_col not in syn_eval.columns:
        syn_eval[target_col] = syn_labels

    real_train = X_train.copy()
    real_train[target_col] = y_train
    real_test = X_test.copy()
    real_test[target_col] = y_test

    # Numeric matrices for manifold plot (shared columns)
    num_cols = [
        c
        for c in preprocessor.info.num_features
        if c in real_train.columns and c in syn_eval.columns
    ]
    real_X_num = (
        real_train[num_cols].fillna(0).values.astype(float) if num_cols else None
    )
    syn_X_num = (
        syn_eval[num_cols].fillna(0).values.astype(float) if num_cols else None
    )

    # Encode labels for manifold colouring
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    if task == "regression":
        # Bin continuous targets for colouring only
        from sklearn.preprocessing import KBinsDiscretizer

        binner = KBinsDiscretizer(n_bins=10, encode="ordinal", strategy="quantile")
        y_stack = np.concatenate(
            [np.asarray(y_train, dtype=float), np.asarray(syn_labels, dtype=float)]
        ).reshape(-1, 1)
        y_bins = binner.fit_transform(y_stack).astype(int).ravel()
        real_y_enc = y_bins[: len(y_train)]
        syn_y_enc = y_bins[len(y_train) :]
    else:
        y_all = list(np.asarray(y_train).astype(str)) + list(
            np.asarray(syn_labels).astype(str)
        )
        le.fit(y_all)
        real_y_enc = le.transform(np.asarray(y_train).astype(str))
        syn_y_enc = le.transform(np.asarray(syn_labels).astype(str))

    evaluator = Evaluator(random_state=cfg["project"]["seed"])
    results = evaluator.evaluate_all(
        real_train=real_train.drop(columns=[target_col]),
        real_test=real_test.drop(columns=[target_col]),
        synthetic=syn_eval.drop(columns=[target_col]),
        target_col=target_col,
        real_train_labels=np.asarray(y_train),
        real_test_labels=np.asarray(y_test),
        synthetic_labels=np.asarray(syn_labels),
        cat_cols=preprocessor.info.cat_features,
        task=task,
    )
    evaluator.print_report()

    meta = {
        "dataset": dataset_name,
        "method": method,
        "privacy": privacy_report or {},
        **(extra_meta or {}),
    }
    metrics_path = out_dir / "metrics.json"
    evaluator.save_results(metrics_path, extra=meta)

    summary = evaluator.summary_row(dataset_name, method=method)
    if privacy_report and privacy_report.get("enabled"):
        summary["epsilon"] = privacy_report.get("current_epsilon")
        summary["target_epsilon"] = privacy_report.get("target_epsilon")
    summary_path = out_dir / "summary_row.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Append to master CSV
    master_csv = out_dir.parent / "results_master.csv"
    fieldnames = list(summary.keys())
    write_header = not master_csv.exists()
    with open(master_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(summary)

    priv = results.get("privacy", {})
    fig_paths = generate_all_figures(
        out_dir=out_dir,
        dataset=dataset_name,
        real_train=real_train,
        syn_df=syn_eval,
        target_col=target_col,
        num_features=preprocessor.info.num_features,
        cat_features=preprocessor.info.cat_features,
        train_loss=train_loss,
        distill_loss=distill_loss,
        roc_curves=evaluator.get_roc_curves(),
        dcr_values=priv.get("dcr_values"),
        real_y=real_y_enc,
        syn_y=syn_y_enc,
        real_X_num=real_X_num,
        syn_X_num=syn_X_num,
    )

    with open(out_dir / "figure_paths.json", "w", encoding="utf-8") as f:
        json.dump(fig_paths, f, indent=2)

    syn_eval.to_csv(out_dir / f"synthetic_{dataset_name}.csv", index=False)
    print(f"\nArtefacts saved under: {out_dir}")
    print(f"  metrics: {metrics_path}")
    print(f"  summary: {summary_path}")
    print(f"  master:  {master_csv}")
    print(f"  figures: {len(fig_paths)} files")
    return summary, results


def main():
    parser = argparse.ArgumentParser(
        description="Run DGD-TabPA experiment end-to-end"
    )
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--distill-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--mask-ratio", type=float, default=1.0)
    parser.add_argument(
        "--method",
        type=str,
        default="dgd_tabpa",
        choices=["dgd_tabpa", "smote", "diffusion_sample"],
        help="dgd_tabpa=train+distill; smote=baseline; diffusion_sample=sample from model only",
    )
    parser.add_argument(
        "--ablation",
        type=str,
        default="none",
        choices=[
            "none",
            "mlp_denoiser",
            "no_attention",
            "minmax",
            "raw_space",
        ],
        help="Ablation study variant",
    )
    parser.add_argument("--privacy", action="store_true", help="Enable DP-SGD")
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Reuse checkpoint in --save-dir instead of training",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Output directory (default: outputs/experiments/<run_id>)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Subsample large datasets (default from config; covertype uses 50000)",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run tag used in output path",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset_name = args.dataset or cfg["data"]["dataset"]
    epochs = args.epochs or cfg["training"]["epochs"]
    if args.distill_epochs is not None:
        cfg["distillation"]["distill_epochs"] = args.distill_epochs
    batch_size = args.batch_size or cfg["training"]["batch_size"]
    lr = args.lr or cfg["training"]["learning_rate"]
    epsilon = (
        args.epsilon
        if args.epsilon is not None
        else cfg.get("privacy", {}).get("epsilon", 8.0)
    )
    privacy_enabled = args.privacy or cfg.get("privacy", {}).get("enabled", False)
    ablation = args.ablation

    set_seed(cfg["project"]["seed"])
    device = get_device(cfg["training"]["device"])

    dataset_cfg = cfg["data"]["datasets"][dataset_name]
    task = dataset_cfg.get("task", "classification")

    data_path = Path(cfg["data"]["data_dir"]) / f"{dataset_name}.csv"
    if not data_path.exists():
        print(
            f"Dataset not found at {data_path}. "
            f"Run: python scripts/download_data.py --dataset {dataset_name}"
        )
        sys.exit(1)

    # Default subsample for very large sets (covertype ~580k rows)
    max_samples = args.max_samples
    if max_samples is None:
        max_samples = cfg.get("data", {}).get("max_samples", {}).get(dataset_name)
    if max_samples is None and dataset_name == "covertype":
        max_samples = 50000

    run_tag = args.run_id or (
        f"{dataset_name}_{args.method}"
        + (f"_{ablation}" if ablation != "none" else "")
        + (f"_eps{epsilon}" if privacy_enabled else "")
    )
    out_dir = Path(args.save_dir) if args.save_dir else Path("outputs") / "experiments" / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Run: {run_tag}")
    print(f"Task: {task}")
    print(f"Output: {out_dir}")

    numerical_transform = (
        "minmax"
        if ablation == "minmax"
        else cfg.get("preprocessing", {}).get("numerical_transform", "quantile_normal")
    )
    preprocessor = TabularPreprocessor(
        random_state=cfg["project"]["seed"],
        numerical_transform=numerical_transform,
        task=task,
    )
    X_train, X_test, y_train, y_test = preprocessor.load_dataset(
        name=dataset_name,
        filepath=str(data_path),
        target_col=dataset_cfg["target"],
        test_size=cfg["data"]["test_size"],
        max_samples=max_samples,
    )
    X_train_t, y_train_t = preprocessor.fit_transform(X_train, y_train)
    X_test_t, y_test_t = preprocessor.transform(X_test, y_test)
    info = preprocessor.info
    target_col = info.target_col

    print(f"\nDataset: {info.name}")
    print(f"  Task: {task}")
    print(f"  Features: num={len(info.num_features)} cat={len(info.cat_features)}")
    print(f"  Dim={info.total_dim}, Cond. classes/bins={info.num_classes}")
    print(f"  Train={len(X_train_t)}, Test={len(X_test_t)}")
    print(f"  Numerical transform: {numerical_transform}")

    with open(out_dir / f"preprocessor_{dataset_name}.pkl", "wb") as f:
        pickle.dump(preprocessor, f)

    train_loss = None
    distill_loss = None
    privacy_report = {"enabled": False}
    method_name = args.method if ablation == "none" else f"{args.method}_{ablation}"

    # ── SMOTE / regression noise baseline (no diffusion training) ──
    if args.method == "smote":
        syn_df = generate_smote_synthetic(
            X_train,
            y_train,
            target_col=target_col,
            n_synthetic=cfg["distillation"]["num_synthetic"],
            random_state=cfg["project"]["seed"],
            task=task,
        )
        syn_labels = syn_df[target_col].values
        evaluate_and_plot(
            cfg,
            dataset_name,
            method_name,
            preprocessor,
            X_train,
            X_test,
            y_train,
            y_test,
            syn_df.drop(columns=[target_col]),
            syn_labels,
            target_col,
            out_dir,
            privacy_report=privacy_report,
            extra_meta={"ablation": ablation, "task": task},
            task=task,
        )
        return

    # ── Train or load diffusion ──
    if args.skip_train:
        ckpt_path = out_dir / f"best_model_{dataset_name}.pt"
        if not ckpt_path.exists():
            # also look under outputs/
            alt = Path("outputs") / f"best_model_{dataset_name}.pt"
            ckpt_path = alt if alt.exists() else ckpt_path
        if not ckpt_path.exists():
            print(f"No checkpoint at {ckpt_path}. Train first.")
            sys.exit(1)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        denoiser = build_denoiser(cfg, info, ablation)
        diffusion = GaussianDiffusion(
            denoiser=denoiser,
            num_timesteps=cfg["diffusion"]["num_timesteps"],
            beta_start=cfg["diffusion"]["beta_start"],
            beta_end=cfg["diffusion"]["beta_end"],
            beta_schedule=cfg["diffusion"]["beta_schedule"],
        ).to(device)
        diffusion.load_state_dict(ckpt["model_state_dict"])
        final_path = out_dir / f"final_model_{dataset_name}.pt"
        if final_path.exists():
            final = torch.load(final_path, map_location=device, weights_only=False)
            train_loss = final.get("loss_history")
        print(f"Loaded checkpoint: {ckpt_path}")
    else:
        diffusion, train_loss, privacy_report, _ = train_diffusion(
            cfg=cfg,
            info=info,
            X_train_t=X_train_t,
            y_train_t=y_train_t,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            mask_ratio=args.mask_ratio,
            ablation=ablation,
            privacy_enabled=privacy_enabled,
            epsilon=epsilon,
            save_dir=out_dir,
            dataset_name=dataset_name,
        )

    diffusion.eval()

    # ── Generate synthetic set ──
    if args.method == "diffusion_sample":
        n = cfg["distillation"]["num_synthetic"]
        labels_per = n // info.num_classes
        gen_labels = torch.cat(
            [
                torch.full((labels_per,), c, dtype=torch.long)
                for c in range(info.num_classes)
            ]
        )[:n].to(device)
        with torch.no_grad():
            generated = diffusion.sample(
                labels=gen_labels,
                shape=(len(gen_labels), info.total_dim),
                device=device,
                sampling_steps=cfg["diffusion"].get("sampling_steps", 20),
            )
        syn_df = preprocessor.inverse_transform(generated)
        syn_labels = preprocessor.inverse_transform_labels(gen_labels.cpu())
    else:
        # Default: distillation
        raw_space = ablation == "raw_space"
        syn_out, syn_labels, distill_loss = run_distillation(
            cfg,
            diffusion,
            X_train_t,
            y_train_t,
            device,
            info,
            raw_space=raw_space,
            X_train_df=X_train,
            y_train_raw=y_train,
        )
        if raw_space:
            syn_df = syn_out
            # syn_labels already original-space names
        else:
            syn_df = preprocessor.inverse_transform(syn_out)
            syn_labels = preprocessor.inverse_transform_labels(syn_labels.cpu())

        save_payload = {"syn_labels": np.asarray(syn_labels), "distill_loss": distill_loss}
        if torch.is_tensor(syn_out):
            save_payload["syn_data"] = syn_out.detach().cpu()
        torch.save(save_payload, out_dir / f"distilled_{dataset_name}.pt")

    evaluate_and_plot(
        cfg,
        dataset_name,
        method_name,
        preprocessor,
        X_train,
        X_test,
        y_train,
        y_test,
        syn_df,
        syn_labels,
        target_col,
        out_dir,
        train_loss=train_loss,
        distill_loss=distill_loss,
        privacy_report=privacy_report,
        extra_meta={
            "ablation": ablation,
            "epochs": epochs,
            "privacy_enabled": privacy_enabled,
            "epsilon_target": epsilon if privacy_enabled else None,
            "numerical_transform": numerical_transform,
            "task": task,
        },
        task=task,
    )


if __name__ == "__main__":
    main()
