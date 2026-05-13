"""
Main training entry point for the DGD-TabPA diffusion model.

Usage:
    python scripts/train.py --config config/default.yaml
    python scripts/train.py --config config/default.yaml --dataset ilpd --epochs 50
"""

import sys
import os
import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing import TabularPreprocessor
from src.models.transformer import TabularTransformerDenoiser
from src.models.diffusion import GaussianDiffusion
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


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train DGD-TabPA diffusion model")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--mask-ratio", type=float, default=None,
                        help="Fraction of features to mask. 1.0=full generation, 0.5=mixed")
    parser.add_argument("--save-dir", type=str, default="outputs")
    args = parser.parse_args()

    cfg = load_config(args.config)

    dataset_name = args.dataset or cfg["data"]["dataset"]
    epochs = args.epochs or cfg["training"]["epochs"]
    batch_size = args.batch_size or cfg["training"]["batch_size"]
    lr = args.lr or cfg["training"]["learning_rate"]
    mask_ratio = args.mask_ratio if args.mask_ratio is not None else 1.0

    set_seed(cfg["project"]["seed"])
    device = get_device(cfg["training"]["device"])
    print(f"Device: {device}")

    # --- Data Loading ---
    dataset_cfg = cfg["data"]["datasets"][dataset_name]
    data_path = Path(cfg["data"]["data_dir"]) / f"{dataset_name}.csv"

    if not data_path.exists():
        print(f"Dataset not found at {data_path}. Run: python scripts/download_data.py")
        sys.exit(1)

    preprocessor = TabularPreprocessor(random_state=cfg["project"]["seed"])
    X_train, X_test, y_train, y_test = preprocessor.load_dataset(
        name=dataset_name,
        filepath=str(data_path),
        target_col=dataset_cfg["target"],
        test_size=cfg["data"]["test_size"],
    )

    X_train_t, y_train_t = preprocessor.fit_transform(X_train, y_train)
    X_test_t, y_test_t = preprocessor.transform(X_test, y_test)

    info = preprocessor.info
    print(f"\nDataset: {info.name}")
    print(f"  Numerical features: {len(info.num_features)} -> dim {info.num_dim}")
    print(f"  Categorical features: {len(info.cat_features)} -> dim {info.cat_dim}")
    print(f"  Total input dim: {info.total_dim}")
    print(f"  Classes: {info.num_classes}")
    print(f"  Train: {len(X_train_t)}, Test: {len(X_test_t)}")

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    # --- Model ---
    model_cfg = cfg["model"]
    denoiser = TabularTransformerDenoiser(
        total_dim=info.total_dim,
        num_classes=info.num_classes,
        d_model=model_cfg["d_model"],
        n_heads=model_cfg["n_heads"],
        n_encoder_layers=model_cfg["n_encoder_layers"],
        n_decoder_layers=model_cfg["n_decoder_layers"],
        d_ff=model_cfg["d_ff"],
        dropout=model_cfg["dropout"],
    )

    diff_cfg = cfg["diffusion"]
    diffusion = GaussianDiffusion(
        denoiser=denoiser,
        num_timesteps=diff_cfg["num_timesteps"],
        beta_start=diff_cfg["beta_start"],
        beta_end=diff_cfg["beta_end"],
        beta_schedule=diff_cfg["beta_schedule"],
        loss_type=diff_cfg["loss_type"],
    ).to(device)

    total_params = sum(p.numel() for p in diffusion.parameters())
    print(f"\nModel parameters: {total_params:,}")

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        diffusion.parameters(),
        lr=lr,
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # --- DP-SGD Privacy Guardrail ---
    dp_wrapper = DPSGDWrapper.from_config(cfg)
    diffusion, optimizer, train_loader = dp_wrapper.attach(
        model=diffusion,
        optimizer=optimizer,
        data_loader=train_loader,
        epochs=epochs,
    )

    # --- Training Loop ---
    print(f"\nTraining for {epochs} epochs (mask_ratio={mask_ratio})...")
    best_loss = float("inf")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    loss_history = []

    for epoch in range(1, epochs + 1):
        diffusion.train()
        epoch_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for X_batch, y_batch in pbar:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            t = torch.randint(0, diff_cfg["num_timesteps"], (X_batch.size(0),), device=device)

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

        if epoch % cfg["training"]["log_interval"] == 0 or epoch == 1:
            print(f"  Epoch {epoch}/{epochs} | Loss: {avg_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.2e}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": diffusion.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_loss,
                "config": cfg,
                "dataset_info": {
                    "name": info.name,
                    "total_dim": info.total_dim,
                    "num_classes": info.num_classes,
                    "num_features": info.num_features,
                    "cat_features": info.cat_features,
                },
            }
            torch.save(checkpoint, save_dir / f"best_model_{dataset_name}.pt")

    # Save final model
    torch.save(
        {
            "epoch": epochs,
            "model_state_dict": diffusion.state_dict(),
            "loss_history": loss_history,
            "config": cfg,
        },
        save_dir / f"final_model_{dataset_name}.pt",
    )

    print(f"\nTraining complete. Best loss: {best_loss:.6f}")
    print(f"Models saved to {save_dir}/")

    # --- Privacy Report ---
    privacy_report = dp_wrapper.get_privacy_report()
    if privacy_report["enabled"]:
        print(f"\n[DP-SGD] Privacy Report:")
        print(f"  Final epsilon: {privacy_report['current_epsilon']:.4f}")
        print(f"  Target epsilon: {privacy_report['target_epsilon']}")
        print(f"  Budget remaining: {privacy_report['budget_remaining']:.4f}")
    else:
        print("\n[DP-SGD] Disabled — training ran without differential privacy.")

    # --- Quick Generation Test ---
    print("\nGenerating test samples...")
    diffusion.eval()
    n_gen = min(100, len(X_train_t))
    sample_labels = y_train_t[:n_gen].to(device)
    generated = diffusion.sample(
        labels=sample_labels,
        shape=(n_gen, info.total_dim),
        device=device,
        sampling_steps=diff_cfg.get("sampling_steps", 20),
    )
    print(f"  Generated {n_gen} samples, shape: {generated.shape}")
    print(f"  Mean: {generated.mean():.4f}, Std: {generated.std():.4f}")
    print(f"  Real mean: {X_train_t.mean():.4f}, Real std: {X_train_t.std():.4f}")

    # Save preprocessor for later use
    import pickle
    with open(save_dir / f"preprocessor_{dataset_name}.pkl", "wb") as f:
        pickle.dump(preprocessor, f)
    print(f"  Preprocessor saved to {save_dir}/preprocessor_{dataset_name}.pkl")


if __name__ == "__main__":
    main()
