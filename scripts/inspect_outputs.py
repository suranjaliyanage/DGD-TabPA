"""
Inspect training outputs: model checkpoints, loss history, and preprocessor.
Generates sample synthetic data and shows quick quality metrics.

Usage:
    python scripts/inspect_outputs.py
    python scripts/inspect_outputs.py --dataset adult --n-samples 200
"""

import sys
import pickle
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.transformer import TabularTransformerDenoiser
from src.models.diffusion import GaussianDiffusion


def main():
    parser = argparse.ArgumentParser(description="Inspect DGD-TabPA training outputs")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--dataset", type=str, default="adult")
    parser.add_argument("--n-samples", type=int, default=200)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    ds = args.dataset

    # ── 1. Load and inspect the best checkpoint ──
    best_path = out_dir / f"best_model_{ds}.pt"
    final_path = out_dir / f"final_model_{ds}.pt"
    prep_path = out_dir / f"preprocessor_{ds}.pkl"

    print("=" * 60)
    print("  DGD-TabPA OUTPUT INSPECTION")
    print("=" * 60)

    if not best_path.exists():
        print(f"\nNo best model found at {best_path}. Train first.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_ckpt = torch.load(best_path, map_location=device, weights_only=False)
    print(f"\n[1] BEST MODEL CHECKPOINT ({best_path.name})")
    print(f"    File size: {best_path.stat().st_size / 1e6:.1f} MB")
    print(f"    Saved at epoch: {best_ckpt['epoch']}")
    print(f"    Best loss: {best_ckpt['loss']:.6f}")

    ds_info = best_ckpt["dataset_info"]
    print(f"\n    Dataset info:")
    print(f"      Name: {ds_info['name']}")
    print(f"      Total dim: {ds_info['total_dim']}")
    print(f"      Classes: {ds_info['num_classes']}")
    print(f"      Num features: {ds_info['num_features']}")
    print(f"      Cat features: {ds_info['cat_features']}")

    cfg = best_ckpt["config"]
    print(f"\n    Model config:")
    print(f"      d_model: {cfg['model']['d_model']}")
    print(f"      n_heads: {cfg['model']['n_heads']}")
    print(f"      encoder layers: {cfg['model']['n_encoder_layers']}")
    print(f"      decoder layers: {cfg['model']['n_decoder_layers']}")
    print(f"      timesteps: {cfg['diffusion']['num_timesteps']}")

    # ── 2. Loss history from final model ──
    if final_path.exists():
        final_ckpt = torch.load(final_path, map_location=device, weights_only=False)
        print(f"\n[2] FINAL MODEL ({final_path.name})")
        print(f"    File size: {final_path.stat().st_size / 1e6:.1f} MB")
        print(f"    Total epochs: {final_ckpt['epoch']}")

        loss_history = final_ckpt.get("loss_history", [])
        if loss_history:
            print(f"    Loss history ({len(loss_history)} epochs):")
            print(f"      First: {loss_history[0]:.6f}")
            print(f"      Last:  {loss_history[-1]:.6f}")
            print(f"      Min:   {min(loss_history):.6f}")
            print(f"      Max:   {max(loss_history):.6f}")

            # Save loss plot
            sns.set_theme(style="whitegrid")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(range(1, len(loss_history) + 1), loss_history, color="steelblue", linewidth=1.5)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("MSE Loss")
            ax.set_title(f"DGD-TabPA Training Loss ({ds})")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plot_path = out_dir / f"training_loss_{ds}.png"
            fig.savefig(plot_path, dpi=150)
            plt.close()
            print(f"\n    Loss plot saved to: {plot_path}")

    # ── 3. Preprocessor ──
    if prep_path.exists():
        with open(prep_path, "rb") as f:
            preprocessor = pickle.load(f)
        info = preprocessor.info
        print(f"\n[3] PREPROCESSOR ({prep_path.name})")
        print(f"    File size: {prep_path.stat().st_size / 1e3:.1f} KB")
        print(f"    Numerical features ({info.num_dim}): {info.num_features}")
        print(f"    Categorical features ({len(info.cat_features)}): {info.cat_features}")
        print(f"    Total encoded dim: {info.total_dim}")
        print(f"    Classes: {info.num_classes}")
    else:
        preprocessor = None

    # ── 4. Generate samples and inspect ──
    print(f"\n[4] GENERATING {args.n_samples} SYNTHETIC SAMPLES...")

    denoiser = TabularTransformerDenoiser(
        total_dim=ds_info["total_dim"],
        num_classes=ds_info["num_classes"],
        d_model=cfg["model"]["d_model"],
        n_heads=cfg["model"]["n_heads"],
        n_encoder_layers=cfg["model"]["n_encoder_layers"],
        n_decoder_layers=cfg["model"]["n_decoder_layers"],
        d_ff=cfg["model"]["d_ff"],
        dropout=0.0,
    )

    diffusion = GaussianDiffusion(
        denoiser=denoiser,
        num_timesteps=cfg["diffusion"]["num_timesteps"],
        beta_start=cfg["diffusion"]["beta_start"],
        beta_end=cfg["diffusion"]["beta_end"],
        beta_schedule=cfg["diffusion"]["beta_schedule"],
    ).to(device)

    diffusion.load_state_dict(best_ckpt["model_state_dict"])
    diffusion.eval()

    n = args.n_samples
    labels_per_class = n // ds_info["num_classes"]
    gen_labels = torch.cat([
        torch.full((labels_per_class,), c, dtype=torch.long)
        for c in range(ds_info["num_classes"])
    ])[:n].to(device)

    sampling_steps = cfg["diffusion"].get("sampling_steps", 20)
    print(f"    Sampling steps: {sampling_steps}")

    with torch.no_grad():
        generated = diffusion.sample(
            labels=gen_labels,
            shape=(n, ds_info["total_dim"]),
            device=device,
            sampling_steps=sampling_steps,
        )

    gen_np = generated.cpu().numpy()

    print(f"\n    Generated tensor shape: {generated.shape}")
    print(f"    Mean: {gen_np.mean():.4f}")
    print(f"    Std:  {gen_np.std():.4f}")
    print(f"    Min:  {gen_np.min():.4f}")
    print(f"    Max:  {gen_np.max():.4f}")
    print(f"    NaN count: {np.isnan(gen_np).sum()}")

    # ── 5. Inverse transform to readable format ──
    if preprocessor is not None:
        print(f"\n[5] INVERSE-TRANSFORMED SYNTHETIC DATA (first 10 rows):")
        syn_df = preprocessor.inverse_transform(generated)
        syn_labels = preprocessor.inverse_transform_labels(gen_labels.cpu())
        syn_df[info.target_col] = syn_labels

        print(syn_df.head(10).to_string())

        csv_path = out_dir / f"synthetic_sample_{ds}.csv"
        syn_df.to_csv(csv_path, index=False)
        print(f"\n    Full synthetic sample saved to: {csv_path}")

        # Distribution comparison plot
        num_cols = info.num_features[:min(6, len(info.num_features))]
        if num_cols:
            fig, axes = plt.subplots(2, len(num_cols), figsize=(4 * len(num_cols), 7))
            if len(num_cols) == 1:
                axes = axes.reshape(2, 1)
            fig.suptitle(f"Real vs Synthetic Distributions ({ds})", fontsize=14)

            import pandas as pd
            real_path = Path(cfg["data"]["data_dir"]) / f"{ds}.csv"
            if real_path.exists():
                real_df = pd.read_csv(real_path)
                for i, col in enumerate(num_cols):
                    if col in real_df.columns and col in syn_df.columns:
                        axes[0, i].hist(real_df[col].dropna().values, bins=40,
                                        alpha=0.7, color="steelblue", density=True)
                        axes[0, i].set_title(f"{col} (Real)")
                        axes[1, i].hist(syn_df[col].dropna().values, bins=40,
                                        alpha=0.7, color="coral", density=True)
                        axes[1, i].set_title(f"{col} (Synthetic)")

                plt.tight_layout()
                dist_path = out_dir / f"distributions_{ds}.png"
                fig.savefig(dist_path, dpi=150)
                plt.close()
                print(f"    Distribution plot saved to: {dist_path}")

    print("\n" + "=" * 60)
    print("  INSPECTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
