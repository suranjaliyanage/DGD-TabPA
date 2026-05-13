"""
Module 5 (Part 2): Flask API Endpoint for DGD-TabPA

Provides a REST API for:
  - POST /distill: Upload CSV, run distillation pipeline, return results
  - GET /health: Service health check
  - POST /generate: Generate synthetic samples from a trained model

Follows a no-permanent-storage design for GDPR/CCPA compliance.
"""

import os
import sys
import io
import json
import yaml
import torch
import pickle
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path
from flask import Flask, request, jsonify, send_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.preprocessing import TabularPreprocessor
from src.models.transformer import TabularTransformerDenoiser
from src.models.diffusion import GaussianDiffusion
from src.distillation.loop import DistillationLoop
from src.evaluation.evaluate import Evaluator
from src.privacy.dp_sgd import DPSGDWrapper

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = BASE_DIR / "config" / "default.yaml"
OUTPUTS_DIR = BASE_DIR / "outputs"


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "project": "DGD-TabPA",
        "gpu_available": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    })


@app.route("/distill", methods=["POST"])
def distill():
    """
    Full distillation pipeline endpoint.

    Expects:
        - CSV file upload (field name: 'file')
        - Form data: target_col, num_synthetic (optional), dp_enabled (optional)

    Returns:
        JSON with distilled data and evaluation metrics.
    """
    if "file" not in request.files:
        return jsonify({"error": "No CSV file uploaded. Use field name 'file'."}), 400

    file = request.files["file"]
    target_col = request.form.get("target_col")
    if not target_col:
        return jsonify({"error": "target_col is required"}), 400

    num_synthetic = int(request.form.get("num_synthetic", 100))
    dp_enabled = request.form.get("dp_enabled", "false").lower() == "true"

    try:
        cfg = load_config()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Read CSV into memory (no permanent storage)
        df = pd.read_csv(io.StringIO(file.read().decode("utf-8")))

        # Preprocess
        preprocessor = TabularPreprocessor(random_state=cfg["project"]["seed"])
        X_train, X_test, y_train, y_test = preprocessor.load_dataset(
            name="upload",
            filepath=None,
            target_col=target_col,
            test_size=cfg["data"]["test_size"],
        ) if False else (None, None, None, None)

        # Direct processing for uploaded data
        preprocessor = TabularPreprocessor(random_state=cfg["project"]["seed"])

        num_features = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in num_features:
            num_features.remove(target_col)
        cat_features = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if target_col in cat_features:
            cat_features.remove(target_col)

        from sklearn.model_selection import train_test_split

        y = df[target_col].values
        X = df[num_features + cat_features]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        from src.preprocessing.pipeline import DatasetInfo

        preprocessor.info = DatasetInfo(
            name="upload",
            num_features=num_features,
            cat_features=cat_features,
            target_col=target_col,
            num_classes=len(np.unique(y)),
            num_dim=len(num_features),
            cat_dim=0,
            total_dim=0,
        )

        X_train_t, y_train_t = preprocessor.fit_transform(X_train, y_train)
        X_test_t, y_test_t = preprocessor.transform(X_test, y_test)
        info = preprocessor.info

        # Build model
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
        ).to(device)

        # Quick training (reduced epochs for API)
        from torch.utils.data import DataLoader, TensorDataset
        train_ds = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, drop_last=True)

        optimizer = torch.optim.AdamW(diffusion.parameters(), lr=0.001)

        api_epochs = min(20, cfg["training"]["epochs"])

        dp_wrapper = DPSGDWrapper(
            enabled=dp_enabled,
            target_epsilon=cfg["privacy"]["epsilon"],
            target_delta=cfg["privacy"]["delta"],
            max_grad_norm=cfg["privacy"]["max_grad_norm"],
        )
        diffusion, optimizer, train_loader = dp_wrapper.attach(
            model=diffusion,
            optimizer=optimizer,
            data_loader=train_loader,
            epochs=api_epochs,
        )

        diffusion.train()
        for epoch in range(api_epochs):
            for X_b, y_b in train_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                t = torch.randint(0, diff_cfg["num_timesteps"], (X_b.size(0),), device=device)
                loss = diffusion.compute_loss(X_b, t, y_b, mask_ratio=1.0)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Distillation
        distiller = DistillationLoop(
            diffusion_model=diffusion,
            total_dim=info.total_dim,
            num_classes=info.num_classes,
            num_synthetic=num_synthetic,
            distill_epochs=50,
            device=device,
        )

        syn_data, syn_labels, _ = distiller.distill(X_train_t, y_train_t)

        # Inverse transform
        syn_df = preprocessor.inverse_transform(syn_data)
        syn_label_names = preprocessor.inverse_transform_labels(syn_labels)
        syn_df[target_col] = syn_label_names

        real_train_df = X_train.copy()
        real_train_df[target_col] = y_train
        real_test_df = X_test.copy()
        real_test_df[target_col] = y_test

        # Evaluate
        evaluator = Evaluator()
        metrics = evaluator.evaluate_all(
            real_train=real_train_df,
            real_test=real_test_df,
            synthetic=syn_df,
            target_col=target_col,
        )

        # Return results (no data stored)
        result = {
            "status": "success",
            "dataset_info": {
                "original_rows": len(df),
                "distilled_rows": num_synthetic,
                "features": info.num_features + info.cat_features,
                "target": target_col,
            },
            "privacy": dp_wrapper.get_privacy_report(),
            "evaluation": metrics,
            "synthetic_sample": syn_df.head(5).to_dict(orient="records"),
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/generate", methods=["POST"])
def generate():
    """
    Generate synthetic samples from a pre-trained model.

    Expects JSON: { "dataset": "adult", "n_samples": 100 }
    """
    data = request.get_json()
    dataset_name = data.get("dataset", "adult")
    n_samples = data.get("n_samples", 100)

    model_path = OUTPUTS_DIR / f"best_model_{dataset_name}.pt"
    preprocessor_path = OUTPUTS_DIR / f"preprocessor_{dataset_name}.pkl"

    if not model_path.exists():
        return jsonify({"error": f"No trained model found for '{dataset_name}'"}), 404

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        cfg = checkpoint["config"]
        ds_info = checkpoint["dataset_info"]

        with open(preprocessor_path, "rb") as f:
            preprocessor = pickle.load(f)

        model_cfg = cfg["model"]
        denoiser = TabularTransformerDenoiser(
            total_dim=ds_info["total_dim"],
            num_classes=ds_info["num_classes"],
            d_model=model_cfg["d_model"],
            n_heads=model_cfg["n_heads"],
            n_encoder_layers=model_cfg["n_encoder_layers"],
            n_decoder_layers=model_cfg["n_decoder_layers"],
            d_ff=model_cfg["d_ff"],
            dropout=model_cfg["dropout"],
        )

        diffusion = GaussianDiffusion(
            denoiser=denoiser,
            num_timesteps=cfg["diffusion"]["num_timesteps"],
        ).to(device)

        diffusion.load_state_dict(checkpoint["model_state_dict"])
        diffusion.eval()

        # Generate balanced labels
        labels_per_class = n_samples // ds_info["num_classes"]
        labels = []
        for c in range(ds_info["num_classes"]):
            labels.extend([c] * labels_per_class)
        remaining = n_samples - len(labels)
        labels.extend([0] * remaining)
        labels_t = torch.tensor(labels, dtype=torch.long, device=device)

        generated = diffusion.sample(
            labels=labels_t,
            shape=(n_samples, ds_info["total_dim"]),
            device=device,
            sampling_steps=cfg["diffusion"].get("sampling_steps", 20),
        )

        syn_df = preprocessor.inverse_transform(generated)
        syn_labels = preprocessor.inverse_transform_labels(labels_t.cpu())
        syn_df[preprocessor.info.target_col] = syn_labels

        csv_buffer = io.StringIO()
        syn_df.to_csv(csv_buffer, index=False)

        return send_file(
            io.BytesIO(csv_buffer.getvalue().encode()),
            mimetype="text/csv",
            as_attachment=True,
            download_name=f"synthetic_{dataset_name}_{n_samples}.csv",
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    cfg = load_config()
    api_cfg = cfg.get("api", {})
    app.run(
        host=api_cfg.get("host", "0.0.0.0"),
        port=api_cfg.get("port", 5000),
        debug=api_cfg.get("debug", True),
    )
