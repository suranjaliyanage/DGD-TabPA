"""
Download benchmark datasets for the DGD-TabPA framework.
Fetches Adult Income and Indian Liver Patient datasets from UCI ML Repository.
"""

import os
import argparse
import pandas as pd
import requests
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

DATASETS = {
    "adult": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
        "test_url": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
        "columns": [
            "age", "workclass", "fnlwgt", "education", "education_num",
            "marital_status", "occupation", "relationship", "race", "sex",
            "capital_gain", "capital_loss", "hours_per_week",
            "native_country", "income"
        ],
        "filename": "adult.csv",
    },
    "ilpd": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00225/Indian%20Liver%20Patient%20Dataset%20(ILPD).csv",
        "columns": [
            "age", "gender", "total_bilirubin", "direct_bilirubin",
            "alkaline_phosphotase", "alamine_aminotransferase",
            "aspartate_aminotransferase", "total_proteins",
            "albumin", "albumin_globulin_ratio", "is_patient"
        ],
        "filename": "ilpd.csv",
    },
}


def download_file(url: str, filepath: Path) -> bool:
    print(f"  Downloading from {url}")
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        filepath.write_bytes(resp.content)
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def prepare_adult(data_dir: Path):
    print("\n[1/2] Preparing Adult Income dataset...")
    meta = DATASETS["adult"]
    raw_train = data_dir / "adult_raw_train.data"
    raw_test = data_dir / "adult_raw_test.data"

    if not download_file(meta["url"], raw_train):
        return
    if not download_file(meta["test_url"], raw_test):
        return

    df_train = pd.read_csv(raw_train, header=None, names=meta["columns"],
                           na_values=" ?", skipinitialspace=True)
    df_test = pd.read_csv(raw_test, header=None, names=meta["columns"],
                          na_values=" ?", skipinitialspace=True, skiprows=1)

    # Clean test labels (they have trailing periods)
    df_test["income"] = df_test["income"].str.rstrip(".")

    df = pd.concat([df_train, df_test], ignore_index=True)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    out_path = data_dir / meta["filename"]
    df.to_csv(out_path, index=False)
    print(f"  Saved {len(df)} rows to {out_path}")

    raw_train.unlink(missing_ok=True)
    raw_test.unlink(missing_ok=True)


def prepare_ilpd(data_dir: Path):
    print("\n[2/2] Preparing Indian Liver Patient dataset...")
    meta = DATASETS["ilpd"]
    raw_file = data_dir / "ilpd_raw.csv"

    if not download_file(meta["url"], raw_file):
        return

    df = pd.read_csv(raw_file, header=None, names=meta["columns"])
    df.dropna(inplace=True)
    # Remap target: 1 = patient, 2 = not patient -> 1/0
    df["is_patient"] = df["is_patient"].map({1: 1, 2: 0})
    df.reset_index(drop=True, inplace=True)

    out_path = data_dir / meta["filename"]
    df.to_csv(out_path, index=False)
    print(f"  Saved {len(df)} rows to {out_path}")

    raw_file.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Download DGD-TabPA benchmark datasets")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    parser.add_argument("--dataset", type=str, choices=["adult", "ilpd", "all"], default="all")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Data directory: {data_dir}")

    if args.dataset in ("adult", "all"):
        prepare_adult(data_dir)
    if args.dataset in ("ilpd", "all"):
        prepare_ilpd(data_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
