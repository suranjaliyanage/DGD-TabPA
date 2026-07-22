"""
Download benchmark datasets for the DGD-TabPA framework.

Sources prefer public UCI / OpenML / scikit-learn mirrors so Kaggle API
credentials are not required. Original catalogue links are recorded in
config/default.yaml.
"""

from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

import pandas as pd
import requests
from sklearn.datasets import fetch_california_housing, fetch_covtype, fetch_openml


DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# Canonical dataset keys match `{name}.csv` under data/raw and config entries.
DATASET_CHOICES = [
    "adult",
    "churn",
    "credit",
    "covertype",
    "cvd",
    "hcv",
    "ilpd",
    "diabetes",
    "california_housing",
    "king_county",
    "all",
]


def download_file(url: str, filepath: Path, timeout: int = 120) -> bool:
    print(f"  Downloading from {url}")
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        filepath.write_bytes(resp.content)
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def save_csv(df: pd.DataFrame, out_path: Path, label: str) -> None:
    df = df.copy()
    df.reset_index(drop=True, inplace=True)
    df.to_csv(out_path, index=False)
    print(f"  Saved {len(df):,} rows x {df.shape[1]} cols -> {out_path} ({label})")


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        str(c).strip().replace(" ", "_").replace("/", "_").replace("&", "and")
        for c in df.columns
    ]
    return df


# ---------------------------------------------------------------------------
# Industrial benchmarks
# ---------------------------------------------------------------------------

def prepare_adult(data_dir: Path) -> None:
    print("\n[adult] Adult Income (UCI)...")
    columns = [
        "age", "workclass", "fnlwgt", "education", "education_num",
        "marital_status", "occupation", "relationship", "race", "sex",
        "capital_gain", "capital_loss", "hours_per_week",
        "native_country", "income",
    ]
    train_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    test_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
    raw_train = data_dir / "adult_raw_train.data"
    raw_test = data_dir / "adult_raw_test.data"

    if not download_file(train_url, raw_train):
        return
    if not download_file(test_url, raw_test):
        return

    df_train = pd.read_csv(
        raw_train, header=None, names=columns, na_values=" ?", skipinitialspace=True
    )
    df_test = pd.read_csv(
        raw_test, header=None, names=columns, na_values=" ?",
        skipinitialspace=True, skiprows=1,
    )
    df_test["income"] = df_test["income"].str.rstrip(".")

    df = pd.concat([df_train, df_test], ignore_index=True)
    df.dropna(inplace=True)
    save_csv(df, data_dir / "adult.csv", "classification / income")

    raw_train.unlink(missing_ok=True)
    raw_test.unlink(missing_ok=True)


def prepare_churn(data_dir: Path) -> None:
    print("\n[churn] Bank Churn Modelling (Kaggle mirror)...")
    url = (
        "https://raw.githubusercontent.com/sharmaroshan/"
        "Churn-Modelling-Dataset/master/Churn_Modelling.csv"
    )
    try:
        df = pd.read_csv(url)
    except Exception as e:
        print(f"  ERROR: {e}")
        return

    drop_cols = [c for c in ("RowNumber", "CustomerId", "Surname") if c in df.columns]
    df = df.drop(columns=drop_cols)
    df.dropna(inplace=True)
    save_csv(df, data_dir / "churn.csv", "classification / Exited")


def prepare_credit(data_dir: Path) -> None:
    print("\n[credit] Default of Credit Card Clients (UCI / CSV mirror)...")
    url = (
        "https://raw.githubusercontent.com/MatteoM95/"
        "Default-of-Credit-Card-Clients-Dataset-Analisys/main/dataset/"
        "default_of_credit_card_clients.csv"
    )
    try:
        df = pd.read_csv(url)
    except Exception as e:
        print(f"  ERROR: {e}")
        return

    if "ID" in df.columns:
        df = df.drop(columns=["ID"])
    target_aliases = ["default payment next month", "default.payment.next.month"]
    for alias in target_aliases:
        if alias in df.columns:
            df = df.rename(columns={alias: "default"})
            break
    df.dropna(inplace=True)
    save_csv(df, data_dir / "credit.csv", "classification / default")


def prepare_covertype(data_dir: Path) -> None:
    print("\n[covertype] Forest Cover Type (UCI via scikit-learn)...")
    print("  Fetching (large ~581k rows; may take a minute)...")
    try:
        bunch = fetch_covtype(as_frame=True)
        df = bunch.frame.copy()
    except Exception as e:
        print(f"  ERROR: {e}")
        return

    if "Cover_Type" not in df.columns and "target" in df.columns:
        df = df.rename(columns={"target": "Cover_Type"})
    df.dropna(inplace=True)
    save_csv(df, data_dir / "covertype.csv", "classification / Cover_Type")


# ---------------------------------------------------------------------------
# Medical benchmarks
# ---------------------------------------------------------------------------

def prepare_cvd(data_dir: Path) -> None:
    print("\n[cvd] Cardiovascular Disease (OpenML 45547 / Kaggle equivalent)...")
    try:
        bunch = fetch_openml(data_id=45547, as_frame=True, parser="auto")
        df = bunch.frame.copy()
    except Exception as e:
        print(f"  ERROR: {e}")
        return

    df = clean_columns(df)
    if "id" in df.columns:
        df = df.drop(columns=["id"])
    # OpenML stores age in days; convert to years for interpretability.
    if "age" in df.columns and df["age"].median() > 200:
        df["age"] = (df["age"] / 365.25).round().astype(int)
    # Persist categoricals as plain values for CSV round-trip.
    for col in df.columns:
        if isinstance(df[col].dtype, pd.CategoricalDtype):
            df[col] = df[col].astype(str)
    df.dropna(inplace=True)
    save_csv(df, data_dir / "cvd.csv", "classification / cardio")


def prepare_hcv(data_dir: Path) -> None:
    print("\n[hcv] Hepatitis C Virus — Egyptian patients (UCI 503)...")
    url = (
        "https://archive.ics.uci.edu/static/public/503/"
        "hepatitis+c+virus+hcv+for+egyptian+patients.zip"
    )
    try:
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            csv_name = next(
                n for n in zf.namelist()
                if n.lower().endswith(".csv") and "disc" not in n.lower()
            )
            with zf.open(csv_name) as f:
                df = pd.read_csv(f)
    except Exception as e:
        print(f"  ERROR: {e}")
        return

    df = clean_columns(df)
    # Canonical target name for config / training.
    for col in list(df.columns):
        normalized = col.lower().replace(" ", "_")
        if normalized in ("baselinehistological_staging", "baseline_histological_staging"):
            df = df.rename(columns={col: "histological_staging"})
            break
    df.dropna(inplace=True)
    save_csv(df, data_dir / "hcv.csv", "classification / histological_staging")


def prepare_ilpd(data_dir: Path) -> None:
    print("\n[ilpd] Indian Liver Patient (UCI)...")
    url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00225/"
        "Indian%20Liver%20Patient%20Dataset%20(ILPD).csv"
    )
    columns = [
        "age", "gender", "total_bilirubin", "direct_bilirubin",
        "alkaline_phosphotase", "alamine_aminotransferase",
        "aspartate_aminotransferase", "total_proteins",
        "albumin", "albumin_globulin_ratio", "is_patient",
    ]
    raw_file = data_dir / "ilpd_raw.csv"
    if not download_file(url, raw_file):
        return

    df = pd.read_csv(raw_file, header=None, names=columns)
    df.dropna(inplace=True)
    df["is_patient"] = df["is_patient"].map({1: 1, 2: 0})
    save_csv(df, data_dir / "ilpd.csv", "classification / is_patient")
    raw_file.unlink(missing_ok=True)


def prepare_diabetes(data_dir: Path) -> None:
    print("\n[diabetes] Pima Indians Diabetes (UCI / Kaggle equivalent)...")
    url = (
        "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
        "pima-indians-diabetes.data.csv"
    )
    columns = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome",
    ]
    try:
        df = pd.read_csv(url, header=None, names=columns)
    except Exception as e:
        print(f"  ERROR: {e}")
        return

    df.dropna(inplace=True)
    save_csv(df, data_dir / "diabetes.csv", "classification / Outcome")


# ---------------------------------------------------------------------------
# Regression benchmarks
# ---------------------------------------------------------------------------

def prepare_california_housing(data_dir: Path) -> None:
    print("\n[california_housing] California Housing (scikit-learn)...")
    try:
        bunch = fetch_california_housing(as_frame=True)
        df = bunch.frame.copy()
    except Exception as e:
        print(f"  ERROR: {e}")
        return

    df.dropna(inplace=True)
    save_csv(df, data_dir / "california_housing.csv", "regression / MedHouseVal")


def prepare_king_county(data_dir: Path) -> None:
    print("\n[king_county] House Sales in King County (OpenML house_sales)...")
    try:
        bunch = fetch_openml(name="house_sales", version=1, as_frame=True, parser="auto")
        df = bunch.frame.copy()
    except Exception as e:
        print(f"  ERROR: {e}")
        return

    df = clean_columns(df)
    # High-cardinality / non-tabular date string — drop for modelling.
    if "date" in df.columns:
        df = df.drop(columns=["date"])
    for col in df.columns:
        if isinstance(df[col].dtype, pd.CategoricalDtype):
            df[col] = df[col].astype(str)
    df.dropna(inplace=True)
    save_csv(df, data_dir / "king_county.csv", "regression / price")


PREPARERS = {
    "adult": prepare_adult,
    "churn": prepare_churn,
    "credit": prepare_credit,
    "covertype": prepare_covertype,
    "cvd": prepare_cvd,
    "hcv": prepare_hcv,
    "ilpd": prepare_ilpd,
    "diabetes": prepare_diabetes,
    "california_housing": prepare_california_housing,
    "king_county": prepare_king_county,
}


def main():
    parser = argparse.ArgumentParser(description="Download DGD-TabPA benchmark datasets")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    parser.add_argument(
        "--dataset",
        type=str,
        choices=DATASET_CHOICES,
        default="all",
        help="Dataset key, or 'all' for the full suite",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Data directory: {data_dir}")

    keys = list(PREPARERS.keys()) if args.dataset == "all" else [args.dataset]
    for key in keys:
        PREPARERS[key](data_dir)

    print("\nDone.")
    print("Train with: python scripts/train.py --config config/default.yaml --dataset <name>")


if __name__ == "__main__":
    main()
