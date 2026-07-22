"""
Module 1: Heterogeneous Data Ingestion and Preprocessing

Transforms raw tabular CSV data into unified float tensors suitable for the
diffusion model. Applies Gaussian Quantile Transformations to numerical features
and One-Hot Encoding to categorical features.

Classification targets use LabelEncoder.
Regression targets are quantile-binned for conditional generation; continuous
values are kept for evaluation.
"""

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import (
    QuantileTransformer,
    MinMaxScaler,
    OneHotEncoder,
    LabelEncoder,
    KBinsDiscretizer,
)
from sklearn.model_selection import train_test_split
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DatasetInfo:
    """Metadata about a preprocessed dataset, needed for inverse transforms."""
    name: str
    num_features: list
    cat_features: list
    target_col: str
    num_classes: int
    num_dim: int
    cat_dim: int
    total_dim: int
    cat_categories: list = field(default_factory=list)
    original_dtypes: dict = field(default_factory=dict)
    task: str = "classification"  # classification | regression


class TabularPreprocessor:
    """
    Preprocesses heterogeneous tabular data for the DGD-TabPA diffusion model.

    Numerical features -> QuantileTransformer (or MinMaxScaler)
    Categorical features -> OneHotEncoder
    Classification target -> LabelEncoder
    Regression target -> KBinsDiscretizer (conditioning) + continuous y retained
    """

    def __init__(
        self,
        n_quantiles: int = 1000,
        random_state: int = 42,
        numerical_transform: str = "quantile_normal",
        task: str = "classification",
        n_regression_bins: int = 10,
    ):
        self.n_quantiles = n_quantiles
        self.random_state = random_state
        self.numerical_transform = numerical_transform
        self.task = task
        self.n_regression_bins = n_regression_bins
        self.num_transformer = None
        self.cat_encoder: Optional[OneHotEncoder] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.target_binner: Optional[KBinsDiscretizer] = None
        self.bin_centers_: Optional[np.ndarray] = None
        self.info: Optional[DatasetInfo] = None
        self._fitted = False

    def load_dataset(
        self,
        name: str,
        filepath: str,
        target_col: str,
        test_size: float = 0.2,
        max_samples: Optional[int] = None,
    ) -> tuple:
        """Load a CSV, detect feature types, optionally subsample, split train/test."""
        df = pd.read_csv(filepath)

        if max_samples is not None and len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=self.random_state).reset_index(
                drop=True
            )
            print(f"  Subsampled to {max_samples} rows for tractable training")

        num_features = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in num_features:
            num_features.remove(target_col)

        cat_features = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if target_col in cat_features:
            cat_features.remove(target_col)

        original_dtypes = {c: str(df[c].dtype) for c in num_features + cat_features}

        y = df[target_col].values
        X = df[num_features + cat_features]

        n_unique = len(np.unique(y))
        can_stratify = (
            self.task == "classification"
            and n_unique > 1
            and n_unique <= 100
        )
        if can_stratify:
            _, counts = np.unique(y, return_counts=True)
            can_stratify = counts.min() >= 2

        if can_stratify:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state, stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state
            )

        self.info = DatasetInfo(
            name=name,
            num_features=num_features,
            cat_features=cat_features,
            target_col=target_col,
            num_classes=n_unique if self.task == "classification" else self.n_regression_bins,
            num_dim=len(num_features),
            cat_dim=0,
            total_dim=0,
            original_dtypes=original_dtypes,
            task=self.task,
        )

        return X_train, X_test, y_train, y_test

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray):
        """Fit transformers on training data."""
        num_features = self.info.num_features
        cat_features = self.info.cat_features

        if num_features:
            if self.numerical_transform == "minmax":
                self.num_transformer = MinMaxScaler(feature_range=(-1, 1))
            else:
                self.num_transformer = QuantileTransformer(
                    n_quantiles=min(self.n_quantiles, len(X_train)),
                    output_distribution="normal",
                    random_state=self.random_state,
                )
            self.num_transformer.fit(X_train[num_features].values)

        if cat_features:
            self.cat_encoder = OneHotEncoder(
                sparse_output=False,
                handle_unknown="ignore",
                dtype=np.float32,
            )
            self.cat_encoder.fit(X_train[cat_features].values)
            self.info.cat_categories = [
                list(cats) for cats in self.cat_encoder.categories_
            ]
            self.info.cat_dim = sum(len(c) for c in self.cat_encoder.categories_)
        else:
            self.info.cat_dim = 0

        self.info.total_dim = self.info.num_dim + self.info.cat_dim

        if self.task == "regression":
            y_col = np.asarray(y_train, dtype=float).reshape(-1, 1)
            n_bins = min(self.n_regression_bins, max(2, len(np.unique(y_col)) // 2))
            self.target_binner = KBinsDiscretizer(
                n_bins=n_bins,
                encode="ordinal",
                strategy="quantile",
            )
            bins = self.target_binner.fit_transform(y_col).astype(np.int64).ravel()
            self.info.num_classes = int(bins.max()) + 1
            # Representative continuous value per bin (median of training targets)
            centers = []
            for b in range(self.info.num_classes):
                vals = y_col.ravel()[bins == b]
                centers.append(float(np.median(vals)) if len(vals) else 0.0)
            self.bin_centers_ = np.array(centers, dtype=float)
        else:
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(y_train)
            self.info.num_classes = len(self.label_encoder.classes_)

        self._fitted = True

    def transform(self, X: pd.DataFrame, y: np.ndarray) -> tuple:
        """Transform features and labels into tensors (labels = conditioning ids)."""
        assert self._fitted, "Call fit() before transform()."

        parts = []

        if self.info.num_features:
            X_num = self.num_transformer.transform(X[self.info.num_features].values)
            parts.append(X_num.astype(np.float32))

        if self.info.cat_features:
            X_cat = self.cat_encoder.transform(X[self.info.cat_features].values)
            parts.append(X_cat.astype(np.float32))

        X_all = np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]

        if self.task == "regression":
            y_col = np.asarray(y, dtype=float).reshape(-1, 1)
            y_enc = self.target_binner.transform(y_col).astype(np.int64).ravel()
            # Clip to valid class ids (edge bins)
            y_enc = np.clip(y_enc, 0, self.info.num_classes - 1)
        else:
            y_enc = self.label_encoder.transform(y).astype(np.int64)

        return torch.tensor(X_all, dtype=torch.float32), torch.tensor(y_enc, dtype=torch.long)

    def fit_transform(self, X_train: pd.DataFrame, y_train: np.ndarray) -> tuple:
        self.fit(X_train, y_train)
        return self.transform(X_train, y_train)

    def inverse_transform_numerical(self, X_num: np.ndarray) -> np.ndarray:
        if self.num_transformer is None:
            return X_num
        return self.num_transformer.inverse_transform(X_num)

    def inverse_transform_categorical(self, X_cat_onehot: np.ndarray) -> np.ndarray:
        if self.cat_encoder is None:
            return X_cat_onehot
        return self.cat_encoder.inverse_transform(X_cat_onehot)

    def inverse_transform(self, X_tensor: torch.Tensor) -> pd.DataFrame:
        X = X_tensor.detach().cpu().numpy()
        num_dim = self.info.num_dim
        cat_dim = self.info.cat_dim

        result = {}

        if num_dim > 0:
            X_num = X[:, :num_dim]
            X_num_inv = self.inverse_transform_numerical(X_num)
            for i, col in enumerate(self.info.num_features):
                result[col] = X_num_inv[:, i]

        if cat_dim > 0:
            X_cat = X[:, num_dim : num_dim + cat_dim]
            X_cat_inv = self.inverse_transform_categorical(X_cat)
            for i, col in enumerate(self.info.cat_features):
                result[col] = X_cat_inv[:, i]

        return pd.DataFrame(result)

    def inverse_transform_labels(self, y_tensor: torch.Tensor) -> np.ndarray:
        """Map conditioning ids back to original label space."""
        y_np = y_tensor.detach().cpu().numpy().astype(int)
        if self.task == "regression":
            y_np = np.clip(y_np, 0, len(self.bin_centers_) - 1)
            return self.bin_centers_[y_np]
        return self.label_encoder.inverse_transform(y_np)
