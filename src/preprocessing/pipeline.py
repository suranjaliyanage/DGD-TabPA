"""
Module 1: Heterogeneous Data Ingestion and Preprocessing

Transforms raw tabular CSV data into unified float tensors suitable for the
diffusion model. Applies Gaussian Quantile Transformations to numerical features
and One-Hot Encoding to categorical features.
"""

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import QuantileTransformer, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass
class DatasetInfo:
    """Metadata about a preprocessed dataset, needed for inverse transforms."""
    name: str
    num_features: list
    cat_features: list
    target_col: str
    num_classes: int
    num_dim: int  # dimension after transforming numerical features
    cat_dim: int  # dimension after one-hot encoding categorical features
    total_dim: int  # num_dim + cat_dim
    cat_categories: list = field(default_factory=list)
    original_dtypes: dict = field(default_factory=dict)


class TabularPreprocessor:
    """
    Preprocesses heterogeneous tabular data for the DGD-TabPA diffusion model.

    Numerical features -> QuantileTransformer (output_distribution='normal')
    Categorical features -> OneHotEncoder (sparse_output=False)
    Target column -> LabelEncoder
    """

    def __init__(self, n_quantiles: int = 1000, random_state: int = 42):
        self.n_quantiles = n_quantiles
        self.random_state = random_state
        self.num_transformer: Optional[QuantileTransformer] = None
        self.cat_encoder: Optional[OneHotEncoder] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.info: Optional[DatasetInfo] = None
        self._fitted = False

    def load_dataset(
        self,
        name: str,
        filepath: str,
        target_col: str,
        test_size: float = 0.2,
    ) -> tuple:
        """Load a CSV, detect feature types, split into train/test."""
        df = pd.read_csv(filepath)

        num_features = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in num_features:
            num_features.remove(target_col)

        cat_features = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if target_col in cat_features:
            cat_features.remove(target_col)

        original_dtypes = {c: str(df[c].dtype) for c in num_features + cat_features}

        y = df[target_col].values
        X = df[num_features + cat_features]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        self.info = DatasetInfo(
            name=name,
            num_features=num_features,
            cat_features=cat_features,
            target_col=target_col,
            num_classes=len(np.unique(y)),
            num_dim=len(num_features),
            cat_dim=0,
            total_dim=0,
            original_dtypes=original_dtypes,
        )

        return X_train, X_test, y_train, y_test

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray):
        """Fit transformers on training data."""
        num_features = self.info.num_features
        cat_features = self.info.cat_features

        if num_features:
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

        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(y_train)

        self._fitted = True

    def transform(self, X: pd.DataFrame, y: np.ndarray) -> tuple:
        """Transform features and labels into tensors."""
        assert self._fitted, "Call fit() before transform()."

        parts = []

        if self.info.num_features:
            X_num = self.num_transformer.transform(X[self.info.num_features].values)
            parts.append(X_num.astype(np.float32))

        if self.info.cat_features:
            X_cat = self.cat_encoder.transform(X[self.info.cat_features].values)
            parts.append(X_cat.astype(np.float32))

        X_all = np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]
        y_enc = self.label_encoder.transform(y).astype(np.int64)

        return torch.tensor(X_all, dtype=torch.float32), torch.tensor(y_enc, dtype=torch.long)

    def fit_transform(self, X_train: pd.DataFrame, y_train: np.ndarray) -> tuple:
        """Convenience: fit on training data, then transform it."""
        self.fit(X_train, y_train)
        return self.transform(X_train, y_train)

    def inverse_transform_numerical(self, X_num: np.ndarray) -> np.ndarray:
        """Inverse quantile transform for numerical columns only."""
        if self.num_transformer is None:
            return X_num
        return self.num_transformer.inverse_transform(X_num)

    def inverse_transform_categorical(self, X_cat_onehot: np.ndarray) -> np.ndarray:
        """Inverse one-hot encoding for categorical columns only."""
        if self.cat_encoder is None:
            return X_cat_onehot
        return self.cat_encoder.inverse_transform(X_cat_onehot)

    def inverse_transform(self, X_tensor: torch.Tensor) -> pd.DataFrame:
        """Full inverse transform from model output tensor back to a DataFrame."""
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
            X_cat = X[:, num_dim:num_dim + cat_dim]
            X_cat_inv = self.inverse_transform_categorical(X_cat)
            for i, col in enumerate(self.info.cat_features):
                result[col] = X_cat_inv[:, i]

        return pd.DataFrame(result)

    def inverse_transform_labels(self, y_tensor: torch.Tensor) -> np.ndarray:
        """Inverse label encoding."""
        return self.label_encoder.inverse_transform(y_tensor.detach().cpu().numpy())
