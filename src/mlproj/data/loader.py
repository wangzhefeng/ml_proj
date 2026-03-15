from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.datasets import load_diabetes, load_iris, load_wine
from sklearn.model_selection import train_test_split

from mlproj.types import DatasetBundle


@dataclass(slots=True)
class DatasetLoader:
    random_state: int = 42

    def load(self, config: dict[str, Any]) -> DatasetBundle:
        source = config.get("source", {})
        source_type = source.get("type", "csv")

        if source_type == "csv":
            return self._load_csv(config)
        if source_type == "sklearn":
            return self._load_sklearn(config)
        raise ValueError(f"Unsupported source.type: {source_type}")

    def _load_csv(self, config: dict[str, Any]) -> DatasetBundle:
        source = config["source"]
        path = source["path"]
        target_col = source.get("target")
        sep = source.get("sep", ",")
        df = pd.read_csv(path, sep=sep)

        if target_col is None:
            X = df
            y = None
        else:
            if target_col not in df.columns:
                raise ValueError(f"Target column not found: {target_col}")
            X = df.drop(columns=[target_col])
            y = df[target_col]

        return self._split(X, y, config)

    def _load_sklearn(self, config: dict[str, Any]) -> DatasetBundle:
        dataset_name = config["source"].get("name", "iris")
        if dataset_name == "iris":
            raw = load_iris(as_frame=True)
        elif dataset_name == "wine":
            raw = load_wine(as_frame=True)
        elif dataset_name == "diabetes":
            raw = load_diabetes(as_frame=True)
        else:
            raise ValueError(f"Unsupported sklearn dataset: {dataset_name}")

        X = raw.data
        y = raw.target if hasattr(raw, "target") else None
        return self._split(X, y, config)

    def _split(self, X: pd.DataFrame, y: pd.Series | None, config: dict[str, Any]) -> DatasetBundle:
        split_cfg = config.get("split", {})
        strategy = split_cfg.get("strategy", "random")
        valid_size = float(split_cfg.get("valid_size", 0.2))
        test_size = float(split_cfg.get("test_size", 0.2))

        if strategy == "timeseries":
            return self._time_split(X, y, valid_size=valid_size, test_size=test_size)

        stratify_y = None
        if y is not None and y.nunique(dropna=True) <= 30:
            stratify_y = y

        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=stratify_y,
        )

        rel_valid = valid_size / max(1e-8, 1.0 - test_size)
        stratify_train = None
        if y_train_full is not None and y_train_full.nunique(dropna=True) <= 30:
            stratify_train = y_train_full

        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train_full,
            y_train_full,
            test_size=rel_valid,
            random_state=self.random_state,
            stratify=stratify_train,
        )

        return DatasetBundle(
            X_train=X_train.reset_index(drop=True),
            y_train=None if y_train is None else y_train.reset_index(drop=True),
            X_valid=X_valid.reset_index(drop=True),
            y_valid=None if y_valid is None else y_valid.reset_index(drop=True),
            X_test=X_test.reset_index(drop=True),
            y_test=None if y_test is None else y_test.reset_index(drop=True),
            metadata={"strategy": strategy},
        )

    def _time_split(
        self, X: pd.DataFrame, y: pd.Series | None, valid_size: float, test_size: float
    ) -> DatasetBundle:
        n = len(X)
        test_n = int(n * test_size)
        valid_n = int(n * valid_size)
        train_n = n - valid_n - test_n
        if train_n <= 0:
            raise ValueError("Split sizes are too large for timeseries split")

        X_train = X.iloc[:train_n]
        X_valid = X.iloc[train_n : train_n + valid_n]
        X_test = X.iloc[train_n + valid_n :]

        if y is None:
            y_train = y_valid = y_test = None
        else:
            y_train = y.iloc[:train_n]
            y_valid = y.iloc[train_n : train_n + valid_n]
            y_test = y.iloc[train_n + valid_n :]

        return DatasetBundle(
            X_train=X_train.reset_index(drop=True),
            y_train=None if y_train is None else y_train.reset_index(drop=True),
            X_valid=X_valid.reset_index(drop=True),
            y_valid=None if y_valid is None else y_valid.reset_index(drop=True),
            X_test=X_test.reset_index(drop=True),
            y_test=None if y_test is None else y_test.reset_index(drop=True),
            metadata={"strategy": "timeseries"},
        )
