from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from mlproj.types import DatasetBundle


@dataclass(slots=True)
class DatasetLoader:
    random_state: int = 42

    def load(self, config: dict[str, Any]) -> DatasetBundle:
        source = config.get("source", {})
        source_type = source.get("type", "csv")

        if source_type != "csv":
            raise ValueError(
                "Unsupported source.type: "
                f"{source_type}. Only user-provided csv files are supported in this framework."
            )

        backend = str(config.get("model", {}).get("backend", "")).lower()
        boosting_bundle = self._maybe_load_boosting_train_test(source, backend, config)
        if boosting_bundle is not None:
            return boosting_bundle

        return self._load_csv(config)

    def _load_csv(self, config: dict[str, Any]) -> DatasetBundle:
        source = config["source"]
        target_col = source.get("target")
        sep = source.get("sep", ",")
        task = str(config.get("task", "classification"))

        if source.get("train_path"):
            return self._load_explicit_splits(
                source=source,
                target_col=target_col,
                sep=sep,
                task=task,
            )

        path = source["path"]
        df = self._read_csv(path, sep)
        X, y = self._split_target(df, target_col)
        return self._split(X, y, config, task=task)

    def _load_explicit_splits(
        self,
        source: dict[str, Any],
        target_col: str | None,
        sep: str,
        task: str,
    ) -> DatasetBundle:
        train_df = self._read_csv(source["train_path"], sep)
        valid_df = self._read_csv(source["valid_path"], sep) if source.get("valid_path") else None
        test_df = self._read_csv(source["test_path"], sep) if source.get("test_path") else None

        X_train, y_train = self._split_target(train_df, target_col)
        X_valid, y_valid = (
            self._split_target(valid_df, target_col) if valid_df is not None else (None, None)
        )
        X_test, y_test = (
            self._split_target(test_df, target_col) if test_df is not None else (None, None)
        )

        if X_valid is None or X_test is None:
            split_cfg = {
                "split": {
                    "strategy": "random",
                    "valid_size": 0.2,
                    "test_size": 0.2,
                }
            }
            split_cfg["split"].update(source.get("split", {}))
            split_bundle = self._split(X_train, y_train, split_cfg, task=task)
            X_train = split_bundle.X_train
            y_train = split_bundle.y_train
            X_valid = split_bundle.X_valid
            y_valid = split_bundle.y_valid
            X_test = split_bundle.X_test
            y_test = split_bundle.y_test

        return DatasetBundle(
            X_train=X_train.reset_index(drop=True),
            y_train=None if y_train is None else y_train.reset_index(drop=True),
            X_valid=None if X_valid is None else X_valid.reset_index(drop=True),
            y_valid=None if y_valid is None else y_valid.reset_index(drop=True),
            X_test=None if X_test is None else X_test.reset_index(drop=True),
            y_test=None if y_test is None else y_test.reset_index(drop=True),
            metadata={"strategy": "explicit_files"},
        )

    def load_lgb_train_test_data(
        self,
        train_path: str,
        test_path: str,
        weight_paths: list[str] | None = None,
    ):
        X_train, y_train, X_test, y_test = self._read_tsv_train_test(train_path, test_path)

        try:
            import lightgbm as lgb
        except Exception as exc:
            raise RuntimeError("lightgbm is required for load_lgb_train_test_data") from exc

        weight_paths = weight_paths or []
        if weight_paths:
            W_train = pd.read_csv(weight_paths[0], header=None)[0]
            W_test = pd.read_csv(weight_paths[1], header=None)[0]
            lgb_train = lgb.Dataset(X_train, y_train, weight=W_train, free_raw_data=False)
            lgb_eval = lgb.Dataset(
                X_test, y_test, reference=lgb_train, weight=W_test, free_raw_data=False
            )
            return W_train, W_test, X_train, y_train, X_test, y_test, lgb_train, lgb_eval

        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
        return X_train, y_train, X_test, y_test, lgb_train, lgb_eval

    def load_xgb_train_test_data(
        self,
        train_path: str,
        test_path: str,
        weight_paths: list[str] | None = None,
    ):
        X_train, y_train, X_test, y_test = self._read_tsv_train_test(train_path, test_path)

        try:
            import xgboost as xgb
        except Exception as exc:
            raise RuntimeError("xgboost is required for load_xgb_train_test_data") from exc

        weight_paths = weight_paths or []
        if weight_paths:
            W_train = pd.read_csv(weight_paths[0], header=None)[0]
            W_test = pd.read_csv(weight_paths[1], header=None)[0]
            dtrain = xgb.DMatrix(X_train, label=y_train, weight=W_train)
            dtest = xgb.DMatrix(X_test, label=y_test, weight=W_test)
            return W_train, W_test, X_train, y_train, X_test, y_test, dtrain, dtest

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        return X_train, y_train, X_test, y_test, dtrain, dtest

    def _read_tsv_train_test(
        self,
        train_path: str,
        test_path: str,
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        df_train = pd.read_csv(train_path, header=None, sep="\t")
        df_test = pd.read_csv(test_path, header=None, sep="\t")

        y_train = df_train[0]
        y_test = df_test[0]
        X_train = df_train.drop(0, axis=1)
        X_test = df_test.drop(0, axis=1)
        return X_train, y_train, X_test, y_test

    def _maybe_load_boosting_train_test(
        self,
        source: dict[str, Any],
        backend: str,
        config: dict[str, Any],
    ) -> DatasetBundle | None:
        train_path = source.get("train_path")
        test_path = source.get("test_path")
        if not train_path or not test_path:
            return None

        if backend not in {"lightgbm", "xgboost"}:
            return None

        weight_paths = source.get("weight_paths")
        valid_size = float(config.get("split", {}).get("valid_size", 0.2))

        backend_train = None
        backend_test = None
        backend_weights: tuple[pd.Series, pd.Series] | None = None

        if backend == "lightgbm":
            loaded = self.load_lgb_train_test_data(train_path, test_path, weight_paths=weight_paths)
            if weight_paths:
                W_train, W_test, X_train_raw, y_train_raw, X_test, y_test, backend_train, backend_test = (
                    loaded
                )
                backend_weights = (W_train, W_test)
            else:
                X_train_raw, y_train_raw, X_test, y_test, backend_train, backend_test = loaded
        else:
            loaded = self.load_xgb_train_test_data(train_path, test_path, weight_paths=weight_paths)
            if weight_paths:
                W_train, W_test, X_train_raw, y_train_raw, X_test, y_test, backend_train, backend_test = (
                    loaded
                )
                backend_weights = (W_train, W_test)
            else:
                X_train_raw, y_train_raw, X_test, y_test, backend_train, backend_test = loaded

        stratify_y = y_train_raw if y_train_raw.nunique(dropna=True) <= 30 else None
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train_raw,
            y_train_raw,
            test_size=valid_size,
            random_state=self.random_state,
            stratify=stratify_y,
        )

        metadata: dict[str, Any] = {
            "strategy": "boosting_explicit_train_test",
            "backend": backend,
            "backend_train": backend_train,
            "backend_test": backend_test,
        }
        if backend_weights is not None:
            metadata["weight_train"] = backend_weights[0]
            metadata["weight_test"] = backend_weights[1]

        return DatasetBundle(
            X_train=X_train.reset_index(drop=True),
            y_train=y_train.reset_index(drop=True),
            X_valid=X_valid.reset_index(drop=True),
            y_valid=y_valid.reset_index(drop=True),
            X_test=X_test.reset_index(drop=True),
            y_test=y_test.reset_index(drop=True),
            metadata=metadata,
        )

    def _read_csv(self, path_like: str | Path, sep: str) -> pd.DataFrame:
        path = Path(path_like)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")
        return pd.read_csv(path, sep=sep)

    def _split_target(
        self, df: pd.DataFrame | None, target_col: str | None
    ) -> tuple[pd.DataFrame | None, pd.Series | None]:
        if df is None:
            return None, None
        if target_col is None:
            return df, None
        if target_col not in df.columns:
            raise ValueError(f"Target column not found: {target_col}")
        return df.drop(columns=[target_col]), df[target_col]

    def _split(
        self,
        X: pd.DataFrame,
        y: pd.Series | None,
        config: dict[str, Any],
        task: str,
    ) -> DatasetBundle:
        split_cfg = config.get("split", {})
        strategy = split_cfg.get("strategy", "random")
        valid_size = float(split_cfg.get("valid_size", 0.2))
        test_size = float(split_cfg.get("test_size", 0.2))

        stratify_y = None
        if task == "classification" and y is not None and y.nunique(dropna=True) <= 30:
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
        if task == "classification" and y_train_full is not None and y_train_full.nunique(dropna=True) <= 30:
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
