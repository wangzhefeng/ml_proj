from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def _resolve_path(config_filename: str | Path) -> Path:
    p = Path(config_filename)
    if p.is_absolute():
        return p
    return Path(__file__).resolve().parents[3] / "data_provider" / p


def load_yaml(file_name: str | Path) -> dict[str, Any]:
    path = _resolve_path(file_name)
    with path.open("r", encoding="utf-8") as infile:
        return yaml.load(infile, Loader=yaml.FullLoader)


def get_params(yaml_path: str) -> dict[str, Any]:
    return load_yaml(yaml_path)


def load_yaml_config(yaml_filename: str) -> dict[str, Any]:
    cfg_params = load_yaml(yaml_filename)
    _ensure_model_save_dir(cfg_params)
    return cfg_params


def load_json_config(config_filename: str) -> dict[str, Any]:
    path = _resolve_path(config_filename)
    with path.open("r", encoding="utf-8") as infile:
        cfg_params = json.load(infile)
    _ensure_model_save_dir(cfg_params)
    return cfg_params


def _ensure_model_save_dir(cfg_params: dict[str, Any]) -> None:
    try:
        save_dir = cfg_params.get("model", {}).get("save_dir")
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def get_lgb_train_test_data(train_path: str, test_path: str, weight_paths: list[str] | None = None):
    df_train = pd.read_csv(train_path, header=None, sep="\t")
    df_test = pd.read_csv(test_path, header=None, sep="\t")

    y_train = df_train[0]
    y_test = df_test[0]
    X_train = df_train.drop(0, axis=1)
    X_test = df_test.drop(0, axis=1)

    try:
        import lightgbm as lgb
    except Exception as exc:
        raise RuntimeError("lightgbm is required for get_lgb_train_test_data") from exc

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


def get_xgb_train_test_data(train_path: str, test_path: str, weight_paths: list[str] | None = None):
    df_train = pd.read_csv(train_path, header=None, sep="\t")
    df_test = pd.read_csv(test_path, header=None, sep="\t")

    y_train = df_train[0]
    y_test = df_test[0]
    X_train = df_train.drop(0, axis=1)
    X_test = df_test.drop(0, axis=1)

    try:
        import xgboost as xgb
    except Exception as exc:
        raise RuntimeError("xgboost is required for get_xgb_train_test_data") from exc

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
