from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class ConfigError(ValueError):
    pass


class SourceConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["csv", "sklearn"] = "csv"
    path: str | None = None
    train_path: str | None = None
    valid_path: str | None = None
    test_path: str | None = None
    target: str | None = None
    sep: str = ","
    name: str | None = None
    data_version: str | None = None

    @model_validator(mode="after")
    def validate_source(self) -> "SourceConfig":
        if self.type == "csv":
            if not self.path and not self.train_path:
                raise ValueError("CSV source requires 'path' or 'train_path'")
            return self

        if self.type == "sklearn" and not self.name:
            raise ValueError("sklearn source requires 'name'")
        return self


class SplitConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    strategy: Literal["random", "timeseries"] = "random"
    valid_size: float = 0.2
    test_size: float = 0.2

    @model_validator(mode="after")
    def validate_split(self) -> "SplitConfig":
        if not (0.0 <= self.valid_size < 1.0):
            raise ValueError("split.valid_size must be in [0, 1)")
        if not (0.0 <= self.test_size < 1.0):
            raise ValueError("split.test_size must be in [0, 1)")
        if self.valid_size + self.test_size >= 1.0:
            raise ValueError("split.valid_size + split.test_size must be < 1")
        return self


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class TuneConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    enabled: bool = False
    method: Literal["grid", "random", "halving_grid", "halving_random"] = "grid"
    n_iter: int = 20
    cv_folds: int = 5
    scoring: str | None = None
    param_grid: dict[str, Any] = Field(default_factory=dict)


class TrainRunConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    task: Literal["classification", "regression", "clustering", "timeseries"]
    artifact_root: str = "artifacts"
    source: SourceConfig
    split: SplitConfig = Field(default_factory=SplitConfig)
    model: ModelConfig
    tune: TuneConfig = Field(default_factory=TuneConfig)
    random_state: int = 42

    @model_validator(mode="after")
    def validate_model_task(self) -> "TrainRunConfig":
        supported = {
            "classification": {"logistic_regression", "random_forest"},
            "regression": {"linear_regression", "random_forest"},
            "clustering": {"kmeans", "kmeans_small"},
            "timeseries": {"linear_regression", "random_forest"},
        }
        if self.model.name not in supported[self.task]:
            raise ValueError(
                f"model.name '{self.model.name}' is not supported for task '{self.task}'"
            )
        if self.task == "timeseries" and self.split.strategy != "timeseries":
            raise ValueError("timeseries task requires split.strategy=timeseries")
        return self


class EvaluateConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    model_uri: str
    input: str
    target_col: str | None = None
    task: Literal["classification", "regression", "timeseries", "clustering"] | None = None
    output_metrics: str | None = None


def _normalize_path(path: str | Path) -> Path:
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    return Path.cwd() / path_obj


def _load_yaml_dict(path: str | Path) -> dict[str, Any]:
    config_path = _normalize_path(path)
    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ConfigError("Config root must be a mapping")
    return data


def load_config(path: str | Path) -> dict[str, Any]:
    data = _load_yaml_dict(path)

    try:
        if {"task", "source", "model"}.issubset(data.keys()):
            return TrainRunConfig.model_validate(data).model_dump()
        if {"model_uri", "input"}.issubset(data.keys()):
            return EvaluateConfig.model_validate(data).model_dump()
        return data
    except ValidationError as exc:
        raise ConfigError(f"Invalid config: {exc}") from exc
