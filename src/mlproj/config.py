from __future__ import annotations

import importlib
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

TaskType = Literal[
    "classification",
    "regression",
    "clustering",
    "pca_reduction",
    "anomaly_detection",
    "topic_modeling",
]
ActionType = Literal["train", "tune", "evaluate", "predict", "serve"]


class ConfigError(ValueError):
    pass


class SourceConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["csv"] = "csv"
    path: str | None = None
    train_path: str | None = None
    valid_path: str | None = None
    test_path: str | None = None
    target: str | None = None
    text_column: str | None = None
    sep: str = ","
    data_version: str | None = None

    @model_validator(mode="after")
    def validate_source(self) -> "SourceConfig":
        if not self.path and not self.train_path:
            raise ValueError("CSV source requires 'path' or 'train_path'")
        return self


class SplitConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    strategy: Literal["random"] = "random"
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

    backend: str
    name: str
    params: dict[str, Any] = Field(default_factory=dict)
    backend_provider: str | None = None


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

    task: TaskType
    artifact_root: str = "artifacts"
    source: SourceConfig
    split: SplitConfig = Field(default_factory=SplitConfig)
    model: ModelConfig
    feature_pipeline: list[dict[str, Any]] = Field(default_factory=list)
    tune: TuneConfig = Field(default_factory=TuneConfig)
    random_state: int = 42

    @model_validator(mode="after")
    def validate_task_requirements(self) -> "TrainRunConfig":
        supervised_tasks = {"classification", "regression"}
        if self.task in supervised_tasks and not self.source.target:
            raise ValueError(f"task '{self.task}' requires source.target")
        if self.task == "topic_modeling" and not self.source.text_column:
            raise ValueError("task 'topic_modeling' requires source.text_column")
        return self


class EvaluateConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    model_uri: str
    input: str
    target_col: str | None = None
    task: TaskType | None = None
    output_metrics: str | None = None


class RunSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: ActionType
    task: TaskType | None = None
    artifact_root: str = "artifacts"
    random_state: int = 42

    source: SourceConfig | None = None
    split: SplitConfig = Field(default_factory=SplitConfig)
    model: ModelConfig | None = None
    feature_pipeline: list[dict[str, Any]] = Field(default_factory=list)
    tune: TuneConfig = Field(default_factory=TuneConfig)

    model_uri: str | None = None
    input: str | None = None
    output: str | None = None
    target_col: str | None = None
    output_metrics: str | None = None
    host: str = "127.0.0.1"
    port: int = 8000

    @model_validator(mode="after")
    def validate_action_requirements(self) -> "RunSpec":
        if self.action in {"train", "tune"}:
            if self.task is None or self.source is None or self.model is None:
                raise ValueError("train/tune requires task, source, and model")
            if self.task in {"classification", "regression"} and not self.source.target:
                raise ValueError(f"task '{self.task}' requires source.target")
            if self.task == "topic_modeling" and not self.source.text_column:
                raise ValueError("task 'topic_modeling' requires source.text_column")
            if self.action == "tune":
                self.tune.enabled = True

        if self.action == "evaluate":
            if not self.model_uri or not self.input:
                raise ValueError("evaluate requires model_uri and input")

        if self.action == "predict":
            if not self.model_uri or not self.input or not self.output:
                raise ValueError("predict requires model_uri, input, and output")

        if self.action == "serve":
            if not self.model_uri:
                raise ValueError("serve requires model_uri")

        return self


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


def _to_dict_object(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump()
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {k: _to_dict_object(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_dict_object(v) for v in value]
    if isinstance(value, tuple):
        return [_to_dict_object(v) for v in value]
    return value


class ConfigResolver:
    def __init__(self, raw: dict[str, Any] | None = None) -> None:
        self.raw: dict[str, Any] = raw or {}

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ConfigResolver":
        return cls(_load_yaml_dict(path))

    @classmethod
    def from_python(cls, module_path: str, class_name: str) -> "ConfigResolver":
        module = importlib.import_module(module_path)
        if not hasattr(module, class_name):
            raise ConfigError(f"Config class '{class_name}' not found in module '{module_path}'")

        cfg_cls = getattr(module, class_name)
        cfg_obj = cfg_cls()

        if hasattr(cfg_obj, "to_dict") and callable(cfg_obj.to_dict):
            data = cfg_obj.to_dict()
        elif isinstance(cfg_obj, BaseModel):
            data = cfg_obj.model_dump()
        else:
            data = {
                key: value
                for key, value in vars(cfg_obj).items()
                if not key.startswith("_") and not callable(value)
            }

        if not isinstance(data, dict):
            raise ConfigError("Python config class must produce a mapping")
        return cls(_to_dict_object(data))

    def apply_overrides(self, overrides: dict[str, Any]) -> "ConfigResolver":
        for dotted_key, value in overrides.items():
            if not dotted_key:
                continue
            cursor = self.raw
            parts = dotted_key.split(".")
            for part in parts[:-1]:
                if part not in cursor or not isinstance(cursor[part], dict):
                    cursor[part] = {}
                cursor = cursor[part]
            cursor[parts[-1]] = value
        return self

    def resolve(self) -> dict[str, Any]:
        try:
            return RunSpec.model_validate(self.raw).model_dump()
        except ValidationError as exc:
            raise ConfigError(f"Invalid run spec: {exc}") from exc

    @staticmethod
    def parse_override_items(items: list[str] | None) -> dict[str, Any]:
        overrides: dict[str, Any] = {}
        for item in items or []:
            if "=" not in item:
                raise ConfigError(f"Invalid override '{item}', expected key=value")
            key, raw_value = item.split("=", 1)
            key = key.strip()
            if not key:
                raise ConfigError(f"Invalid override '{item}', key is empty")
            value = yaml.safe_load(raw_value)
            overrides[key] = value
        return overrides


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
