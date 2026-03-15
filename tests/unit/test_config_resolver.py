from __future__ import annotations

import sys

import pytest

from mlproj.config import ConfigError, ConfigResolver


def test_config_resolver_from_yaml_and_overrides(tmp_path):
    path = tmp_path / "train.yaml"
    path.write_text(
        """task: classification
source:
  type: csv
  path: dataset/classification/train.csv
  target: target
model:
  backend: sklearn
  name: logistic_regression
  params: {}
feature_pipeline: []
""",
        encoding="utf-8",
    )

    resolver = ConfigResolver.from_yaml(path)
    resolver.apply_overrides({"action": "train", "random_state": 123})
    spec = resolver.resolve()
    assert spec["action"] == "train"
    assert spec["task"] == "classification"
    assert spec["random_state"] == 123


def test_config_resolver_from_python(tmp_path):
    mod = tmp_path / "my_cfg.py"
    mod.write_text(
        """
class MyCfg:
    def __init__(self):
        self.task = 'regression'
        self.source = {'type': 'csv', 'path': 'dataset/regression/train.csv', 'target': 'target'}
        self.model = {'backend': 'sklearn', 'name': 'linear_regression', 'params': {}}
        self.feature_pipeline = []
""",
        encoding="utf-8",
    )

    sys.path.insert(0, str(tmp_path))
    try:
        resolver = ConfigResolver.from_python("my_cfg", "MyCfg")
        resolver.apply_overrides({"action": "train"})
        spec = resolver.resolve()
    finally:
        sys.path.remove(str(tmp_path))

    assert spec["task"] == "regression"
    assert spec["model"]["name"] == "linear_regression"


def test_override_parse_invalid():
    with pytest.raises(ConfigError):
        ConfigResolver.parse_override_items(["invalid"])
