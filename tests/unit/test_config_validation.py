import pytest

from mlproj.config import ConfigError, load_config


def test_load_train_config_valid(tmp_path):
    cfg = tmp_path / "train.yaml"
    cfg.write_text(
        """task: classification
source:
  type: sklearn
  name: wine
split:
  strategy: random
  valid_size: 0.2
  test_size: 0.2
model:
  name: logistic_regression
  params: {}
tune:
  enabled: false
""",
        encoding="utf-8",
    )
    out = load_config(cfg)
    assert out["task"] == "classification"
    assert out["source"]["name"] == "wine"


def test_load_config_split_invalid(tmp_path):
    cfg = tmp_path / "bad_split.yaml"
    cfg.write_text(
        """task: regression
source:
  type: sklearn
  name: diabetes
split:
  strategy: random
  valid_size: 0.7
  test_size: 0.4
model:
  name: linear_regression
  params: {}
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError):
        load_config(cfg)


def test_load_config_source_invalid(tmp_path):
    cfg = tmp_path / "bad_source.yaml"
    cfg.write_text(
        """task: classification
source:
  type: csv
model:
  name: logistic_regression
  params: {}
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError):
        load_config(cfg)
