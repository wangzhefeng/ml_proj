from mlproj.cli import main


def test_cli_train(tmp_path):
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        f"""task: classification
artifact_root: {(tmp_path / 'artifacts').as_posix()}
source:
  type: csv
  path: dataset/classification/train.csv
  target: target
split:
  strategy: random
  valid_size: 0.2
  test_size: 0.2
model:
  backend: sklearn
  name: logistic_regression
  params: {{}}
tune:
  enabled: false
feature_pipeline: []
""",
        encoding="utf-8",
    )

    code = main(["train", "--config-yaml", str(config_path)])
    assert code == 0
