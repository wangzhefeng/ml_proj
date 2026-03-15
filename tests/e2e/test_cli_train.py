from mlproj.cli import main


def test_cli_train(tmp_path):
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        f"""task: classification
artifact_root: {(tmp_path / 'artifacts').as_posix()}
source:
  type: sklearn
  name: wine
split:
  strategy: random
  valid_size: 0.2
  test_size: 0.2
model:
  name: logistic_regression
  params: {{}}
tune:
  enabled: false
""",
        encoding="utf-8",
    )

    code = main(["train", "--config", str(config_path)])
    assert code == 0
