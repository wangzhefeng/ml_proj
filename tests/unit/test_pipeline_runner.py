from __future__ import annotations

import json
from pathlib import Path

from mlproj.pipeline.runner import PipelineRunner


def test_pipeline_runner_train_writes_stage_and_spec(tmp_path):
    spec = {
        "action": "train",
        "task": "classification",
        "artifact_root": str(tmp_path / "artifacts"),
        "random_state": 42,
        "source": {"type": "csv", "path": "dataset/classification/train.csv", "target": "target"},
        "split": {"strategy": "random", "valid_size": 0.2, "test_size": 0.2},
        "model": {"backend": "sklearn", "name": "logistic_regression", "params": {}},
        "feature_pipeline": [],
        "tune": {"enabled": False},
    }

    result = PipelineRunner(artifact_root=spec["artifact_root"]).run(spec)
    assert result.artifact is not None

    run_dir = Path(result.artifact.model_uri).parent
    assert (run_dir / "run_spec.json").exists()
    assert (run_dir / "stage_trace.json").exists()

    stage_trace = json.loads((run_dir / "stage_trace.json").read_text(encoding="utf-8"))
    assert stage_trace
    assert all("stage" in row and "status" in row for row in stage_trace)


def test_pipeline_runner_predict(tmp_path):
    train_spec = {
        "action": "train",
        "task": "classification",
        "artifact_root": str(tmp_path / "artifacts"),
        "random_state": 42,
        "source": {"type": "csv", "path": "dataset/classification/train.csv", "target": "target"},
        "split": {"strategy": "random", "valid_size": 0.2, "test_size": 0.2},
        "model": {"backend": "sklearn", "name": "logistic_regression", "params": {}},
        "feature_pipeline": [],
        "tune": {"enabled": False},
    }
    runner = PipelineRunner(artifact_root=train_spec["artifact_root"])
    train_result = runner.run(train_spec)
    model_uri = str(train_result.artifact.model_uri)

    output_path = tmp_path / "pred.csv"
    pred_spec = {
        "action": "predict",
        "model_uri": model_uri,
        "input": "dataset/classification/test.csv",
        "output": str(output_path),
    }
    pred_result = runner.run(pred_spec)

    assert output_path.exists()
    assert str(output_path) == pred_result.payload["output"]
