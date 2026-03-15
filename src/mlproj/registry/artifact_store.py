from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib

from mlproj.types import TrainArtifact


class ArtifactStore:
    def __init__(self, root: str | Path = "artifacts") -> None:
        self.root = Path(root)

    def create_run_dir(self, task: str, model: str, run_id: str | None = None) -> tuple[str, Path]:
        run = run_id or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        run_dir = self.root / task / model / run
        run_dir.mkdir(parents=True, exist_ok=True)
        return run, run_dir

    def save_train_outputs(
        self,
        task: str,
        model_name: str,
        model: Any,
        metrics: dict[str, float],
        feature_columns: list[str],
        params: dict[str, Any],
    ) -> TrainArtifact:
        run_id, run_dir = self.create_run_dir(task, model_name)
        model_uri = run_dir / "model.joblib"
        metrics_uri = run_dir / "metrics.json"
        feature_schema_uri = run_dir / "feature_schema.json"
        params_uri = run_dir / "params.json"

        joblib.dump(model, model_uri)
        metrics_uri.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        feature_schema_uri.write_text(
            json.dumps(feature_columns, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        params_uri.write_text(
            json.dumps(params, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )

        return TrainArtifact(
            task=task,
            model=model_name,
            run_id=run_id,
            model_uri=model_uri,
            metrics_uri=metrics_uri,
            feature_schema_uri=feature_schema_uri,
            params_uri=params_uri,
        )
