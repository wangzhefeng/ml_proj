from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mlproj.data.loader import DatasetLoader
from mlproj.evaluation.evaluator import Evaluator
from mlproj.features.pipeline import FeaturePipeline
from mlproj.models.factory import create_model
from mlproj.preprocess.base import SklearnPreprocessor
from mlproj.registry.artifact_store import ArtifactStore
from mlproj.selection.tuner import Tuner
from mlproj.types import TrainArtifact


class Trainer:
    def __init__(self, artifact_root: str = "artifacts") -> None:
        self.loader = DatasetLoader()
        self.preprocessor = SklearnPreprocessor()
        self.features = FeaturePipeline()
        self.evaluator = Evaluator()
        self.artifacts = ArtifactStore(root=artifact_root)
        self.tuner = Tuner()

    def train(self, run_config: dict[str, Any]) -> TrainArtifact:
        task = run_config["task"]
        model_name = run_config["model"]["name"]
        model_params = run_config["model"].get("params", {})

        data_bundle = self.loader.load(run_config)

        X_train = self.preprocessor.fit(data_bundle.X_train, data_bundle.y_train).transform(
            data_bundle.X_train
        )
        X_valid = (
            self.preprocessor.transform(data_bundle.X_valid)
            if data_bundle.X_valid is not None
            else None
        )

        X_train = self.features.fit(X_train, data_bundle.y_train).transform(X_train)
        X_valid = self.features.transform(X_valid) if X_valid is not None else None

        adapter = create_model(task=task, model_name=model_name, params=model_params)

        tune_cfg = run_config.get("tune", {})
        if tune_cfg.get("enabled", False):
            search = self.tuner.tune(
                adapter.model, X_train, data_bundle.y_train, {**tune_cfg, "task": task}
            )
            adapter.model = search.best_estimator_

        adapter.fit(X_train, data_bundle.y_train)

        if task == "clustering":
            y_pred = adapter.predict(X_train if X_valid is None else X_valid)
            X_metric = X_train if X_valid is None else X_valid
            report = self.evaluator.evaluate(
                y_true=None,
                y_pred=y_pred,
                task=task,
                X_for_cluster=X_metric,
            )
        else:
            if X_valid is None or data_bundle.y_valid is None:
                raise ValueError("Validation split with labels is required for supervised tasks")
            y_pred = adapter.predict(X_valid)
            y_score = None
            try:
                y_score = adapter.predict_proba(X_valid)
            except Exception:
                y_score = None
            report = self.evaluator.evaluate(
                data_bundle.y_valid, y_pred, y_score=y_score, task=task
            )

        final_params = dict(model_params)
        if hasattr(adapter.model, "get_params"):
            final_params = adapter.model.get_params(deep=False)

        artifact = self.artifacts.save_train_outputs(
            task=task,
            model_name=model_name,
            model={
                "preprocessor": self.preprocessor,
                "features": self.features,
                "estimator": adapter.model,
            },
            metrics=report.metrics,
            feature_columns=self.features.selected_columns or [],
            params=final_params,
        )

        summary = {
            "task": task,
            "model": model_name,
            "run_id": artifact.run_id,
            "metrics": report.metrics,
        }
        (Path(artifact.model_uri).parent / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        return artifact
