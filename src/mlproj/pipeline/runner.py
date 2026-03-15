from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
import sklearn

from mlproj.data.loader import DatasetLoader
from mlproj.evaluation.evaluator import Evaluator
from mlproj.features.pipeline import FeaturePipeline
from mlproj.inference.predictor import Predictor
from mlproj.models.factory import create_model
from mlproj.models.hooks import ModelLifecycleHooks
from mlproj.preprocess.base import IdentityPreprocessor, SklearnPreprocessor
from mlproj.registry.artifact_store import ArtifactStore
from mlproj.selection.tuner import Tuner
from mlproj.training.task_strategies import get_task_strategy
from mlproj.types import DatasetBundle, MetricReport, TrainArtifact


@dataclass
class StageTrace:
    stage: str
    status: str
    duration_ms: float
    error: str | None = None


@dataclass
class RunContext:
    spec: dict[str, Any]
    trace: list[StageTrace] = field(default_factory=list)

    data_bundle: DatasetBundle | None = None
    preprocessor: Any | None = None
    features: FeaturePipeline | None = None
    X_train: Any | None = None
    X_valid: Any | None = None
    adapter: Any | None = None
    report: MetricReport | None = None
    artifact: TrainArtifact | None = None


@dataclass
class RunResult:
    action: str
    payload: dict[str, Any] = field(default_factory=dict)
    artifact: TrainArtifact | None = None


class Stage(Protocol):
    name: str

    def execute(self, ctx: RunContext, runner: "PipelineRunner") -> RunContext:
        ...


@dataclass
class LoadStage:
    name: str = "load"

    def execute(self, ctx: RunContext, runner: "PipelineRunner") -> RunContext:
        runner.loader.random_state = int(ctx.spec.get("random_state", runner.loader.random_state))
        ctx.data_bundle = runner.loader.load(ctx.spec)
        return ctx


@dataclass
class PreprocessStage:
    name: str = "preprocess"

    def execute(self, ctx: RunContext, runner: "PipelineRunner") -> RunContext:
        task = ctx.spec["task"]
        ctx.preprocessor = IdentityPreprocessor() if task == "topic_modeling" else SklearnPreprocessor()
        data_bundle = ctx.data_bundle
        if data_bundle is None:
            raise RuntimeError("data_bundle is required before preprocess")

        ctx.X_train = ctx.preprocessor.fit(data_bundle.X_train, data_bundle.y_train).transform(data_bundle.X_train)
        ctx.X_valid = (
            ctx.preprocessor.transform(data_bundle.X_valid) if data_bundle.X_valid is not None else None
        )
        return ctx


@dataclass
class FeatureStage:
    name: str = "feature"

    def execute(self, ctx: RunContext, runner: "PipelineRunner") -> RunContext:
        data_bundle = ctx.data_bundle
        if data_bundle is None or ctx.X_train is None:
            raise RuntimeError("feature stage requires loaded and preprocessed data")

        ctx.features = FeaturePipeline(steps=ctx.spec.get("feature_pipeline", []))
        ctx.X_train = ctx.features.fit(ctx.X_train, data_bundle.y_train).transform(ctx.X_train)
        ctx.X_valid = ctx.features.transform(ctx.X_valid) if ctx.X_valid is not None else None
        return ctx


@dataclass
class ModelStage:
    name: str = "model"

    def execute(self, ctx: RunContext, runner: "PipelineRunner") -> RunContext:
        model_cfg = ctx.spec["model"]
        ctx.adapter = create_model(
            task=ctx.spec["task"],
            model_name=model_cfg["name"],
            params=model_cfg.get("params", {}),
            backend=model_cfg["backend"],
            backend_provider=model_cfg.get("backend_provider"),
        )
        return ctx


@dataclass
class TuneStage:
    name: str = "tune"

    def execute(self, ctx: RunContext, runner: "PipelineRunner") -> RunContext:
        if ctx.adapter is None or ctx.X_train is None or ctx.data_bundle is None:
            raise RuntimeError("tune stage requires model and prepared train data")

        tune_cfg = dict(ctx.spec.get("tune", {}))
        if not tune_cfg.get("enabled", False):
            return ctx

        search = runner.tuner.tune(
            ctx.adapter.model,
            ctx.X_train,
            ctx.data_bundle.y_train,
            {**tune_cfg, "task": ctx.spec["task"]},
        )
        ctx.adapter.model = search.best_estimator_
        return ctx


@dataclass
class FitEvaluateStage:
    name: str = "evaluate"

    def execute(self, ctx: RunContext, runner: "PipelineRunner") -> RunContext:
        if ctx.adapter is None or ctx.data_bundle is None or ctx.X_train is None:
            raise RuntimeError("fit/evaluate stage requires model and train data")

        strategy = get_task_strategy(ctx.spec["task"])
        strategy.fit(ctx.adapter, ctx.X_train, ctx.data_bundle.y_train, runner.hooks)
        result = strategy.evaluate(
            ctx.adapter,
            runner.evaluator,
            ctx.data_bundle,
            ctx.X_train,
            ctx.X_valid,
            runner.hooks,
        )
        ctx.report = result.report
        return ctx


@dataclass
class PersistStage:
    name: str = "persist"

    def execute(self, ctx: RunContext, runner: "PipelineRunner") -> RunContext:
        if ctx.adapter is None or ctx.report is None or ctx.features is None or ctx.preprocessor is None:
            raise RuntimeError("persist stage requires model/report/features/preprocessor")

        model_cfg = ctx.spec["model"]
        model_name = model_cfg["name"]
        model_backend = model_cfg["backend"]

        final_params = dict(model_cfg.get("params", {}))
        if hasattr(ctx.adapter.model, "get_params"):
            final_params = ctx.adapter.model.get_params(deep=False)

        ctx.artifact = runner.artifacts.save_train_outputs(
            task=ctx.spec["task"],
            model_name=f"{model_backend}_{model_name}",
            model={
                "task": ctx.spec["task"],
                "backend": model_backend,
                "preprocessor": ctx.preprocessor,
                "features": ctx.features,
                "estimator": ctx.adapter.model,
            },
            metrics=ctx.report.metrics,
            feature_columns=ctx.features.selected_columns or [],
            params=final_params,
        )
        return ctx


class PipelineRunner:
    def __init__(self, artifact_root: str = "artifacts") -> None:
        self.loader = DatasetLoader()
        self.evaluator = Evaluator()
        self.tuner = Tuner()
        self.artifacts = ArtifactStore(root=artifact_root)
        self.hooks = ModelLifecycleHooks()

    def run(self, run_spec: dict[str, Any]) -> RunResult:
        action = run_spec["action"]
        self.artifacts.root = Path(run_spec.get("artifact_root", self.artifacts.root))

        if action in {"train", "tune"}:
            return self._run_train_like(run_spec)
        if action == "evaluate":
            return self._run_evaluate(run_spec)
        if action == "predict":
            return self._run_predict(run_spec)
        if action == "serve":
            return self._run_serve(run_spec)

        raise ValueError(f"Unsupported action: {action}")

    def _run_train_like(self, spec: dict[str, Any]) -> RunResult:
        ctx = RunContext(spec=spec)
        stages: list[Stage] = [
            LoadStage(),
            PreprocessStage(),
            FeatureStage(),
            ModelStage(),
            TuneStage(),
            FitEvaluateStage(),
            PersistStage(),
        ]

        for stage in stages:
            self._run_stage(stage, ctx)

        if ctx.artifact is None or ctx.report is None:
            raise RuntimeError("Train pipeline did not produce artifact/report")

        self._write_train_side_outputs(ctx)

        payload = {
            "run_id": ctx.artifact.run_id,
            "model_uri": str(ctx.artifact.model_uri),
            "metrics_uri": str(ctx.artifact.metrics_uri),
            "action": spec["action"],
        }
        return RunResult(action=spec["action"], payload=payload, artifact=ctx.artifact)

    def _run_stage(self, stage: Stage, ctx: RunContext) -> None:
        start = time.perf_counter()
        try:
            stage.execute(ctx, self)
            duration_ms = (time.perf_counter() - start) * 1000
            ctx.trace.append(StageTrace(stage=stage.name, status="ok", duration_ms=duration_ms))
        except Exception as exc:
            duration_ms = (time.perf_counter() - start) * 1000
            ctx.trace.append(
                StageTrace(stage=stage.name, status="error", duration_ms=duration_ms, error=str(exc))
            )
            raise

    def _write_train_side_outputs(self, ctx: RunContext) -> None:
        if ctx.artifact is None or ctx.report is None:
            return

        run_dir = Path(ctx.artifact.model_uri).parent
        run_metadata = self._build_run_metadata(ctx.spec, ctx.data_bundle, int(ctx.spec.get("random_state", 42)))

        summary = {
            "task": ctx.spec["task"],
            "backend": ctx.spec["model"]["backend"],
            "model": ctx.spec["model"]["name"],
            "run_id": ctx.artifact.run_id,
            "metrics": ctx.report.metrics,
            "run_metadata": run_metadata,
        }
        stage_trace = [
            {
                "stage": entry.stage,
                "status": entry.status,
                "duration_ms": entry.duration_ms,
                "error": entry.error,
            }
            for entry in ctx.trace
        ]

        (run_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (run_dir / "run_spec.json").write_text(
            json.dumps(ctx.spec, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        (run_dir / "stage_trace.json").write_text(
            json.dumps(stage_trace, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _run_evaluate(self, spec: dict[str, Any]) -> RunResult:
        predictor = Predictor(spec["model_uri"])
        est = predictor.bundle["estimator"]

        raw_df = self._read_table(spec["input"])
        y_true = None
        X_raw = raw_df.copy()
        if spec.get("target_col"):
            target_col = spec["target_col"]
            if target_col not in raw_df.columns:
                raise ValueError(f"Target column not found in input: {target_col}")
            y_true = raw_df[target_col]
            X_raw = raw_df.drop(columns=[target_col])

        X_eval = X_raw
        pre = predictor.bundle.get("preprocessor")
        feats = predictor.bundle.get("features")
        if pre is not None:
            X_eval = pre.transform(X_eval)
        if feats is not None:
            X_eval = feats.transform(X_eval)

        eval_task = spec.get("task") or predictor.bundle.get("task") or self._infer_task(est, y_true)
        metrics = self._evaluate_by_task(eval_task, est, X_eval, y_true)

        payload = {
            "task": eval_task,
            "rows": int(len(X_raw)),
            "metrics": metrics,
            "model_uri": str(spec["model_uri"]),
            "input": str(spec["input"]),
        }

        if spec.get("output_metrics"):
            output_path = Path(spec["output_metrics"])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            payload["metrics_output"] = str(output_path)

        return RunResult(action="evaluate", payload=payload)

    def _run_predict(self, spec: dict[str, Any]) -> RunResult:
        predictor = Predictor(spec["model_uri"])
        output = predictor.predict_file(spec["input"], spec["output"])
        return RunResult(action="predict", payload={"output": str(output)})

    def _run_serve(self, spec: dict[str, Any]) -> RunResult:
        import uvicorn

        from mlproj.inference.service import create_app

        app = create_app(spec["model_uri"])
        uvicorn.run(app, host=spec.get("host", "127.0.0.1"), port=int(spec.get("port", 8000)))
        return RunResult(action="serve", payload={"status": "started"})

    def _evaluate_by_task(self, task: str, estimator, X_eval, y_true):
        if task == "clustering":
            y_pred = estimator.predict(X_eval)
            report = self.evaluator.evaluate(
                y_true=y_true,
                y_pred=y_pred,
                task="clustering",
                X_for_cluster=X_eval,
                estimator=estimator,
            )
            return report.metrics

        if task == "pca_reduction":
            transformed = estimator.transform(X_eval)
            report = self.evaluator.evaluate(
                y_true=None,
                y_pred=transformed,
                y_score=None,
                task=task,
                estimator=estimator,
            )
            return report.metrics

        if task == "anomaly_detection":
            y_pred = estimator.predict(X_eval)
            y_score = estimator.score_samples(X_eval) if hasattr(estimator, "score_samples") else None
            report = self.evaluator.evaluate(
                y_true=y_true,
                y_pred=y_pred,
                y_score=y_score,
                task=task,
                estimator=estimator,
            )
            return report.metrics

        if task == "topic_modeling":
            topic_dist = estimator.transform(X_eval)
            topic_pred = np.argmax(topic_dist, axis=1)
            report = self.evaluator.evaluate(
                y_true=None,
                y_pred=topic_pred,
                y_score=topic_dist,
                task=task,
                estimator=estimator,
            )
            return report.metrics

        if y_true is None:
            raise ValueError("Supervised evaluate requires target_col")

        y_pred = estimator.predict(X_eval)
        y_score = None
        if hasattr(estimator, "predict_proba"):
            try:
                y_score = estimator.predict_proba(X_eval)
            except Exception:
                y_score = None
        report = self.evaluator.evaluate(y_true=y_true, y_pred=y_pred, y_score=y_score, task=task)
        return report.metrics

    def _infer_task(self, estimator: Any, y_true: pd.Series | None) -> str:
        if hasattr(estimator, "vectorizer") and hasattr(estimator, "model"):
            return "topic_modeling"
        if hasattr(estimator, "explained_variance_ratio_"):
            return "pca_reduction"
        if hasattr(estimator, "score_samples") and not hasattr(estimator, "predict_proba"):
            return "anomaly_detection"
        if hasattr(estimator, "cluster_centers_"):
            return "clustering"
        if y_true is not None and y_true.nunique(dropna=True) <= 30:
            return "classification"
        return "regression"

    def _read_table(self, path: str | Path) -> pd.DataFrame:
        path = Path(path)
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path)

    def _build_run_metadata(
        self,
        run_config: dict[str, Any],
        data_bundle: DatasetBundle | None,
        random_seed: int,
    ) -> dict[str, Any]:
        source = run_config.get("source", {})

        dataset_info = {
            "train_rows": 0,
            "valid_rows": 0,
            "test_rows": 0,
            "feature_count": 0,
        }
        if data_bundle is not None:
            dataset_info = {
                "train_rows": int(len(data_bundle.X_train)),
                "valid_rows": int(len(data_bundle.X_valid)) if data_bundle.X_valid is not None else 0,
                "test_rows": int(len(data_bundle.X_test)) if data_bundle.X_test is not None else 0,
                "feature_count": int(data_bundle.X_train.shape[1]),
            }

        return {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "random_seed": random_seed,
            "data_version": source.get("data_version") or self._infer_data_version(source),
            "dataset": dataset_info,
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit_learn": sklearn.__version__,
            },
        }

    def _infer_data_version(self, source: dict[str, Any]) -> dict[str, Any] | str:
        files = [
            source.get("path"),
            source.get("train_path"),
            source.get("valid_path"),
            source.get("test_path"),
        ]
        file_infos: list[dict[str, Any]] = []
        for file_path in files:
            if not file_path:
                continue
            path_obj = Path(file_path)
            if path_obj.exists():
                stat = path_obj.stat()
                file_infos.append(
                    {
                        "path": str(path_obj),
                        "size": int(stat.st_size),
                        "modified_utc": datetime.fromtimestamp(
                            stat.st_mtime, tz=timezone.utc
                        ).isoformat(),
                    }
                )
        if file_infos:
            return {"files": file_infos}
        return "unknown"
