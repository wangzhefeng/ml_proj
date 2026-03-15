from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from mlproj.config import load_config
from mlproj.evaluation.evaluator import Evaluator
from mlproj.inference.predictor import Predictor
from mlproj.training.trainer import Trainer


def cmd_train(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    trainer = Trainer(artifact_root=cfg.get("artifact_root", "artifacts"))
    artifact = trainer.train(cfg)
    print(
        json.dumps(
            {
                "run_id": artifact.run_id,
                "model_uri": str(artifact.model_uri),
                "metrics_uri": str(artifact.metrics_uri),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def cmd_tune(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    cfg.setdefault("tune", {})["enabled"] = True
    trainer = Trainer(artifact_root=cfg.get("artifact_root", "artifacts"))
    artifact = trainer.train(cfg)
    print(
        json.dumps(
            {
                "run_id": artifact.run_id,
                "model_uri": str(artifact.model_uri),
                "params_uri": str(artifact.params_uri),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    predictor = Predictor(args.model_uri)
    output = predictor.predict_file(args.input, args.output)
    print(json.dumps({"output": str(output)}, ensure_ascii=False, indent=2))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from mlproj.inference.service import create_app

    app = create_app(args.model_uri)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def _read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _infer_task(estimator: Any, y_true: pd.Series | None) -> str:
    if hasattr(estimator, "cluster_centers_"):
        return "clustering"
    if y_true is not None and y_true.nunique(dropna=True) <= 30:
        return "classification"
    return "regression"


def _evaluate_with_model(
    model_uri: str,
    input_path: str,
    target_col: str | None,
    task: str | None,
    output_metrics: str | None,
) -> dict[str, Any]:
    predictor = Predictor(model_uri)
    evaluator = Evaluator()

    raw_df = _read_table(input_path)
    y_true = None
    X_raw = raw_df.copy()
    if target_col:
        if target_col not in raw_df.columns:
            raise ValueError(f"Target column not found in input: {target_col}")
        y_true = raw_df[target_col]
        X_raw = raw_df.drop(columns=[target_col])

    pre = predictor.bundle["preprocessor"]
    feats = predictor.bundle["features"]
    est = predictor.bundle["estimator"]

    X_eval = feats.transform(pre.transform(X_raw))
    y_pred = est.predict(X_eval)

    eval_task = task or _infer_task(est, y_true)
    if eval_task == "clustering":
        report = evaluator.evaluate(
            y_true=y_true,
            y_pred=y_pred,
            y_score=None,
            task="clustering",
            X_for_cluster=X_eval,
        )
    else:
        if y_true is None:
            raise ValueError("Supervised evaluate requires --target-col")
        y_score = None
        if hasattr(est, "predict_proba"):
            try:
                y_score = est.predict_proba(X_eval)
            except Exception:
                y_score = None
        report = evaluator.evaluate(y_true=y_true, y_pred=y_pred, y_score=y_score, task=eval_task)

    result = {
        "task": eval_task,
        "rows": int(len(X_raw)),
        "metrics": report.metrics,
        "model_uri": str(model_uri),
        "input": str(input_path),
    }
    if output_metrics:
        output_path = Path(output_metrics)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        result["metrics_output"] = str(output_path)
    return result


def cmd_evaluate(args: argparse.Namespace) -> int:
    if args.config:
        cfg = load_config(args.config)
        result = _evaluate_with_model(
            model_uri=cfg["model_uri"],
            input_path=cfg["input"],
            target_col=cfg.get("target_col"),
            task=cfg.get("task"),
            output_metrics=cfg.get("output_metrics"),
        )
    else:
        if not args.model_uri or not args.input:
            raise ValueError(
                "evaluate requires --model-uri and --input when --config is not provided"
            )
        result = _evaluate_with_model(
            model_uri=args.model_uri,
            input_path=args.input,
            target_col=args.target_col,
            task=args.task,
            output_metrics=args.output_metrics,
        )

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mlproj")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train a model and save artifacts")
    p_train.add_argument("--config", required=True)
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("evaluate", help="Evaluate using an existing trained model")
    p_eval.add_argument("--config", required=False)
    p_eval.add_argument("--model-uri", required=False)
    p_eval.add_argument("--input", required=False)
    p_eval.add_argument("--target-col", required=False)
    p_eval.add_argument(
        "--task",
        required=False,
        choices=["classification", "regression", "timeseries", "clustering"],
    )
    p_eval.add_argument("--output-metrics", required=False)
    p_eval.set_defaults(func=cmd_evaluate)

    p_predict = sub.add_parser("predict", help="Offline prediction for CSV/Parquet")
    p_predict.add_argument("--model-uri", required=True)
    p_predict.add_argument("--input", required=True)
    p_predict.add_argument("--output", required=True)
    p_predict.set_defaults(func=cmd_predict)

    p_tune = sub.add_parser("tune", help="Hyperparameter tuning")
    p_tune.add_argument("--config", required=True)
    p_tune.set_defaults(func=cmd_tune)

    p_serve = sub.add_parser("serve", help="Serve model with FastAPI")
    p_serve.add_argument("--model-uri", required=True)
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", default=8000, type=int)
    p_serve.set_defaults(func=cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
