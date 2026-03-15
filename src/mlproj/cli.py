from __future__ import annotations

import argparse
import json

from mlproj.config import load_config
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


def cmd_evaluate(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    trainer = Trainer(artifact_root=cfg.get("artifact_root", "artifacts"))
    artifact = trainer.train(cfg)
    print(
        json.dumps(
            {
                "run_id": artifact.run_id,
                "metrics_uri": str(artifact.metrics_uri),
                "note": "evaluate executes train+valid evaluation in v1",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mlproj")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train a model and save artifacts")
    p_train.add_argument("--config", required=True)
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("evaluate", help="Run evaluation flow")
    p_eval.add_argument("--config", required=True)
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
