from __future__ import annotations

import argparse
import json

from mlproj.config import ConfigResolver
from mlproj.pipeline.runner import PipelineRunner

TASK_CHOICES = [
    "classification",
    "regression",
    "clustering",
    "pca_reduction",
    "anomaly_detection",
    "topic_modeling",
]
ACTION_CHOICES = ["train", "tune", "evaluate", "predict", "serve"]


def _build_resolver(args: argparse.Namespace) -> ConfigResolver:
    if args.config_yaml:
        resolver = ConfigResolver.from_yaml(args.config_yaml)
    elif args.config_module and args.config_class:
        resolver = ConfigResolver.from_python(args.config_module, args.config_class)
    else:
        raise ValueError(
            "Please provide config via --config-yaml or --config-module + --config-class"
        )
    return resolver


def _resolve_spec(args: argparse.Namespace, action: str) -> dict:
    resolver = _build_resolver(args)
    overrides = ConfigResolver.parse_override_items(getattr(args, "override", None))
    overrides["action"] = action

    # CLI explicit args override config values.
    for attr, key in [
        ("task", "task"),
        ("model_uri", "model_uri"),
        ("input", "input"),
        ("output", "output"),
        ("target_col", "target_col"),
        ("output_metrics", "output_metrics"),
        ("host", "host"),
        ("port", "port"),
    ]:
        value = getattr(args, attr, None)
        if value is not None:
            overrides[key] = value

    resolver.apply_overrides(overrides)
    return resolver.resolve()


def _run_action(spec: dict) -> int:
    runner = PipelineRunner(artifact_root=spec.get("artifact_root", "artifacts"))
    result = runner.run(spec)
    print(json.dumps(result.payload, ensure_ascii=False, indent=2))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    spec = _resolve_spec(args, args.action)
    return _run_action(spec)


def cmd_train(args: argparse.Namespace) -> int:
    spec = _resolve_spec(args, "train")
    return _run_action(spec)


def cmd_tune(args: argparse.Namespace) -> int:
    spec = _resolve_spec(args, "tune")
    return _run_action(spec)


def cmd_evaluate(args: argparse.Namespace) -> int:
    spec = _resolve_spec(args, "evaluate")
    return _run_action(spec)


def cmd_predict(args: argparse.Namespace) -> int:
    spec = _resolve_spec(args, "predict")
    return _run_action(spec)


def cmd_serve(args: argparse.Namespace) -> int:
    spec = _resolve_spec(args, "serve")
    return _run_action(spec)


def _add_config_source_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config-yaml", required=False)
    parser.add_argument("--config-module", required=False)
    parser.add_argument("--config-class", required=False)
    parser.add_argument("--override", action="append", default=[])


def _add_common_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--task", required=False, choices=TASK_CHOICES)
    parser.add_argument("--model-uri", required=False)
    parser.add_argument("--input", required=False)
    parser.add_argument("--output", required=False)
    parser.add_argument("--target-col", required=False)
    parser.add_argument("--output-metrics", required=False)
    parser.add_argument("--host", required=False)
    parser.add_argument("--port", required=False, type=int)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlproj",
        description="v2 CLI (breaking): use --config-yaml or --config-module/--config-class",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Unified pipeline entry")
    _add_config_source_args(p_run)
    _add_common_runtime_args(p_run)
    p_run.add_argument("--action", required=True, choices=ACTION_CHOICES)
    p_run.set_defaults(func=cmd_run)

    p_train = sub.add_parser("train", help="Thin wrapper -> run action=train")
    _add_config_source_args(p_train)
    _add_common_runtime_args(p_train)
    p_train.set_defaults(func=cmd_train)

    p_tune = sub.add_parser("tune", help="Thin wrapper -> run action=tune")
    _add_config_source_args(p_tune)
    _add_common_runtime_args(p_tune)
    p_tune.set_defaults(func=cmd_tune)

    p_eval = sub.add_parser("evaluate", help="Thin wrapper -> run action=evaluate")
    _add_config_source_args(p_eval)
    _add_common_runtime_args(p_eval)
    p_eval.set_defaults(func=cmd_evaluate)

    p_predict = sub.add_parser("predict", help="Thin wrapper -> run action=predict")
    _add_config_source_args(p_predict)
    _add_common_runtime_args(p_predict)
    p_predict.set_defaults(func=cmd_predict)

    p_serve = sub.add_parser("serve", help="Thin wrapper -> run action=serve")
    _add_config_source_args(p_serve)
    _add_common_runtime_args(p_serve)
    p_serve.set_defaults(func=cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
