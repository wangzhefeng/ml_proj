from __future__ import annotations

from pathlib import Path

from mlproj.config import load_config
from mlproj.training.trainer import Trainer


def _infer_unsupervised_model(script_path: str) -> str:
    name = Path(script_path.replace("\\", "/").lower()).name
    if name in {"mini_batch_kmeans.py", "bisecting_kmeans.py"}:
        return "kmeans_small"
    return "kmeans"


def run_unsupervised_legacy_demo(script_path: str) -> dict[str, str]:
    model_name = _infer_unsupervised_model(script_path)

    cfg = load_config("configs/clustering/train.yaml")
    cfg["task"] = "clustering"
    cfg.setdefault("model", {})
    cfg["model"]["name"] = model_name
    cfg["model"]["params"] = {}

    trainer = Trainer(artifact_root=cfg.get("artifact_root", "artifacts"))
    artifact = trainer.train(cfg)

    return {
        "script": script_path,
        "task": "clustering",
        "model_name": model_name,
        "run_id": artifact.run_id,
        "model_uri": str(artifact.model_uri),
        "metrics_uri": str(artifact.metrics_uri),
    }
