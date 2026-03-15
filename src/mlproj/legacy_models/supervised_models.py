from __future__ import annotations

from pathlib import Path

from mlproj.config import load_config
from mlproj.training.trainer import Trainer

CLASSIFICATION_HINTS = {
    "logisticregression.py",
    "decisiontree.py",
    "randomforest.py",
    "extra_trees.py",
    "bagging.py",
    "svm_classification.py",
    "lgb_binary_classifier.py",
    "lgb_multiclass_classifier.py",
    "lgb_pipeline.py",
    "xgb_binary_classifier.py",
    "xgb_multiclass_classifier.py",
    "bernoulli_nb.py",
    "complement_nb.py",
    "gaussian_nb.py",
    "multinomial_nb.py",
}

REGRESSION_HINTS = {
    "svm_regression.py",
    "ridge.py",
    "lasso.py",
    "elastic_net.py",
    "earlystopping.py",
    "xgb_regressor.py",
    "lgb_regressor.py",
    "pls.py",
    "ngbooster.py",
    "gbdt.py",
    "tuning.py",
}


def _infer_supervised_task(script_path: str) -> tuple[str, str, str]:
    sp = script_path.replace("\\", "/").lower()
    name = Path(sp).name

    if name in CLASSIFICATION_HINTS or "/nb/" in sp:
        return "classification", "configs/classification/train.yaml", "random_forest"
    if name in REGRESSION_HINTS or "regress" in name:
        return "regression", "configs/regression/train.yaml", "random_forest"

    return "classification", "configs/classification/train.yaml", "logistic_regression"


def run_supervised_legacy_demo(script_path: str) -> dict[str, str]:
    task, config_path, model_name = _infer_supervised_task(script_path)

    cfg = load_config(config_path)
    cfg["task"] = task
    cfg.setdefault("model", {})
    cfg["model"]["name"] = model_name
    cfg["model"]["params"] = {}

    trainer = Trainer(artifact_root=cfg.get("artifact_root", "artifacts"))
    artifact = trainer.train(cfg)

    return {
        "script": script_path,
        "task": task,
        "model_name": model_name,
        "run_id": artifact.run_id,
        "model_uri": str(artifact.model_uri),
        "metrics_uri": str(artifact.metrics_uri),
    }
