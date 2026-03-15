from __future__ import annotations

from mlproj.legacy_models.supervised_models import run_supervised_legacy_demo
from mlproj.legacy_models.unsupervised_models import run_unsupervised_legacy_demo


def run_subdir_legacy_demo(script_path: str) -> dict[str, str]:
    sp = script_path.replace("\\", "/").lower()
    if "models/supervised/" in sp:
        return run_supervised_legacy_demo(script_path)
    if "models/unsupervised/" in sp:
        return run_unsupervised_legacy_demo(script_path)

    raise ValueError(f"Unsupported legacy script path: {script_path}")
