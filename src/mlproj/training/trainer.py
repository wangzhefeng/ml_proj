from __future__ import annotations

from typing import Any

from mlproj.pipeline.runner import PipelineRunner
from mlproj.types import TrainArtifact


class Trainer:
    def __init__(self, artifact_root: str = "artifacts") -> None:
        self.runner = PipelineRunner(artifact_root=artifact_root)

    def train(self, run_config: dict[str, Any]) -> TrainArtifact:
        spec = dict(run_config)
        if "action" not in spec:
            tune_enabled = bool(spec.get("tune", {}).get("enabled", False))
            spec["action"] = "tune" if tune_enabled else "train"

        result = self.runner.run(spec)
        if result.artifact is None:
            raise RuntimeError("Train pipeline did not produce artifact")
        return result.artifact
