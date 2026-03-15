from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(slots=True)
class DatasetBundle:
    X_train: pd.DataFrame
    y_train: pd.Series | None
    X_valid: pd.DataFrame | None = None
    y_valid: pd.Series | None = None
    X_test: pd.DataFrame | None = None
    y_test: pd.Series | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MetricReport:
    task: str
    metrics: dict[str, float]


@dataclass(slots=True)
class TrainArtifact:
    task: str
    model: str
    run_id: str
    model_uri: Path
    metrics_uri: Path
    feature_schema_uri: Path
    params_uri: Path
