from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from mlproj.evaluation.evaluator import Evaluator
from mlproj.models.hooks import ModelLifecycleHooks


@dataclass
class TaskStrategyResult:
    report: Any


class TaskStrategy(ABC):
    @abstractmethod
    def fit(self, adapter, X_train, y_train, hooks: ModelLifecycleHooks):
        ...

    @abstractmethod
    def evaluate(
        self,
        adapter,
        evaluator: Evaluator,
        data_bundle,
        X_train,
        X_valid,
        hooks: ModelLifecycleHooks,
    ) -> TaskStrategyResult:
        ...


class ClassificationStrategy(TaskStrategy):
    def fit(self, adapter, X_train, y_train, hooks: ModelLifecycleHooks):
        hooks.before_fit(adapter.model, X_train, y_train)
        adapter.fit(X_train, y_train)
        hooks.after_fit(adapter.model)

    def evaluate(self, adapter, evaluator, data_bundle, X_train, X_valid, hooks: ModelLifecycleHooks) -> TaskStrategyResult:
        if X_valid is None or data_bundle.y_valid is None:
            raise ValueError("Validation split with labels is required for supervised tasks")
        hooks.before_predict(adapter.model, X_valid)
        y_pred = adapter.predict(X_valid)
        hooks.after_predict(adapter.model, y_pred)
        y_score = None
        try:
            y_score = adapter.predict_proba(X_valid)
        except Exception:
            y_score = None
        report = evaluator.evaluate(
            y_true=data_bundle.y_valid,
            y_pred=y_pred,
            y_score=y_score,
            task="classification",
            estimator=adapter.model,
        )
        return TaskStrategyResult(report=report)


class RegressionStrategy(TaskStrategy):
    def fit(self, adapter, X_train, y_train, hooks: ModelLifecycleHooks):
        hooks.before_fit(adapter.model, X_train, y_train)
        adapter.fit(X_train, y_train)
        hooks.after_fit(adapter.model)

    def evaluate(self, adapter, evaluator, data_bundle, X_train, X_valid, hooks: ModelLifecycleHooks) -> TaskStrategyResult:
        if X_valid is None or data_bundle.y_valid is None:
            raise ValueError("Validation split with labels is required for supervised tasks")
        hooks.before_predict(adapter.model, X_valid)
        y_pred = adapter.predict(X_valid)
        hooks.after_predict(adapter.model, y_pred)
        report = evaluator.evaluate(
            y_true=data_bundle.y_valid,
            y_pred=y_pred,
            task="regression",
            estimator=adapter.model,
        )
        return TaskStrategyResult(report=report)


class ClusteringStrategy(TaskStrategy):
    def fit(self, adapter, X_train, y_train, hooks: ModelLifecycleHooks):
        hooks.before_fit(adapter.model, X_train, None)
        adapter.fit(X_train)
        hooks.after_fit(adapter.model)

    def evaluate(self, adapter, evaluator, data_bundle, X_train, X_valid, hooks: ModelLifecycleHooks) -> TaskStrategyResult:
        X_metric = X_valid if X_valid is not None else X_train
        y_metric = data_bundle.y_valid if data_bundle.y_valid is not None else data_bundle.y_train
        hooks.before_predict(adapter.model, X_metric)
        y_pred = adapter.predict(X_metric)
        hooks.after_predict(adapter.model, y_pred)
        report = evaluator.evaluate(
            y_true=y_metric,
            y_pred=y_pred,
            task="clustering",
            X_for_cluster=X_metric,
            estimator=adapter.model,
        )
        return TaskStrategyResult(report=report)


class PcaReductionStrategy(TaskStrategy):
    def fit(self, adapter, X_train, y_train, hooks: ModelLifecycleHooks):
        hooks.before_fit(adapter.model, X_train, None)
        adapter.fit(X_train)
        hooks.after_fit(adapter.model)

    def evaluate(self, adapter, evaluator, data_bundle, X_train, X_valid, hooks: ModelLifecycleHooks) -> TaskStrategyResult:
        X_metric = X_valid if X_valid is not None else X_train
        transformed = adapter.transform(X_metric)
        report = evaluator.evaluate(
            y_true=None,
            y_pred=transformed,
            y_score=None,
            task="pca_reduction",
            estimator=adapter.model,
        )
        return TaskStrategyResult(report=report)


class AnomalyDetectionStrategy(TaskStrategy):
    def fit(self, adapter, X_train, y_train, hooks: ModelLifecycleHooks):
        hooks.before_fit(adapter.model, X_train, None)
        adapter.fit(X_train)
        hooks.after_fit(adapter.model)

    def evaluate(self, adapter, evaluator, data_bundle, X_train, X_valid, hooks: ModelLifecycleHooks) -> TaskStrategyResult:
        X_metric = X_valid if X_valid is not None else X_train
        y_metric = data_bundle.y_valid if data_bundle.y_valid is not None else data_bundle.y_train
        hooks.before_predict(adapter.model, X_metric)
        y_pred = adapter.predict(X_metric)
        hooks.after_predict(adapter.model, y_pred)
        y_score = None
        try:
            y_score = adapter.score_samples(X_metric)
        except Exception:
            y_score = None
        report = evaluator.evaluate(
            y_true=y_metric,
            y_pred=y_pred,
            y_score=y_score,
            task="anomaly_detection",
            estimator=adapter.model,
        )
        return TaskStrategyResult(report=report)


class TopicModelingStrategy(TaskStrategy):
    def fit(self, adapter, X_train, y_train, hooks: ModelLifecycleHooks):
        hooks.before_fit(adapter.model, X_train, None)
        adapter.fit(X_train)
        hooks.after_fit(adapter.model)

    def evaluate(self, adapter, evaluator, data_bundle, X_train, X_valid, hooks: ModelLifecycleHooks) -> TaskStrategyResult:
        X_metric = X_valid if X_valid is not None else X_train
        topic_dist = adapter.transform(X_metric)
        topic_pred = np.argmax(topic_dist, axis=1)
        report = evaluator.evaluate(
            y_true=None,
            y_pred=topic_pred,
            y_score=topic_dist,
            task="topic_modeling",
            estimator=adapter.model,
        )
        return TaskStrategyResult(report=report)


def get_task_strategy(task: str) -> TaskStrategy:
    mapping: dict[str, TaskStrategy] = {
        "classification": ClassificationStrategy(),
        "regression": RegressionStrategy(),
        "clustering": ClusteringStrategy(),
        "pca_reduction": PcaReductionStrategy(),
        "anomaly_detection": AnomalyDetectionStrategy(),
        "topic_modeling": TopicModelingStrategy(),
    }
    if task not in mapping:
        raise ValueError(f"Unsupported task strategy: {task}")
    return mapping[task]
