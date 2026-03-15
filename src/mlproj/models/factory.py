from __future__ import annotations

from typing import Any

from .backend_registry import global_backend_registry
from .backends import (
    build_catboost_adapter,
    build_lightgbm_adapter,
    build_sklearn_adapter,
    build_xgboost_adapter,
)

_REGISTERED = False

_REQUIRED_CAPABILITIES = {
    "classification": {"predict"},
    "regression": {"predict"},
    "clustering": {"predict"},
    "pca_reduction": {"transform"},
    "anomaly_detection": {"predict"},
    "topic_modeling": {"transform"},
}


def _ensure_builtin_backends() -> None:
    global _REGISTERED
    if _REGISTERED:
        return

    all_tasks = {
        "classification",
        "regression",
        "clustering",
        "pca_reduction",
        "anomaly_detection",
        "topic_modeling",
    }
    global_backend_registry.register_backend(
        name="sklearn",
        adapter_factory=build_sklearn_adapter,
        supported_tasks=all_tasks,
        capabilities={"predict", "predict_proba", "decision_function", "transform", "score_samples"},
    )
    global_backend_registry.register_backend(
        name="lightgbm",
        adapter_factory=build_lightgbm_adapter,
        supported_tasks={"classification", "regression"},
        capabilities={"predict", "predict_proba"},
    )
    global_backend_registry.register_backend(
        name="xgboost",
        adapter_factory=build_xgboost_adapter,
        supported_tasks={"classification", "regression"},
        capabilities={"predict", "predict_proba"},
    )
    global_backend_registry.register_backend(
        name="catboost",
        adapter_factory=build_catboost_adapter,
        supported_tasks={"classification", "regression"},
        capabilities={"predict", "predict_proba"},
    )
    _REGISTERED = True


def register_backend(
    name: str,
    adapter_factory,
    supported_tasks: set[str],
    capabilities: set[str] | None = None,
    override: bool = False,
) -> None:
    _ensure_builtin_backends()
    global_backend_registry.register_backend(
        name=name,
        adapter_factory=adapter_factory,
        supported_tasks=supported_tasks,
        capabilities=capabilities,
        override=override,
    )


def get_backend(name: str):
    _ensure_builtin_backends()
    return global_backend_registry.get_backend(name)


def list_backends() -> list[str]:
    _ensure_builtin_backends()
    return global_backend_registry.list_backends()


def _validate_backend_capabilities(task: str, backend: str, declared_caps: set[str], adapter_model: Any) -> None:
    required = _REQUIRED_CAPABILITIES.get(task, set())
    missing_declared = sorted(cap for cap in required if cap not in declared_caps)
    if missing_declared:
        raise ValueError(
            f"Backend '{backend}' does not declare required capabilities for task '{task}': {missing_declared}"
        )

    missing_runtime = sorted(cap for cap in required if not hasattr(adapter_model, cap))
    if missing_runtime:
        raise ValueError(
            f"Model instance from backend '{backend}' misses required runtime methods for task '{task}': {missing_runtime}"
        )


def create_model(
    task: str,
    model_name: str,
    params: dict[str, Any] | None = None,
    backend: str = "sklearn",
    backend_provider: str | None = None,
):
    _ensure_builtin_backends()
    if backend_provider:
        global_backend_registry.load_backend_provider(backend_provider)

    spec = global_backend_registry.get_backend(backend)
    if task not in spec.supported_tasks:
        raise ValueError(f"Backend '{backend}' does not support task '{task}'")

    adapter = spec.adapter_factory(task, model_name, params or {})
    _validate_backend_capabilities(task, backend, spec.capabilities, adapter.model)
    return adapter
