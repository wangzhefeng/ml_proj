from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, Callable


@dataclass(slots=True)
class BackendSpec:
    name: str
    adapter_factory: Callable[[str, str, dict[str, Any]], Any]
    supported_tasks: set[str] = field(default_factory=set)
    capabilities: set[str] = field(default_factory=set)


class BackendRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, BackendSpec] = {}

    def register_backend(
        self,
        name: str,
        adapter_factory: Callable[[str, str, dict[str, Any]], Any],
        supported_tasks: set[str],
        capabilities: set[str] | None = None,
        override: bool = False,
    ) -> None:
        key = name.lower().strip()
        if key in self._specs and not override:
            raise ValueError(f"Backend already registered: {name}")
        self._specs[key] = BackendSpec(
            name=key,
            adapter_factory=adapter_factory,
            supported_tasks=set(supported_tasks),
            capabilities=set(capabilities or set()),
        )

    def get_backend(self, name: str) -> BackendSpec:
        key = name.lower().strip()
        if key not in self._specs:
            available = ", ".join(sorted(self._specs.keys())) or "<none>"
            raise ValueError(f"Unknown backend: {name}. Available backends: {available}")
        return self._specs[key]

    def list_backends(self) -> list[str]:
        return sorted(self._specs.keys())

    def load_backend_provider(self, module_path: str) -> None:
        module = import_module(module_path)
        register_fn = getattr(module, "register_backends", None)
        if register_fn is None or not callable(register_fn):
            raise ValueError(
                "Backend provider module must define callable register_backends(registry)"
            )
        register_fn(self)


global_backend_registry = BackendRegistry()
