from __future__ import annotations

from typing import Dict

try:
    from mlproj.data.legacy_provider import get_params, load_yaml
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[1]
    SRC = ROOT / "src"
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from mlproj.data.legacy_provider import get_params, load_yaml


__all__ = ["load_yaml", "get_params"]


def main() -> None:
    cfg_params: Dict = get_params("config.yaml")
    print(type(cfg_params), list(cfg_params.keys())[:3])


if __name__ == "__main__":
    main()
