from __future__ import annotations

try:
    from mlproj.legacy_models import run_quadratic_legacy_demo
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[1]
    SRC = ROOT / "src"
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from mlproj.legacy_models import run_quadratic_legacy_demo


def main() -> None:
    print(run_quadratic_legacy_demo())


if __name__ == "__main__":
    main()
