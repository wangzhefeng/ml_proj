from __future__ import annotations

try:
    from mlproj.features.legacy_engine import MissingPreprocessing
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[1]
    SRC = ROOT / "src"
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from mlproj.features.legacy_engine import MissingPreprocessing


__all__ = ['MissingPreprocessing']

def main() -> None:
    print("SampleMissing migrated to mlproj.features.legacy_engine.MissingPreprocessing")


if __name__ == "__main__":
    main()

