from __future__ import annotations

try:
    from mlproj.features.legacy_engine import Stacking
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[1]
    SRC = ROOT / "src"
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from mlproj.features.legacy_engine import Stacking


__all__ = ['Stacking']

def main() -> None:
    print("FeatureStacking migrated to mlproj.features.legacy_engine.Stacking")


if __name__ == "__main__":
    main()

