from __future__ import annotations

try:
    from mlproj.features.legacy_engine import downsample_majority, simple_over_under_sample
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[1]
    SRC = ROOT / "src"
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from mlproj.features.legacy_engine import downsample_majority, simple_over_under_sample


__all__ = ['downsample_majority', 'simple_over_under_sample']

def main() -> None:
    print("SampleImbalance migrated to mlproj.features.legacy_engine")


if __name__ == "__main__":
    main()

