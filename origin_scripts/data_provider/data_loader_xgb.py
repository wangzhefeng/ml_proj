from __future__ import annotations

try:
    from mlproj.data.legacy_provider import get_xgb_train_test_data
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[1]
    SRC = ROOT / "src"
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from mlproj.data.legacy_provider import get_xgb_train_test_data


__all__ = ['get_xgb_train_test_data']

def main() -> None:
    print("data_loader_xgb migrated to mlproj.data.legacy_provider")


if __name__ == "__main__":
    main()

