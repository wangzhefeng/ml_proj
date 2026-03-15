from __future__ import annotations

try:
    from mlproj.data.legacy_provider import load_json_config
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[1]
    SRC = ROOT / "src"
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from mlproj.data.legacy_provider import load_json_config


__all__ = ['load_json_config']

def main() -> None:
    print("json_loader migrated to mlproj.data.legacy_provider")


if __name__ == "__main__":
    main()

