from __future__ import annotations

try:
    from mlproj.legacy_models import run_deploy_legacy_demo
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[2]
    SRC = ROOT / "src"
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from mlproj.legacy_models import run_deploy_legacy_demo


def main() -> None:
    out = run_deploy_legacy_demo(__file__)
    print(out)


if __name__ == "__main__":
    main()
