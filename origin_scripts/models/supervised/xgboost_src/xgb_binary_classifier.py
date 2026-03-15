from __future__ import annotations

try:
    from mlproj.legacy_models.subdir_models import run_subdir_legacy_demo
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    current = Path(__file__).resolve()
    root = None
    for candidate in [current.parent, *current.parents]:
        if (candidate / "src").exists():
            root = candidate
            break
    if root is None:
        raise RuntimeError("Cannot locate project root containing src/")

    src_path = root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    from mlproj.legacy_models.subdir_models import run_subdir_legacy_demo


def main() -> None:
    print(run_subdir_legacy_demo(__file__))


if __name__ == "__main__":
    main()
