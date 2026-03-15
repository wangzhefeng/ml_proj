from __future__ import annotations

try:
    from mlproj.analysis.causal import run_causal_demo
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[1]
    SRC = ROOT / "src"
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from mlproj.analysis.causal import run_causal_demo


def main() -> None:
    result = run_causal_demo(n=1000)
    print(result)


if __name__ == "__main__":
    main()
