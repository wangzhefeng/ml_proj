from __future__ import annotations

try:
    from mlproj.selection.search import run_bayes_search_demo
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[1]
    SRC = ROOT / "src"
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from mlproj.selection.search import run_bayes_search_demo


def black_box_function(x, y):
    return -(x**2) - (y - 1) ** 2 + 1


def main() -> None:
    pbounds = {"x": (2, 4), "y": (-3, 3)}
    try:
        best = run_bayes_search_demo(black_box_function, pbounds=pbounds, init_points=2, n_iter=3)
        print(best)
    except RuntimeError as err:
        print(err)


if __name__ == "__main__":
    main()
