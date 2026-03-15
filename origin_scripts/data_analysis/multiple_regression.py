from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from mlproj.analysis.multiple_regression import run_multiple_regression
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[1]
    SRC = ROOT / "src"
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from mlproj.analysis.multiple_regression import run_multiple_regression


def main() -> None:
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "x1": np.random.randn(100),
            "x2": np.random.randn(100),
            "x3": np.random.randn(100),
            "y": np.random.randn(100),
        }
    )

    result_table = run_multiple_regression(data=df, xcols=["x1", "x2", "x3"], ycol="y")
    print(result_table.T)


if __name__ == "__main__":
    main()
