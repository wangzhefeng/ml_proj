from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from mlproj.analysis.factor_analysis import run_factor_analysis
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[1]
    SRC = ROOT / "src"
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from mlproj.analysis.factor_analysis import run_factor_analysis


def main() -> None:
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "feat_1": rng.normal(size=200),
            "feat_2": rng.normal(size=200),
            "feat_3": rng.normal(size=200),
            "feat_4": rng.normal(size=200),
        }
    )

    out = run_factor_analysis(df, n_factors=2)
    print("loadings:")
    print(out["loadings"])


if __name__ == "__main__":
    main()
