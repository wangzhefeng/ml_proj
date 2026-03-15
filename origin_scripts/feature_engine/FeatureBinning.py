from __future__ import annotations

try:
    from mlproj.features.legacy_engine import binarization, kbins
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[1]
    SRC = ROOT / "src"
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from mlproj.features.legacy_engine import binarization, kbins


def main() -> None:
    import numpy as np
    import pandas as pd

    df = pd.DataFrame({"a": range(10), "d": np.random.randn(10)})
    print(binarization(df["a"], threshold=1.0))
    print(kbins(df["d"], n_bins=5))


if __name__ == "__main__":
    main()
