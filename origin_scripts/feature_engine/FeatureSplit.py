from __future__ import annotations

try:
    from mlproj.features.legacy_engine import NumericCategoricalSplit
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[1]
    SRC = ROOT / "src"
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from mlproj.features.legacy_engine import NumericCategoricalSplit


def main() -> None:
    import pandas as pd

    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    print(NumericCategoricalSplit(df, limit_value=2)[1:])


if __name__ == "__main__":
    main()
