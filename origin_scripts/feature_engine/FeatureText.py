from __future__ import annotations

try:
    from mlproj.features.legacy_engine import split_text_feature_column
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[1]
    SRC = ROOT / "src"
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from mlproj.features.legacy_engine import split_text_feature_column


def example_1():
    import pandas as pd

    df = pd.DataFrame({"feature": ["Apple_iPhone_6", "Google_Pixel_3"]})
    df["feature_1st"] = split_text_feature_column(df["feature"], index=0)
    df["feature_2nd"] = split_text_feature_column(df["feature"], index=1)
    df["feature_3rd"] = split_text_feature_column(df["feature"], index=2)
    return df


def main() -> None:
    print(example_1())


if __name__ == "__main__":
    main()
