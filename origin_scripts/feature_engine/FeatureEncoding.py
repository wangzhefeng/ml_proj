from __future__ import annotations

try:
    from mlproj.features.legacy_engine import (
        CategoryFeatureEncoder,
        label_encoder,
        one_hot_encoder,
        oneHotEncoding,
        order_encoder,
    )
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[1]
    SRC = ROOT / "src"
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from mlproj.features.legacy_engine import (
        CategoryFeatureEncoder,
        label_encoder,
        one_hot_encoder,
        oneHotEncoding,
        order_encoder,
    )


__all__ = ['CategoryFeatureEncoder', 'oneHotEncoding', 'one_hot_encoder', 'order_encoder', 'label_encoder']

def main() -> None:
    import pandas as pd

    df = pd.DataFrame({"City": ["SF", "NYC", "SF"], "Rent": [1, 2, 3]})
    print(oneHotEncoding(df))
    print(label_encoder(df[["City"]]))
    print(CategoryFeatureEncoder.value_counts_encode(df["City"]))


if __name__ == "__main__":
    main()

