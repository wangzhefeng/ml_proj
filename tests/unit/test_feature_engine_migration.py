import numpy as np
import pandas as pd

from mlproj.features import (
    MissingPreprocessing,
    NumericCategoricalSplit,
    binarization,
    kbins,
    normalizer_min_max_feature,
)


def test_feature_binning_and_split_bridge():
    df = pd.DataFrame(
        {
            "num": [0, 1, 2, 3, 4],
            "cat": ["a", "b", "a", "b", "c"],
            "score": [0.1, 0.5, 0.2, 0.9, 0.7],
        }
    )
    b = binarization(df["num"], threshold=1.5)
    assert b.shape[0] == len(df)

    k = kbins(df["score"], n_bins=3)
    assert k.shape[0] == len(df)

    num_df, num_cols, cat_df, cat_cols = NumericCategoricalSplit(df, limit_value=2)
    assert "num" in num_cols
    assert "cat" in cat_cols
    assert num_df.shape[0] == cat_df.shape[0] == len(df)


def test_missing_preprocess_bridge():
    df = pd.DataFrame({"a": [1.0, None, 3.0], "b": [None, 2.0, 3.0]})
    mp = MissingPreprocessing(df)

    simple = mp.simple_imputer()
    assert simple.shape == (3, 2)

    filled = mp.nan_fill(df, limit_value=2)
    assert filled.shape[0] == 3


def test_normalizer_feature_bridge():
    arr = np.array([1.0, 2.0, 3.0])
    out = normalizer_min_max_feature(arr)
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0

