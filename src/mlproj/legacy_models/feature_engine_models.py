from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mlproj.features.legacy_engine import (
    MissingPreprocessing,
    NumericCategoricalSplit,
    SkewedFeatures,
    binarization,
)


def run_feature_engine_legacy_demo(script_path: str) -> dict[str, object]:
    name = Path(script_path.replace("\\", "/").lower()).name
    df = pd.DataFrame(
        {
            "a": [0, 1, 2, 3, 4],
            "b": ["x", "y", "x", "z", "x"],
            "c": [1.0, np.nan, 3.0, 4.0, np.nan],
        }
    )

    if name == "featurebinning.py":
        out = binarization(df["a"], threshold=1.0)
        return {"script": script_path, "method": "binarization", "rows": int(out.shape[0])}

    if name == "featuresplit.py":
        _, num_cols, _, cat_cols = NumericCategoricalSplit(df, limit_value=2)
        return {
            "script": script_path,
            "method": "split",
            "num_cols": num_cols,
            "cat_cols": cat_cols,
        }

    if name == "samplemissing.py":
        filled = MissingPreprocessing(df).nan_fill(df, limit_value=2)
        return {"script": script_path, "method": "missing_fill", "shape": list(filled.shape)}

    if name == "normalitytransform.py":
        idx = SkewedFeatures(df.fillna(0), ["a", "c"], limit_value=0.1)
        return {"script": script_path, "method": "skewed_features", "features": list(idx)}

    return {"script": script_path, "method": "feature_engine_bridge"}
