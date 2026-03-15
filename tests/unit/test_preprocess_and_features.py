import pandas as pd

from mlproj.features.pipeline import FeaturePipeline
from mlproj.preprocess.base import SklearnPreprocessor


def test_preprocess_feature_pipeline_consistency():
    df = pd.DataFrame({"num": [1.0, 2.0, 3.0, 4.0], "cat": ["a", "b", "a", "c"]})
    pre = SklearnPreprocessor().fit(df)
    x1 = pre.transform(df)

    feats = FeaturePipeline().fit(x1)
    x2 = feats.transform(x1)

    assert x1.shape[0] == x2.shape[0]
    assert x2.shape[1] == len(feats.selected_columns)
