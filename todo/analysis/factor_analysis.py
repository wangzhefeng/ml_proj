from __future__ import annotations

import pandas as pd
from sklearn.decomposition import FactorAnalysis


def run_factor_analysis(
    df: pd.DataFrame,
    drop_cols: list[str] | None = None,
    n_factors: int = 2,
) -> dict[str, pd.DataFrame]:
    """Run factor analysis and return loadings and factor scores."""
    drop_cols = drop_cols or []
    use_df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    use_df = use_df.select_dtypes(include=["number"]).copy()

    if use_df.shape[1] < 2:
        raise ValueError("Not enough numeric features for factor analysis")

    fa = FactorAnalysis(n_components=n_factors, random_state=42)
    transformed = fa.fit_transform(use_df)

    loadings = pd.DataFrame(
        fa.components_.T,
        index=use_df.columns,
        columns=[f"factor_{i + 1}" for i in range(n_factors)],
    )
    factor_scores = pd.DataFrame(
        transformed,
        columns=[f"factor_{i + 1}" for i in range(n_factors)],
        index=use_df.index,
    )

    return {
        "loadings": loadings,
        "factor_scores": factor_scores,
    }
