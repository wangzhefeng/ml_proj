from __future__ import annotations

import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def run_multiple_regression(
    data: pd.DataFrame,
    xcols: list[str],
    ycol: str,
    regul: str = "lasso",
    alpha: float = 0.1,
) -> pd.DataFrame:
    """Return standardized linear-model coefficients for selected features."""
    if not xcols:
        raise ValueError("xcols must not be empty")
    if ycol not in data.columns:
        raise ValueError(f"Target column not found: {ycol}")

    if regul == "ridge":
        model = Ridge(alpha=alpha)
        model_key = "ridge"
    elif regul == "lasso":
        model = Lasso(alpha=alpha)
        model_key = "lasso"
    else:
        model = LinearRegression()
        model_key = "linearregression"

    reg = make_pipeline(StandardScaler(), model)
    reg.fit(data[xcols], data[ycol])
    coefs = reg[model_key].coef_

    return pd.DataFrame({name: [coef] for name, coef in zip(xcols, coefs)})
