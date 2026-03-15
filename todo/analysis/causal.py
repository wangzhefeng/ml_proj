from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LinearRegression


def estimate_ate_linear(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    control_cols: list[str] | None = None,
) -> float:
    """Estimate ATE via linear regression with backdoor adjustment."""
    control_cols = control_cols or []
    features = [treatment_col, *control_cols]
    missing = [c for c in [treatment_col, outcome_col, *control_cols] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    x = df[features]
    y = df[outcome_col]

    model = LinearRegression()
    model.fit(x, y)
    return float(model.coef_[0])


def run_causal_demo(n: int = 1000) -> dict[str, float]:
    """Build deterministic demo data and return estimated ATE."""
    demo = pd.DataFrame(
        {
            "w1": pd.Series(range(n)).sample(frac=1, random_state=42).reset_index(drop=True) / n,
        }
    )
    demo["w2"] = (demo["w1"] * 0.8 + 0.2).clip(0, 1)
    demo["treatment"] = (demo["w1"] + demo["w2"] > 1.0).astype(int)
    demo["outcome"] = 2.0 * demo["treatment"] + 1.5 * demo["w1"] + 0.8 * demo["w2"]

    ate = estimate_ate_linear(demo, "treatment", "outcome", ["w1", "w2"])
    return {"estimated_ate": ate}
