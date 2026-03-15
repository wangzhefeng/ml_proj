from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr


@dataclass
class CorrHypotheticalTest:
    df: pd.DataFrame
    xcol: str
    ycol: str
    alpha: float = 0.05
    null_info: str = "Variables are independent."
    alter_info: str = "Variables may be correlated."

    def _run_test(self, p_value: float) -> str:
        if p_value > self.alpha:
            return f"Cannot reject null hypothesis: {self.null_info}"
        return f"Reject null hypothesis: {self.alter_info}"

    def _coef(self, method: str) -> float:
        return float(self.df[[self.xcol, self.ycol]].corr(method=method).iloc[0, 1])

    def pearson_test(self) -> tuple[float, float, float, str]:
        stat, p = pearsonr(self.df[self.xcol].values, self.df[self.ycol].values)
        return float(stat), float(p), self._coef("pearson"), self._run_test(float(p))

    def spearman_test(self) -> tuple[float, float, float, str]:
        stat, p = spearmanr(self.df[self.xcol].values, self.df[self.ycol].values)
        return float(stat), float(p), self._coef("spearman"), self._run_test(float(p))

    def kendalltau_test(self) -> tuple[float, float, float, str]:
        stat, p = kendalltau(self.df[self.xcol].values, self.df[self.ycol].values)
        return float(stat), float(p), self._coef("kendall"), self._run_test(float(p))


def corr_test(
    df: pd.DataFrame,
    xcols: list[str],
    ycols: list[str],
    alpha: float = 0.05,
    methods: tuple[str, ...] = ("pearson",),
) -> pd.DataFrame:
    """Batch correlation hypothesis test."""
    if len(xcols) != len(ycols):
        raise ValueError("xcols and ycols must have the same length")

    rows: list[dict[str, object]] = []
    for xcol, ycol in zip(xcols, ycols):
        tester = CorrHypotheticalTest(df=df, xcol=xcol, ycol=ycol, alpha=alpha)
        for method in methods:
            if method == "pearson":
                stat, p, coef, res = tester.pearson_test()
            elif method == "spearman":
                stat, p, coef, res = tester.spearman_test()
            elif method == "kendall":
                stat, p, coef, res = tester.kendalltau_test()
            else:
                raise ValueError(f"Unsupported method: {method}")

            rows.append(
                {
                    "var_1": xcol,
                    "var_2": ycol,
                    "method": method,
                    "stat": stat,
                    "p_value": p,
                    "corr_coef": coef,
                    "test_result": res,
                }
            )

    return pd.DataFrame(rows)
