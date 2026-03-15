from __future__ import annotations

try:
    from mlproj.analysis.metrics import regression_metrics
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[1]
    SRC = ROOT / "src"
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from mlproj.analysis.metrics import regression_metrics


def MAE(y_true, y_pred):
    return regression_metrics(y_true, y_pred)["mae"]


def MSE(y_true, y_pred):
    return regression_metrics(y_true, y_pred)["mse"]


def RMSE(y_true, y_pred):
    return regression_metrics(y_true, y_pred)["rmse"]


def main() -> None:
    print("metrics_regression migrated to mlproj.analysis.metrics")


if __name__ == "__main__":
    main()
