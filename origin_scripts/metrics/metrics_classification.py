from __future__ import annotations

try:
    from mlproj.analysis.metrics import classification_metrics
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[1]
    SRC = ROOT / "src"
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from mlproj.analysis.metrics import classification_metrics


class binary_score:
    def Accuracy(self, y_true, y_pred):
        return classification_metrics(y_true, y_pred)["accuracy"]

    def Precision(self, y_true, y_pred):
        return classification_metrics(y_true, y_pred)["precision"]

    def Recall(self, y_true, y_pred):
        return classification_metrics(y_true, y_pred)["recall"]

    def F1(self, y_true, y_pred):
        return classification_metrics(y_true, y_pred)["f1"]


def main() -> None:
    print("metrics_classification migrated to mlproj.analysis.metrics")


if __name__ == "__main__":
    main()
