from __future__ import annotations

from sklearn.metrics import adjusted_rand_score

try:
    from mlproj.analysis.metrics import clustering_metrics
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[1]
    SRC = ROOT / "src"
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from mlproj.analysis.metrics import clustering_metrics


class cluster_score:
    def __init__(self, labels_true, labels_pred, X=None):
        self.labels_true = labels_true
        self.labels_pred = labels_pred
        self.X = X

    def adjusted_rand_index(self):
        return float(adjusted_rand_score(self.labels_true, self.labels_pred))

    def davies_bouldin_index(self):
        return clustering_metrics(self.X, self.labels_pred, self.labels_true)["davies_bouldin"]

    def silhouette_coefficient(self):
        return clustering_metrics(self.X, self.labels_pred, self.labels_true)["silhouette"]


def main() -> None:
    print("metrics_cluster migrated to mlproj.analysis.metrics")


if __name__ == "__main__":
    main()
