from __future__ import annotations

try:
    from mlproj.evaluation.legacy_metric import (
        ROC_plot,
        confusion_matrix_report,
        param_cvsearch_report,
        precision_recall_curve_report,
    )
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[1]
    SRC = ROOT / "src"
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from mlproj.evaluation.legacy_metric import (
        ROC_plot,
        confusion_matrix_report,
        param_cvsearch_report,
        precision_recall_curve_report,
    )


__all__ = ['precision_recall_curve_report', 'ROC_plot', 'confusion_matrix_report', 'param_cvsearch_report']

def main() -> None:
    print("metric_report migrated to mlproj.evaluation.legacy_metric")


if __name__ == "__main__":
    main()

