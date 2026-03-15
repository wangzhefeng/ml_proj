from __future__ import annotations

try:
    from mlproj.features.legacy_engine import (
        col_filter,
        lda,
        low_variance_feature_remove,
        model_based_feature_selection,
        nan_feature_remove,
        pca,
    )
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[1]
    SRC = ROOT / "src"
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from mlproj.features.legacy_engine import (
        col_filter,
        lda,
        low_variance_feature_remove,
        model_based_feature_selection,
        nan_feature_remove,
        pca,
    )


__all__ = ['col_filter', 'lda', 'low_variance_feature_remove', 'model_based_feature_selection', 'nan_feature_remove', 'pca']

def main() -> None:
    print("FeatureSelection migrated to mlproj.features.legacy_engine")


if __name__ == "__main__":
    main()

