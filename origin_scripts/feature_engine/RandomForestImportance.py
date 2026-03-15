from __future__ import annotations

try:
    from mlproj.features.legacy_engine import (
        random_forest_importance_classifier,
        random_forest_importance_regressor,
    )
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[1]
    SRC = ROOT / "src"
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from mlproj.features.legacy_engine import (
        random_forest_importance_classifier,
        random_forest_importance_regressor,
    )


def RandomForestClf(x_train, y_train, feature_labels, threshold: float = 0.15):
    return random_forest_importance_classifier(x_train, y_train, feature_labels, threshold)[
        "x_selected"
    ]


def RandomForestReg(x_train, y_train, feature_labels, threshold: float = 0.15):
    return random_forest_importance_regressor(x_train, y_train, feature_labels, threshold)[
        "x_selected"
    ]


def main() -> None:
    print("RandomForestImportance migrated to mlproj.features.legacy_engine")


if __name__ == "__main__":
    main()
