from __future__ import annotations

try:
    from mlproj.features.legacy_engine import FeatureBuilding
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[1]
    SRC = ROOT / "src"
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from mlproj.features.legacy_engine import FeatureBuilding


def main() -> None:
    import numpy as np

    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    print(FeatureBuilding().gen_polynomial_features(X, degree=2).shape)


if __name__ == "__main__":
    main()
