from __future__ import annotations

try:
    from mlproj.features.legacy_engine import (
        NormalityTransform,
        box_cox_transform,
        log1p_transform,
        log_transform_feature,
        normalizer_L2,
        normalizer_Ln,
        normalizer_min_max,
        normalizer_min_max_feature,
        ploynomial_transform,
        quantileNorm,
        quantileUniform,
        robust_tansform,
        standard_center,
        yeo_johnson_transform,
    )
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[1]
    SRC = ROOT / "src"
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from mlproj.features.legacy_engine import (
        NormalityTransform,
        box_cox_transform,
        log1p_transform,
        log_transform_feature,
        normalizer_L2,
        normalizer_Ln,
        normalizer_min_max,
        normalizer_min_max_feature,
        ploynomial_transform,
        quantileNorm,
        quantileUniform,
        robust_tansform,
        standard_center,
        yeo_johnson_transform,
    )


__all__ = ['NormalityTransform', 'box_cox_transform', 'log1p_transform', 'log_transform_feature', 'normalizer_L2', 'normalizer_Ln', 'normalizer_min_max', 'normalizer_min_max_feature', 'ploynomial_transform', 'quantileNorm', 'quantileUniform', 'robust_tansform', 'standard_center', 'yeo_johnson_transform']

def main() -> None:
    print("FeatureTransform migrated to mlproj.features.legacy_engine")


if __name__ == "__main__":
    main()

