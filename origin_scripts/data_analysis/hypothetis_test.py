from __future__ import annotations

import pandas as pd

try:
    from mlproj.analysis.hypothesis import corr_test
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[1]
    SRC = ROOT / "src"
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from mlproj.analysis.hypothesis import corr_test


def main() -> None:
    data1 = [0.873, 2.817, 0.121, -0.945, -0.055, -1.436, 0.360, -1.478, -1.637, -1.869]
    data2 = [0.353, 3.517, 0.125, -7.545, -0.555, -1.536, 3.350, -1.578, -3.537, -1.579]
    df = pd.DataFrame({"dt1": data1, "dt2": data2})

    res_table = corr_test(df, xcols=["dt1"], ycols=["dt2"], methods=("pearson", "spearman"))
    print(res_table)


if __name__ == "__main__":
    main()
