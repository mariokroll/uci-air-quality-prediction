"""Phase 1 demo script: load data, run BLR imputation, save output."""

import pandas as pd

from src.config import PROCESSED_DATA_DIR
from src.data.loader import load_raw
from src.data.preprocessor import missing_summary
from src.imputation.bayesian_linear import BayesianLinearImputer


def main() -> None:
    print("Loading raw data …")
    df = load_raw()
    print(f"  Shape: {df.shape}")
    print("\nMissing values before imputation:")
    print(missing_summary(df).to_string())

    print("\nFitting Bayesian Linear Regression imputer …")
    imputer = BayesianLinearImputer(alpha=1.0, noise_var=1.0)
    df_imputed = imputer.fit_transform(df)

    print("\nMissing values after imputation:")
    print(missing_summary(df_imputed).to_string())

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DATA_DIR / "blr_imputed.csv"
    df_imputed.to_csv(out_path)
    print(f"\nSaved imputed data → {out_path}")


if __name__ == "__main__":
    main()
