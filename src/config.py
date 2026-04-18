from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

DATASET_FILENAME = "AirQualityUCI.csv"
MISSING_VALUE_SENTINEL = -200

TARGET_COL = "CO(GT)"
NMHC_COL = "NMHC(GT)"  # 89% missing — excluded from regression features

FEATURE_COLS = [
    "CO(GT)",
    "PT08.S1(CO)",
    "NMHC(GT)",
    "C6H6(GT)",
    "PT08.S2(NMHC)",
    "NOx(GT)",
    "PT08.S3(NOx)",
    "NO2(GT)",
    "PT08.S4(NO2)",
    "PT08.S5(O3)",
    "T",
    "RH",
    "AH",
]

# Sensor columns usable as regressors (exclude high-missingness NMHC)
REGRESSOR_COLS = [c for c in FEATURE_COLS if c != NMHC_COL]
