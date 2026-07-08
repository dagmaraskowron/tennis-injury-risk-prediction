import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "nanogrip_tennis_biomechanics.csv"
CLEAN_DATA_PATH = PROJECT_ROOT / "data" / "clean_tennis_biomechanics.csv"


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df.drop_duplicates()

    columns_to_drop = ["Subject_ID", "Trial_ID"]
    df = df.drop(columns=columns_to_drop)

    return df


if __name__ == "__main__":
    df = pd.read_csv(RAW_DATA_PATH)

    print("Initial dataset shape:")
    print(df.shape)

    print("\nMissing values before cleaning:")
    print(df.isnull().sum())

    print("\nDuplicated rows:")
    print(df.duplicated().sum())

    clean_df = clean_data(df)

    print("\nClean dataset shape:")
    print(clean_df.shape)

    print("\nMissing values after cleaning:")
    print(clean_df.isnull().sum())

    print("\nColumns after cleaning:")
    print(clean_df.columns.tolist())

    clean_df.to_csv(CLEAN_DATA_PATH, index=False)

    print(f"\nClean dataset saved to: {CLEAN_DATA_PATH}")