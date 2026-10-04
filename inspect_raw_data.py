from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")

for file in sorted(RAW_DIR.glob("*.csv")):
    df = pd.read_csv(file)

    print("=" * 80)
    print(f"FILE: {file.name}")
    print(f"ROWS: {df.shape[0]:,}")
    print(f"COLUMNS: {df.shape[1]}")

    print("\nCOLUMN NAMES:")
    print(list(df.columns))

    print("\nDATA TYPES:")
    print(df.dtypes)

    print("\nMISSING VALUES:")
    print(df.isna().sum())

    print("\nDUPLICATES:")
    print(df.duplicated().sum())

    print("\nSAMPLE:")
    print(df.head(3).to_string(index=False))

    print()
