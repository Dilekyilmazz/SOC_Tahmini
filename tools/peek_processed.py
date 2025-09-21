import glob, pathlib
import pandas as pd

paths = sorted(glob.glob("data/processed/*.parquet"))
print("[found]", len(paths), "parquet files\n")

for p in paths:
    df = pd.read_parquet(p)
    print("=== File:", p, "===")
    print("Columns:", list(df.columns))
    # İsteğe bağlı: ilk 3 satırı göster (kolonların doğru göründüğünü teyit için)
    print(df.head(3).to_string(index=False))
    print()
