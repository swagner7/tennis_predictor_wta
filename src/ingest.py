import pandas as pd
import glob
import os

RAW_DIR = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

def ingest_excels():
    file_paths = glob.glob(os.path.join(RAW_DIR, "*.xlsx"))
    dfs = []
    for f in file_paths:
        print(f"Loading {f}")
        df = pd.read_excel(f, engine="openpyxl")
        dfs.append(df)
    all_df = pd.concat(dfs, ignore_index=True)
    # Ensure Date is datetime
    all_df['Date'] = pd.to_datetime(all_df['Date'])
    out_path = os.path.join(PROCESSED_DIR, "all_matches.csv")
    all_df.to_csv(out_path, index=False)
    print(f"Saved merged data to {out_path}")

if __name__ == "__main__":
    ingest_excels()
