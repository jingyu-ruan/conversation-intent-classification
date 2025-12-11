import os
import json
from pathlib import Path

import pandas as pd

# =============================
# 0. paths
# =============================
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"

TRANSCRIPT_FILE = DATA_DIR / "full_transcripts.csv"
DEEPSEEK_FILE = DATA_DIR / "task3_results_deepseek.csv"  # Adjust the path as needed

# =============================
# 1. Load data
# =============================
df_trans = pd.read_csv(TRANSCRIPT_FILE)
df_ds = pd.read_csv(DEEPSEEK_FILE)

print("Transcripts columns:", df_trans.columns.tolist())
print("DeepSeek columns:", df_ds.columns.tolist())
print("Transcripts head:")
print(df_trans.head(3))
print("DeepSeek head:")
print(df_ds.head(3))

# =============================
# 1.1 Merge results
# =============================

KEY_COL = "simulation_id"  

# Rename columns to avoid conflicts
df_ds = df_ds.rename(columns={
    "reason": "reason_gpt",
    "steps": "steps_gpt"
})

keep_cols_ds = [KEY_COL, "reason_gpt", "steps_gpt"]
df_ds = df_ds[keep_cols_ds].drop_duplicates(subset=[KEY_COL])

df_all = df_trans.merge(df_ds, on=KEY_COL, how="left")

print("Merged shape:", df_all.shape)
print(df_all[[KEY_COL, "reason_gpt", "steps_gpt"]].head(5))

missing = df_all["reason_gpt"].isna().mean()
print(f"Missing GPT output ratio: {missing:.2%}")

OUT_MERGED = DATA_DIR / "task3_merged_gpt.csv"
df_all.to_csv(OUT_MERGED, index=False)
print(f"Saved merged results to: {OUT_MERGED}")