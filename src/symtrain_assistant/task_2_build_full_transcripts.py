from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRANS_PATH = ROOT / "data" / "processed" / "transcripts.csv"
OUT_PATH = ROOT / "data" / "processed" / "full_transcripts.csv"

df = pd.read_csv(TRANS_PATH)

def build_full_transcript(group: pd.DataFrame) -> str:
    group = group.sort_values("sequence")
    lines = [f"{actor}: {text}" for actor, text in zip(group["actor"], group["text"])]
    return "\n".join(lines)

full_df = (
    df.groupby("simulation_id")
      .apply(build_full_transcript)
      .reset_index(name="full_transcript")
)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
full_df.to_csv(OUT_PATH, index=False, encoding="utf-8")

print(f"Saved {len(full_df)} simulations to {OUT_PATH}")
