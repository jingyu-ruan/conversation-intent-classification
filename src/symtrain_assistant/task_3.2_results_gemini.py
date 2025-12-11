import os
import json
import time
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from pathlib import Path
from openai import OpenAI

# ==========================================
# 1. Configuration and Paths
# ==========================================

# Project root directory: two levels above the current file
ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "data" / "processed" / "full_transcripts_with_trainee.csv"
OUTPUT_FILE = ROOT / "data" / "processed" / "task3_results_deepseek.csv"

# Ensure output directory exists
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# Load .env file
load_dotenv()

# Load DeepSeek API key
API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not API_KEY:
    raise ValueError("Error: DEEPSEEK_API_KEY not found in .env")

# Initialize DeepSeek (OpenAI-compatible)
client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.deepseek.com",
)

# ==========================================
# 2. Function to call DeepSeek (JSON output enforced)
# ==========================================

def extract_info_with_deepseek(transcript: str) -> str:
    """
    Call DeepSeek to extract 'reason' and 'steps'.

    Returns:
        A JSON-formatted string (strict JSON), which can be parsed by json.loads.
    """

    system_prompt = """
You are a data analysis assistant responsible for analyzing English call center transcripts.

Output the results in JSON format with the following fields:
{
  "reason": "A one-sentence summary of the customer's primary reason for calling, in English",
  "steps": ["Step 1", "Step 2"]
}

Requirements:
- 'reason' must be a concise English sentence.
- 'steps' must be a list of short English strings, each describing a step to resolve the issue.
- The response must be **strict JSON**, with no extra text outside the JSON object.
    """.strip()

    user_prompt = f"""
Below is a call center transcript. Analyze it and output JSON.

Transcript:
{transcript}
    """.strip()

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            # Enforce JSON output
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content
        return content

    except Exception as e:
        print(f"[Error] DeepSeek call failed: {e}")
        # Return fallback JSON to ensure parseability
        return json.dumps({"reason": "Extraction Failed", "steps": []})

# ==========================================
# 3. Main Logic
# ==========================================

if __name__ == "__main__":
    # 3.1 Load data
    if not INPUT_FILE.exists():
        print(f"Error: Input file {INPUT_FILE} not found.")
        print(f"Detected root path: {ROOT}")
        exit(1)

    df = pd.read_csv(INPUT_FILE)
    print(f"Successfully loaded data from: {INPUT_FILE}")
    print(f"Total records: {len(df)}")
    print("Starting extraction using DeepSeek (deepseek-chat)...")

    results_raw = []

    # 3.2 Call DeepSeek row by row
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Extracting"):
        transcript = row.get("full_transcript", "")

        # Skip API call for empty transcripts
        if not isinstance(transcript, str) or not transcript.strip():
            results_raw.append(json.dumps({"reason": "Empty Transcript", "steps": []}))
            continue

        json_str = extract_info_with_deepseek(transcript)
        results_raw.append(json_str)

        # Delay to avoid hitting rate limits
        time.sleep(1.5)

    # 3.3 Save raw JSON strings
    df["deepseek_raw_json"] = results_raw

    # 3.4 Parse JSON column into reason / steps
    print("\nParsing JSON results and formatting the output table...")

    def parse_json_column(json_str):
        try:
            data = json.loads(json_str)
            reason = data.get("reason", "Unknown")
            steps = data.get("steps", [])
            # Ensure steps is always a list
            if not isinstance(steps, list):
                steps = [str(steps)]
            return pd.Series([reason, steps])
        except Exception:
            return pd.Series(["Parse Error", []])

    df[["reason", "steps"]] = df["deepseek_raw_json"].apply(parse_json_column)

    # 3.5 Select columns to save
    cols_to_keep = ["simulation_id", "full_transcript", "reason", "steps"]
    final_columns = [c for c in cols_to_keep if c in df.columns]
    final_df = df[final_columns]

    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Task completed successfully. Results saved to: {OUTPUT_FILE}")

    print("\n=== Result Preview ===")
    if "reason" in final_df.columns and "steps" in final_df.columns:
        print(final_df[["reason", "steps"]].head(2))
