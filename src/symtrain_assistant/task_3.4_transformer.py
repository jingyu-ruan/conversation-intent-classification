import re
import json
from pathlib import Path

import pandas as pd
from transformers import pipeline

# =============================
# 0. paths
# =============================

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"

INPUT_FILE = DATA_DIR / "task3_merged_gpt.csv"  # the file where previously merged DeepSeek results
OUTPUT_FILE = DATA_DIR / "task3_with_hf_baseline.csv"

print(f"Reading: {INPUT_FILE}")
df_all = pd.read_csv(INPUT_FILE)

KEY_COL = "simulation_id"
TEXT_COL = "trainee_text" if "trainee_text" in df_all.columns else "full_transcript"

print("Columns:", df_all.columns.tolist())
print("Using text column:", TEXT_COL)

# =============================
# 1. Initialize Hugging Face pipeline
# =============================

hf_pipe = pipeline(
    task="text2text-generation",
    model="google/flan-t5-base",
    device=-1  # force CPU
)

# =============================
# 2. Build prompt and truncation function
# =============================

def build_task3_prompt_hf(text: str) -> str:
    return f"""
You are analyzing a customer service training conversation.

Task:
1. Summarize the main reason for the call in ONE sentence.
2. Then list 3 to 7 clear numbered steps the agent should follow to help the customer.

Output format (use English):

Reason: <one sentence reason>

Steps:
1. <step one>
2. <step two>
3. <step three>
...

Now analyze the following conversation and fill in the template.

Conversation:
{text}
""".strip()


def shorten_conversation(text: str, max_chars: int = 2500) -> str:
    """
    A simple truncation to fit max_chars.
    """
    if not isinstance(text, str):
        return ""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars]

# =============================
# 3. Parse model output
# =============================

def parse_reason_steps_from_output(output_text: str):
    """
    Try to extract a 'reason' and several 'steps' from the T5 output.
    """
    if not isinstance(output_text, str):
        return "", []

    lines = [l.strip() for l in output_text.splitlines() if l.strip()]
    if not lines:
        return "", []

    reason = ""
    steps = []

    for line in lines:
        lower = line.lower()
        if lower.startswith("reason:"):
            reason = line.split(":", 1)[1].strip()
            break

    # Collect steps
    for line in lines:
        stripped = line.strip()
        lower = stripped.lower()

        if lower.startswith("steps:"):
            continue

        # Step 1: xxx
        if lower.startswith("step"):
            step_text = re.sub(r"^step\s*\w*[:\.\)]?\s*", "", stripped, flags=re.IGNORECASE)
            step_text = step_text.strip()
            if step_text:
                steps.append(step_text)
            continue

        # 1. xxx / 2) xxx / 3: xxx
        if re.match(r"^\d+[\.\):]\s*", stripped):
            step_text = re.sub(r"^\d+[\.\):]\s*", "", stripped).strip()
            if step_text:
                steps.append(step_text)
            continue

    # If no Reason, find the first non-step line as Reason
    if not reason:
        for line in lines:
            stripped = line.strip()
            lower = stripped.lower()
            if lower.startswith("step") or re.match(r"^\d+[\.\):]\s*", stripped):
                continue
            # Remove prefixes like "TRAINEE:" "SYM:" etc.
            cleaned = re.sub(r"^\w+:\s*", "", stripped)
            reason = cleaned.strip()
            if reason:
                break

    return reason, steps

# =============================
# 4. Function to call HF model and get reason/steps
# =============================

def call_hf_reason_steps(text: str, max_new_tokens: int = 256, debug: bool = False):
    if not isinstance(text, str) or text.strip() == "":
        return "", []

    short_text = shorten_conversation(text, max_chars=2500)
    prompt = build_task3_prompt_hf(short_text)

    result = hf_pipe(
        prompt,
        max_new_tokens=max_new_tokens,
        num_beams=4,
        do_sample=False,
        truncation=True,
    )

    generated = result[0]["generated_text"]

    if debug:
        print("===== RAW MODEL OUTPUT (first 400 chars) =====")
        print(generated[:400])
        print("=============================================")

    reason, steps = parse_reason_steps_from_output(generated)
    return reason, steps

# =============================
# 5. Process all rows and save
# =============================

reasons_tf_all = []
steps_tf_all = []

n = len(df_all)
for idx, row in df_all.iterrows():
    text = row[TEXT_COL]
    reason_tf, steps_tf = call_hf_reason_steps(text, debug=False)
    reasons_tf_all.append(reason_tf)
    steps_tf_all.append(json.dumps(steps_tf, ensure_ascii=False))
    print(f"[{idx + 1}/{n}] simulation_id={row[KEY_COL]} | reason_len={len(reason_tf)} | steps_n={len(steps_tf)}")

df_all["reason_tf"] = reasons_tf_all
df_all["steps_tf"] = steps_tf_all

df_all.to_csv(OUTPUT_FILE, index=False)
print(f"Saved merged GPT + HF baseline file to: {OUTPUT_FILE}")
