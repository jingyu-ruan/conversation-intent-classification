from pathlib import Path
from typing import List, Dict

import json
import pandas as pd
from transformers import pipeline


# =============================
# 0. Paths and configuration
# =============================

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
OUTPUT_FILE = DATA_DIR / "task5_transformer_only.csv"

print("[INFO] Initializing transformer (flan-t5-base) pipeline...")
tf_pipe = pipeline(
    task="text2text-generation",
    model="google/flan-t5-base",
    device=-1,  # CPU; change to 0 if you have a GPU
)


# =============================
# 1. Define the 6 test inputs
# =============================

TEST_INPUTS = [
    {
        "test_id": "test_1",
        "text": (
            "Hi, I ordered a shirt last week and paid with my American Express card. "
            "I need to update the payment method because there is an issue with that card. "
            "Can you help me?"
        ),
    },
    {
        "test_id": "test_2",
        "text": (
            "Hi, I need to update the payment method for one of my recent orders. "
            "Can you help me with that?"
        ),
    },
    {
        "test_id": "test_3",
        "text": (
            "Hi, I am Sam. I was in a car accident this morning and need to file an "
            "insurance claim. Can you help me?"
        ),
    },
    {
        "test_id": "test_4",
        "text": "Hi, can you help me file a claim?",
    },
    {
        "test_id": "test_5",
        "text": (
            "Hi, I recently ordered a book online. "
            "Can you give me an update on the order status?"
        ),
    },
    {
        "test_id": "test_6",
        "text": (
            "Hi, I have been waiting for two weeks for the book I ordered. "
            "What is going on with it? Can you give me an update?"
        ),
    },
]
# If your assignment PDF uses slightly different wording,
# you can edit the 'text' field above accordingly.


# =============================
# 2. Build prompt and parse output
# =============================

def build_tf_prompt(user_input: str) -> str:
    """
    Build a concise instruction prompt for flan-t5:
    ask it to output a 'Reason:' line and a 'Steps:' list.
    """
    prompt = (
        "Summarize the user's reason for contacting customer support "
        "and provide clear step-by-step actions an agent should take.\n\n"
        f"User input: {user_input}\n\n"
        "Output format:\n"
        "Reason: <one-sentence reason>\n"
        "Steps:\n"
        "1. <step one>\n"
        "2. <step two>\n"
        "3. <step three> ..."
    )
    return prompt


def parse_tf_output(output_text: str) -> Dict:
    """
    Parse flan-t5 output into:
        reason_tf: str
        steps_tf: List[str]

    Expected pattern:
        Reason: ...
        Steps:
        1. ...
        2. ...
        ...

    If the format is imperfect, use best-effort parsing.
    """
    if not isinstance(output_text, str):
        return {"reason_tf": "", "steps_tf": []}

    lines = [line.strip() for line in output_text.splitlines() if line.strip()]
    reason = ""
    steps: List[str] = []

    for line in lines:
        lower = line.lower()
        if lower.startswith("reason:") and not reason:
            # Extract content after "Reason:"
            reason = line[len("reason:"):].strip()
        elif lower.startswith("steps:"):
            # Just a header line; ignore it
            continue
        else:
            # Try to detect a numbered step
            if any(line.startswith(prefix) for prefix in ["1.", "2.", "3.", "4.", "5.", "6.", "7."]):
                parts = line.split(".", 1)
                if len(parts) == 2:
                    steps.append(parts[1].strip())
                else:
                    steps.append(line)
            else:
                # If reason is still empty and this line does not look like a step,
                # treat it as a fallback reason.
                if not reason:
                    reason = line
                else:
                    # Otherwise, treat it as an additional step-like line.
                    steps.append(line)

    return {"reason_tf": reason, "steps_tf": steps}


def generate_transformer_output_for_input(user_input: str) -> Dict:
    """
    Use flan-t5-base to generate a baseline reason and steps
    for a new caller input.
    """
    prompt = build_tf_prompt(user_input)

    try:
        result = tf_pipe(
            prompt,
            max_new_tokens=256,
            num_return_sequences=1,
        )
        text = result[0]["generated_text"]
    except Exception as e:
        print(f"[WARN] Transformer generation failed for input: {user_input[:50]}..., error: {e}")
        return {
            "reason_tf": "",
            "steps_tf": [f"Transformer error: {str(e)}"],
        }

    parsed = parse_tf_output(text)
    return parsed


# =============================
# 3. Main flow
# =============================

def main():
    print("[INFO] Running transformer-only baseline for new callers...")

    records = []

    for item in TEST_INPUTS:
        test_id = item["test_id"]
        text = item["text"]

        print(f"\n[INFO] Processing {test_id}: {text[:60]}...")

        tf_result = generate_transformer_output_for_input(text)

        record = {
            "test_id": test_id,
            "user_input": text,
            "reason_tf": tf_result.get("reason_tf", ""),
            "steps_tf": json.dumps(tf_result.get("steps_tf", []), ensure_ascii=False),
        }
        records.append(record)

        print("  -> TF reason:", record["reason_tf"])
        print("  -> TF steps:")
        for i, s in enumerate(tf_result.get("steps_tf", []), 1):
            print(f"     {i}. {s}")

    df_out = pd.DataFrame(records)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print(f"\n[DONE] Saved transformer-only results for new callers to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()