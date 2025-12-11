import os
import json
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from transformers import pipeline
from openai import OpenAI

# =============================
# 0. Paths & config
# =============================

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"

INPUT_FILE = DATA_DIR / "task3_merged_gpt.csv"
OUTPUT_FILE = DATA_DIR / "task4_categorized.csv"

print(f"[INFO] Reading merged GPT file: {INPUT_FILE}")

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Input file not found: {INPUT_FILE}\n"
                            f"Please run task_3.3_finalize.py first.")

df = pd.read_csv(INPUT_FILE)

if "reason_gpt" in df.columns and df["reason_gpt"].notna().any():
    TEXT_COL = "reason_gpt"
elif "trainee_text" in df.columns and df["trainee_text"].notna().any():
    TEXT_COL = "trainee_text"
else:
    TEXT_COL = "full_transcript"

KEY_COL = "simulation_id"

print("[INFO] Columns:", df.columns.tolist())
print(f"[INFO] Using text column for categorization: {TEXT_COL}")

# =============================
# 1. Category space
# =============================

CATEGORIES = [
    "update_payment_method",
    "file_insurance_claim",
    "track_order_status",
    "other",
]

# =============================
# 2. Simple rule-based baseline
# =============================

def rule_based_category(text: str) -> str:
    """
    Very simple keyword-based classifier.
    Works on English 'reason' text or full transcript.
    """
    if not isinstance(text, str):
        return "other"

    t = text.lower()

    # update payment / card
    if "payment" in t or "credit card" in t or "debit card" in t or "card" in t:
        return "update_payment_method"

    # insurance claim
    if "claim" in t or "insurance" in t or "accident" in t:
        return "file_insurance_claim"

    # order / shipping / delivery status
    if "order" in t or "shipping" in t or "shipped" in t or "status" in t or "package" in t:
        return "track_order_status"

    return "other"


print("[INFO] Running rule-based categorization...")
df["category_rule"] = df[TEXT_COL].apply(rule_based_category)
print(df[["simulation_id", TEXT_COL, "category_rule"]].head(3))

# =============================
# 3. GPT / DeepSeek categorization
# =============================

print("[INFO] Setting up DeepSeek client for GPT categorization...")

load_dotenv()
API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not API_KEY:
    raise ValueError("DEEPSEEK_API_KEY not found in environment (.env)")

client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.deepseek.com",
)

CATEGORY_DESC = {
    "update_payment_method": "Customer wants to update or change a payment method or card information.",
    "file_insurance_claim": "Customer wants to file or manage an insurance claim about an accident or incident.",
    "track_order_status": "Customer wants to check or update the status of an order or delivery.",
    "other": "Any other type of request not covered above.",
}

def build_gpt_category_prompt(text: str) -> str:
    """
    Build a short prompt for classifying the call into one of the categories.
    """
    category_descriptions = "\n".join(
        [f"- {name}: {desc}" for name, desc in CATEGORY_DESC.items()]
    )

    return f"""
You are a classification assistant.

You must classify the following customer service conversation or reason
into EXACTLY ONE of these categories:

{category_descriptions}

Return ONLY a JSON object with a single field "category",
where the value is one of:
{CATEGORIES}

Example:
{{
  "category": "update_payment_method"
}}

Now classify this text:

{text}
""".strip()


def classify_with_gpt(text: str) -> str:
    """
    Call DeepSeek (GPT-style) to get a JSON object with "category".
    """
    if not isinstance(text, str) or not text.strip():
        return "other"

    prompt = build_gpt_category_prompt(text)

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        data = json.loads(content)
        cat = data.get("category", "other")
        if cat not in CATEGORIES:
            return "other"
        return cat
    except Exception as e:
        print(f"[WARN] GPT classification failed: {e}")
        return "other"


print("[INFO] Running GPT/DeepSeek categorization (this may take a while)...")

categories_gpt = []
for idx, row in df.iterrows():
    text = row[TEXT_COL]
    cat = classify_with_gpt(text)
    categories_gpt.append(cat)
    if (idx + 1) % 10 == 0:
        print(f"  [GPT] Processed {idx + 1} / {len(df)} rows")

df["category_gpt"] = categories_gpt

# =============================
# 4. Transformer (HF) categorization
# =============================

print("[INFO] Initializing HF zero-shot-classification pipeline...")

hf_classifier = pipeline(
    task="zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=-1,  # CPU
)

def classify_with_hf(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return "other"

    try:
        result = hf_classifier(
            text,
            candidate_labels=CATEGORIES,
            hypothesis_template="This call is about {}.",
        )
        #   {'sequence': ..., 'labels': [...], 'scores': [...]}
        labels = result["labels"]
        scores = result["scores"]
        if not labels:
            return "other"
        return labels[0]
    except Exception as e:
        print(f"[WARN] HF classification failed: {e}")
        return "other"


print("[INFO] Running HF transformer categorization...")

categories_tf = []
for idx, row in df.iterrows():
    text = row[TEXT_COL]
    cat = classify_with_hf(text)
    categories_tf.append(cat)
    if (idx + 1) % 10 == 0:
        print(f"  [HF] Processed {idx + 1} / {len(df)} rows")

df["category_tf"] = categories_tf

# =============================
# 5. Save results
# =============================

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

print(f"[DONE] Saved categorized simulations to: {OUTPUT_FILE}")
print(df[[KEY_COL, "category_rule", "category_gpt", "category_tf"]].head(5))