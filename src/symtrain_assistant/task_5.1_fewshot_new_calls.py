import os
import json
import ast
from pathlib import Path
from typing import List, Dict

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


# =============================
# 0. Paths & config
# =============================

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"

INPUT_FILE = DATA_DIR / "task4_categorized.csv"
OUTPUT_FILE = DATA_DIR / "task5_new_calls_fewshot.csv"

print(f"[INFO] Reading categorized simulations from: {INPUT_FILE}")
df = pd.read_csv(INPUT_FILE)

# Category space (must be consistent with Task 4)
CATEGORIES = [
    "update_payment_method",
    "file_insurance_claim",
    "track_order_status",
    "other",
]

# Preferred text column for examples: use GPT summary if available
TEXT_COL_FOR_EXAMPLES = "reason_gpt" if "reason_gpt" in df.columns else "full_transcript"

# Initialize DeepSeek API (reusing DEEPSEEK_API_KEY from .env)
load_dotenv()
API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not API_KEY:
    raise ValueError("DEEPSEEK_API_KEY not found in .env")

client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.deepseek.com",
)


# =============================
# 1. Simple rule-based classifier
# =============================

def rule_based_category(text: str) -> str:
    """
    Very simple keyword-based classifier.
    This is sufficient for the 6 test inputs.
    """
    if not isinstance(text, str):
        return "other"

    t = text.lower()

    # Update payment / card
    if "payment" in t or "credit card" in t or "debit card" in t or "card" in t:
        return "update_payment_method"

    # Insurance claim
    if "claim" in t or "insurance" in t or "accident" in t:
        return "file_insurance_claim"

    # Order / shipping / delivery status
    if "order" in t or "shipping" in t or "shipped" in t or "status" in t or "package" in t:
        return "track_order_status"

    return "other"


def predict_category_for_text(text: str) -> str:
    """
    Predict a category for a new caller's input.
    For now we reuse the rule-based classifier for simplicity.
    You could replace this with the transformer/GPT classifier from Task 4.
    """
    return rule_based_category(text)


# =============================
# 2. Select few-shot examples from training data
# =============================

def parse_steps_column(s) -> List[str]:
    """
    Parse the steps_gpt column from CSV.
    It may be stored as a string representing a Python list,
    so we use ast.literal_eval when possible.
    """
    if isinstance(s, list):
        return s
    if not isinstance(s, str):
        return []
    s = s.strip()
    if not s:
        return []
    try:
        val = ast.literal_eval(s)
        if isinstance(val, list):
            return [str(x) for x in val]
        return [str(val)]
    except Exception:
        # If parsing fails, treat the whole string as a single step
        return [s]


def select_fewshot_examples(
    df: pd.DataFrame,
    target_category: str,
    k: int = 3
) -> List[Dict]:
    """
    Select k examples from the training data with the same category.
    If there are not enough examples in that category,
    fall back to arbitrary examples from the whole dataset.
    """
    # Prefer GPT-based category if available; otherwise fall back to rule-based
    if "category_gpt" in df.columns:
        df_cat = df[df["category_gpt"] == target_category]
    else:
        df_cat = df[df["category_rule"] == target_category]

    if len(df_cat) == 0:
        # If no examples in the target category, fall back to entire dataset
        df_cat = df

    # For simplicity, take the first k rows (you could also use sample(k))
    df_cat = df_cat.head(k)

    examples = []
    for _, row in df_cat.iterrows():
        reason = row.get("reason_gpt") or row.get(TEXT_COL_FOR_EXAMPLES) or ""
        steps_gpt = row.get("steps_gpt", "")
        steps_list = parse_steps_column(steps_gpt)
        examples.append(
            {
                "category": row.get("category_gpt")
                            or row.get("category_rule")
                            or target_category,
                "reason": reason,
                "steps": steps_list,
            }
        )
    return examples


# =============================
# 3. Build few-shot prompt and call DeepSeek
# =============================

def build_fewshot_prompt(
    user_input: str,
    target_category: str,
    examples: List[Dict],
) -> str:
    """
    Build the few-shot prompt to ask the model to return JSON:
      {"category": "...", "reason": "...", "steps": ["...", "..."]}.
    """

    example_blocks = []
    for ex in examples:
        steps_str = "\n".join([f"- {s}" for s in ex["steps"]])
        block = (
            f"Example\n"
            f"User reason: {ex['reason']}\n"
            f"Category: {ex['category']}\n"
            f"Steps:\n{steps_str}"
        )
        example_blocks.append(block)

    examples_text = "\n\n".join(example_blocks)

    categories_list = ", ".join([f'"{c}"' for c in CATEGORIES])

    prompt = f"""
You are a helpful customer support assistant.

We have several past examples of calls, each with:
- the user's reason for calling,
- the category of the call,
- and the steps that the agent should follow.

Use the examples to understand the style and structure of the output.

{examples_text}

Now we have a NEW caller:

User input: {user_input}

First, decide which category the new call belongs to. The category must be ONE of:
{categories_list}

Then:
1. Write a short, clear English summary of the user's reason (1–2 sentences).
2. Generate 3–7 clear, actionable steps an agent should follow to help this user.

Return ONLY a valid JSON object with EXACTLY these fields:
{{
  "category": "...",   // one of {categories_list}
  "reason": "...",     // short English summary
  "steps": ["...", "..."]   // ordered list of steps
}}
""".strip()

    return prompt


def generate_fewshot_output_for_input(
    user_input: str,
    df_examples: pd.DataFrame,
    k: int = 3,
) -> Dict:
    """
    Generate few-shot GPT output for a single user input.
    """
    # 1. Predict category (rough initial guess)
    rough_category = predict_category_for_text(user_input)

    # 2. Select few-shot examples
    examples = select_fewshot_examples(df_examples, rough_category, k=k)

    # 3. Build prompt
    prompt = build_fewshot_prompt(user_input, rough_category, examples)

    # 4. Call DeepSeek with JSON response format
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        data = json.loads(content)
    except Exception as e:
        print(f"[WARN] DeepSeek call failed for input: {user_input[:50]}..., error: {e}")
        # Fallback: at least return a structurally valid JSON
        data = {
            "category": rough_category,
            "reason": user_input,
            "steps": [f"Fallback error: {str(e)}"],
        }

    # Ensure all expected keys exist and have correct types
    if "category" not in data:
        data["category"] = rough_category
    if "reason" not in data:
        data["reason"] = user_input
    if "steps" not in data or not isinstance(data["steps"], list):
        data["steps"] = [str(data.get("steps", "No steps generated."))]

    return data


# =============================
# 4. Define the 6 test inputs
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
        "text": "Hi, I need to update the payment method for one of my recent orders. Can you help me with that?",
    },
    {
        "test_id": "test_3",
        "text": "Hi, I am Sam. I was in a car accident this morning and need to file an insurance claim. Can you help me?",
    },
    {
        "test_id": "test_4",
        "text": "Hi, can you help me file a claim?",
    },
    {
        "test_id": "test_5",
        "text": "Hi, I recently ordered a book online. Can you give me an update on the order status?",
    },
    {
        "test_id": "test_6",
        "text": (
            "Hi, I have been waiting for two weeks for the book I ordered. "
            "What is going on with it? Can you give me an update?"
        ),
    },
]
# If your assignment PDF has slightly different wording, edit the text values above.


# =============================
# 5. Main flow: run few-shot for all 6 inputs
# =============================

def main():
    print("[INFO] Running few-shot GPT for new callers...")

    # Prepare examples: keep only rows with non-null reason_gpt and steps_gpt, if available
    df_examples = df.copy()
    if "reason_gpt" in df_examples.columns:
        df_examples = df_examples[df_examples["reason_gpt"].notna()]
    if "steps_gpt" in df_examples.columns:
        df_examples = df_examples[df_examples["steps_gpt"].notna()]

    records = []

    for item in TEST_INPUTS:
        test_id = item["test_id"]
        text = item["text"]

        print(f"\n[INFO] Processing {test_id}: {text[:60]}...")

        result = generate_fewshot_output_for_input(text, df_examples, k=3)

        record = {
            "test_id": test_id,
            "user_input": text,
            "category_pred": result.get("category"),
            "reason_pred": result.get("reason"),
            "steps_pred": json.dumps(result.get("steps", []), ensure_ascii=False),
        }
        records.append(record)

        print("  -> category:", record["category_pred"])
        print("  -> reason:  ", record["reason_pred"])
        print("  -> steps:")
        for i, s in enumerate(result.get("steps", []), 1):
            print(f"     {i}. {s}")

    df_out = pd.DataFrame(records)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print(f"\n[DONE] Saved few-shot results for new callers to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()