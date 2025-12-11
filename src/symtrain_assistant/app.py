#Instruction: 
# In the project root directory, execute:
# "streamlit run src/app.py"

import os
import json
import ast
from pathlib import Path
from typing import List, Dict

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI


# =============================
# 0. Paths and global config
# =============================

# Project root: parent of src/
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
EXAMPLES_FILE = DATA_DIR / "task4_categorized.csv"

CATEGORIES = [
    "update_payment_method",
    "file_insurance_claim",
    "track_order_status",
    "other",
]


# Load environment variables (for DEEPSEEK_API_KEY)
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

if DEEPSEEK_API_KEY:
    client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com",
    )
else:
    client = None


# =============================
# 1. Data loading helpers
# =============================

@st.cache_data
def load_examples() -> pd.DataFrame:
    """
    Load the categorized simulations (Task 4 output) as few-shot examples.
    We expect:
        - simulation_id
        - reason_gpt
        - steps_gpt
        - category_gpt / category_rule / category_tf
    """
    if not EXAMPLES_FILE.exists():
        raise FileNotFoundError(
            f"Examples file not found: {EXAMPLES_FILE}. "
            f"Please run Task 4 to generate task4_categorized.csv."
        )

    df = pd.read_csv(EXAMPLES_FILE)
    # Keep only rows where we have GPT reason and steps
    if "reason_gpt" in df.columns:
        df = df[df["reason_gpt"].notna()]
    if "steps_gpt" in df.columns:
        df = df[df["steps_gpt"].notna()]
    return df


# =============================
# 2. Rule-based category classifier
# =============================

def rule_based_category(text: str) -> str:
    """
    Very simple keyword-based classifier used for the first category prediction.
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


# =============================
# 3. Few-shot selection utilities
# =============================

def parse_steps_column(s) -> List[str]:
    """
    Parse a steps column that may be stored as a string of a Python list.
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
        return [s]


def select_fewshot_examples(
    df: pd.DataFrame,
    target_category: str,
    k: int = 3
) -> List[Dict]:
    """
    Select k examples from the training data with the same category.
    If not enough examples exist, fall back to arbitrary examples.
    """
    if "category_gpt" in df.columns:
        df_cat = df[df["category_gpt"] == target_category]
    else:
        df_cat = df[df["category_rule"] == target_category]

    if len(df_cat) == 0:
        df_cat = df

    df_cat = df_cat.head(k)

    examples: List[Dict] = []
    for _, row in df_cat.iterrows():
        reason = row.get("reason_gpt") or row.get("full_transcript") or ""
        steps_raw = row.get("steps_gpt", "")
        steps_list = parse_steps_column(steps_raw)
        examples.append(
            {
                "simulation_id": row.get("simulation_id"),
                "category": row.get("category_gpt")
                            or row.get("category_rule")
                            or row.get("category_tf")
                            or target_category,
                "reason": reason,
                "steps": steps_list,
            }
        )
    return examples


# =============================
# 4. Build prompt and call DeepSeek
# =============================

def build_fewshot_prompt(
    user_input: str,
    target_category: str,
    examples: List[Dict],
) -> str:
    """
    Build a few-shot prompt to ask the model to return JSON:
    {
      "category": "...",
      "reason": "...",
      "steps": ["...", "..."]
    }
    """

    example_blocks = []
    for ex in examples:
        steps_str = "\n".join([f"- {s}" for s in ex["steps"]])
        block = (
            f"Example (simulation_id={ex['simulation_id']})\n"
            f"User reason: {ex['reason']}\n"
            f"Category: {ex['category']}\n"
            f"Steps:\n{steps_str}"
        )
        example_blocks.append(block)

    examples_text = "\n\n".join(example_blocks)
    categories_list = ", ".join([f'"{c}"' for c in CATEGORIES])

    prompt = f"""
You are a helpful customer support assistant.

We have several past training calls, each with:
- the user's reason for calling,
- the category of the call,
- and the steps that the agent should follow.

Use the examples to understand the style and structure of the output.

{examples_text}

Now we have a NEW caller:

User input: {user_input}

First, decide which category this new call belongs to. The category must be ONE of:
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


def call_deepseek_with_json(prompt: str) -> Dict:
    """
    Call DeepSeek (OpenAI-compatible) and enforce JSON output.
    """
    if client is None:
        return {
            "category": "other",
            "reason": "API key is missing.",
            "steps": ["Please set DEEPSEEK_API_KEY in the .env file."],
        }

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
        data = {
            "category": "other",
            "reason": f"Model call failed: {e}",
            "steps": [],
        }

    # Ensure required keys exist
    if "category" not in data:
        data["category"] = "other"
    if "reason" not in data:
        data["reason"] = ""
    if "steps" not in data or not isinstance(data["steps"], list):
        data["steps"] = [str(data.get("steps", "No steps generated."))]

    return data


# =============================
# 5. Streamlit UI
# =============================

def main():
    st.set_page_config(
        page_title="SymTrain Simulation Assistant",
        page_icon="🤖",
        layout="wide",
    )

    st.title("SymTrain Simulation Assistant")
    st.markdown(
        "This app takes a customer request, predicts a category, "
        "and generates a reason and step-by-step guidance using few-shot GPT."
    )

    # Display warning if API key is missing
    if client is None:
        st.error(
            "DEEPSEEK_API_KEY is not set. "
            "Please create a .env file in the project root with:\n"
            "`DEEPSEEK_API_KEY=your_api_key_here`"
        )

    # Load examples (Task 4 output)
    try:
        df_examples = load_examples()
    except Exception as e:
        st.error(f"Failed to load examples from Task 4: {e}")
        st.stop()

    with st.sidebar:
        st.header("Settings")
        k_examples = st.slider(
            "Number of few-shot examples per category",
            min_value=1,
            max_value=5,
            value=3,
            step=1,
        )

    st.subheader("Customer Request")
    user_input = st.text_area(
        "Enter a customer request in natural language:",
        height=150,
        placeholder="Example: Hi, I need to update the payment method for my last order...",
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        run_button = st.button("Generate Reason and Steps", type="primary")

    if run_button:
        if not user_input.strip():
            st.warning("Please enter a customer request first.")
            st.stop()

        # 1. Rule-based category prediction
        rough_category = rule_based_category(user_input)

        # 2. Select few-shot examples from Task 4 data
        examples = select_fewshot_examples(df_examples, rough_category, k=k_examples)

        # 3. Build prompt and call DeepSeek
        prompt = build_fewshot_prompt(user_input, rough_category, examples)
        result = call_deepseek_with_json(prompt)

        gpt_category = result.get("category", "other")
        gpt_reason = result.get("reason", "")
        gpt_steps = result.get("steps", [])

        # ========== Display results ==========
        st.markdown("---")
        st.subheader("Predicted Category")

        st.markdown(
            f"- **Rule-based prediction:** `{rough_category}`  \n"
            f"- **GPT (DeepSeek) prediction:** `{gpt_category}`"
        )

        st.subheader("Generated Reason")
        st.write(gpt_reason)

        st.subheader("Generated Steps")
        for i, step in enumerate(gpt_steps, start=1):
            st.markdown(f"{i}. {step}")

        with st.expander("Few-shot examples used"):
            for ex in examples:
                st.markdown(
                    f"**simulation_id:** `{ex['simulation_id']}`  \n"
                    f"**category:** `{ex['category']}`  \n"
                    f"**reason:** {ex['reason']}"
                )
                st.markdown("**steps:**")
                for i, s in enumerate(ex["steps"], start=1):
                    st.markdown(f"- {i}. {s}")
                st.markdown("---")

        with st.expander("Raw JSON output from GPT"):
            st.code(json.dumps(result, indent=2, ensure_ascii=False), language="json")


if __name__ == "__main__":
    main()

