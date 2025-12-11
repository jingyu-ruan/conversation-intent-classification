from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULT_FILE = ROOT / "data" / "processed" / "task5_new_calls_fewshot.csv"


def main():
    if not RESULT_FILE.exists():
        raise FileNotFoundError(f"Result file not found: {RESULT_FILE}")

    df = pd.read_csv(RESULT_FILE)

    print("Columns:", df.columns.tolist())
    print()

    for _, row in df.iterrows():
        print("=" * 60)
        print(f"Test ID   : {row['test_id']}")
        print(f"User input: {row['user_input']}")
        print(f"Category  : {row['category_pred']}")
        print(f"Reason    : {row['reason_pred']}")

        # steps_pred is stored as a JSON string; parse it back to a list
        steps_str = row.get("steps_pred", "[]")
        try:
            steps = json.loads(steps_str)
        except Exception:
            steps = [steps_str]

        print("Steps:")
        for i, s in enumerate(steps, 1):
            print(f"  {i}. {s}")

    print("\n[INFO] Done.")


if __name__ == "__main__":
    main()