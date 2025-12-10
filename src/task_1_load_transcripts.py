from pathlib import Path
import zipfile
import json
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw_zips"
OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "transcripts.csv"

def main():
    records = []

    for zip_path in RAW_DIR.glob("*.zip"):
        sim_id = zip_path.stem  # Assuming the zip file name is the simulation ID

        print(f"Processing {zip_path.name} ...")

        with zipfile.ZipFile(zip_path, "r") as zf:
            json_files = [name for name in zf.namelist() if name.endswith(".json")]

            if not json_files:
                print(f"  [WARN] no json file in {zip_path.name}")
                continue

            for json_name in json_files:
                with zf.open(json_name) as f:
                    data = json.load(f)

                items = data.get("audioContentItems", [])
                if not items:
                    print(f"  [WARN] {json_name} has no audioContentItems")
                    continue

                for item in items:
                    records.append(
                        {
                            "simulation_id": sim_id,
                            "json_file": json_name,
                            "sequence": item.get("sequenceNumber"),
                            "actor": item.get("actor"),
                            "text": item.get("fileTranscript"),
                            "external_id": item.get("externalId"),
                            "audio_file_id": item.get("fileId"),
                        }
                    )

    if not records:
        print("No records found. Check RAW_DIR and json structure.")
        return

    df = pd.DataFrame(records)
    df = df.sort_values(["simulation_id", "sequence"]).reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False, encoding="utf-8")

    print(f"Saved {len(df)} rows to {OUT_PATH}")

if __name__ == "__main__":
    main()
