# convert_contexts_to_ds.py

from datasets import load_dataset
from pathlib import Path

def main():
    # 1) Find your project root (one level up if this script lives in Meditron/)
    project_root = Path(__file__).parents[1]

    # 2) Point to the JSONL in your Dataset folder
    contexts_jsonl = project_root / "Dataset" / "processed" / "contexts.jsonl"
    if not contexts_jsonl.exists():
        raise FileNotFoundError(f"Cannot find {contexts_jsonl}")

    # 3) Load JSONL as a Dataset
    ds = load_dataset(
        "json",
        data_files=str(contexts_jsonl),
        split="train"
    )

    # 4) Save it back out in HF format
    out_dir = project_root / "Dataset" / "processed" / "contexts_dataset"
    ds.save_to_disk(str(out_dir))
    print(f"✅ Saved HuggingFace dataset to {out_dir}")

if __name__ == "__main__":
    main()
