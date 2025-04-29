# convert_contexts_to_ds.py

from datasets import Dataset
from pathlib import Path
import json
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # 1) Find your project root
    project_root = Path(__file__).parents[1]

    # 2) Point to the JSONL in your Dataset folder
    contexts_jsonl = project_root / "Dataset" / "processed" / "contexts.jsonl"
    if not contexts_jsonl.exists():
        raise FileNotFoundError(f"Cannot find {contexts_jsonl}")

    # 3) Load and process JSONL
    logger.info(f"Loading contexts from {contexts_jsonl}")
    contexts = []
    with open(contexts_jsonl, 'r') as f:
        for line in tqdm(f, desc="Loading contexts"):
            try:
                context = json.loads(line.strip())
                if 'text' in context and context['text'].strip():
                    contexts.append({
                        'text': context['text'].strip(),
                        'source': context.get('source', 'unknown')
                    })
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing line: {str(e)}")
                continue

    # 4) Create dataset
    logger.info(f"Creating dataset with {len(contexts)} contexts")
    ds = Dataset.from_list(contexts)

    # 5) Save it in HF format
    out_dir = project_root / "Dataset" / "processed" / "contexts_dataset"
    logger.info(f"Saving dataset to {out_dir}")
    ds.save_to_disk(str(out_dir))

    # 6) Verify the dataset
    logger.info("Verifying dataset...")
    try:
        test_ds = Dataset.load_from_disk(str(out_dir))
        if len(test_ds) == len(contexts):
            logger.info("✅ Dataset verified successfully")
        else:
            logger.error("❌ Dataset verification failed: size mismatch")
    except Exception as e:
        logger.error(f"❌ Error verifying dataset: {str(e)}")

if __name__ == "__main__":
    main()
