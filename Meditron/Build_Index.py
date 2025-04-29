# Build_Index.py

import torch
from pathlib import Path
from datasets import load_from_disk
import faiss
import numpy as np
import logging
from tqdm import tqdm
import shutil
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # Load paths from yaml
    with open('path.yaml', 'r') as f:
        paths = yaml.safe_load(f)

    # Set up paths
    project_root = Path(__file__).parent.parent
    dataset_dir = project_root / "Dataset" / "processed" / "contexts_dataset"
    index_dir = project_root / "Dataset" / "processed" / "vector_db"
    index_path = index_dir / "index.faiss"

    logger.info(f"Project root: {project_root}")
    logger.info(f"Dataset directory: {dataset_dir}")
    logger.info(f"Index directory: {index_dir}")

    # Remove old index if it exists
    if index_dir.exists():
        logger.info("Removing old index directory...")
        shutil.rmtree(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    logger.info(f"Loading dataset from {dataset_dir}")
    try:
        dataset = load_from_disk(str(dataset_dir))
        logger.info(f"Dataset loaded successfully with {len(dataset)} examples")
    except Exception as e:
        logger.error(f"Failed to load dataset: {str(e)}")
        return

    # Get embeddings
    logger.info("Extracting embeddings...")
    embeddings = []
    valid_indices = []

    for i in tqdm(range(len(dataset)), desc="Processing embeddings"):
        try:
            emb = dataset[i]["embeddings"]
            if emb is not None and len(emb) > 0:
                embeddings.append(emb)
                valid_indices.append(i)
        except Exception as e:
            logger.error(f"Error processing embedding {i}: {str(e)}")
            continue

    if not embeddings:
        logger.error("No valid embeddings found!")
        return

    # Convert to numpy array
    embeddings = np.array(embeddings, dtype=np.float32)
    logger.info(f"Embeddings shape: {embeddings.shape}")

    # Build FAISS index
    logger.info("Building FAISS index...")
    dimension = embeddings.shape[1]

    # Create index
    index = faiss.IndexFlatL2(dimension)

    # Add vectors to index
    index.add(embeddings)

    # Save index
    logger.info(f"Saving index to {index_path}")
    faiss.write_index(index, str(index_path))

    # Verify index
    logger.info("Verifying index...")
    try:
        test_index = faiss.read_index(str(index_path))
        if test_index.ntotal == len(embeddings):
            logger.info("✅ Index verified successfully")
            logger.info(f"Index size: {test_index.ntotal}")
            logger.info(f"Dataset size: {len(dataset)}")
            logger.info(f"Valid embeddings: {len(valid_indices)}")
        else:
            logger.error("❌ Index verification failed: size mismatch")
    except Exception as e:
        logger.error(f"❌ Error verifying index: {str(e)}")


if __name__ == "__main__":
    main()
