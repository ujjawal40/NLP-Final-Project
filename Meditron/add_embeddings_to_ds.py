# add_embeddings_to_ds.py

import torch
import shutil
from pathlib import Path
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer
import logging
from tqdm import tqdm
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # 1) Locate the existing HF dataset
    project_root = Path(__file__).parents[1]
    ds_dir = project_root / "Dataset" / "processed" / "contexts_dataset"
    assert ds_dir.exists(), f"{ds_dir} not found"

    logger.info(f"Loading dataset from {ds_dir}")
    ds = load_from_disk(str(ds_dir))

    # 2) Initialize embedder with medical model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    logger.info("Loading medical model...")
    embedder = SentenceTransformer(
        "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        device=device
    )

    # 3) Add 'title' column
    logger.info("Adding 'title' column")
    ds = ds.map(lambda ex: {"title": ex["text"]}, batched=False)

    # 4) Compute embeddings in batches
    logger.info("Computing embeddings...")
    def embed_batch(batch):
        try:
            # Clean and prepare text
            texts = [str(text).strip() for text in batch["text"]]
            texts = [text for text in texts if text]  # Remove empty strings
            
            if not texts:
                return {"embeddings": [[] for _ in range(len(batch["text"]))]}
            
            # Compute embeddings
            embs = embedder.encode(
                texts,
                show_progress_bar=False,
                convert_to_tensor=True,
                normalize_embeddings=True
            )
            
            # Convert to list and handle any errors
            embeddings = []
            for i, emb in enumerate(embs):
                try:
                    embeddings.append(emb.cpu().numpy().tolist())
                except Exception as e:
                    logger.error(f"Error processing embedding {i}: {str(e)}")
                    embeddings.append([])
            
            return {"embeddings": embeddings}
        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}")
            return {"embeddings": [[] for _ in range(len(batch["text"]))]}

    # Process in smaller batches for better memory management
    batch_size = 32
    ds = ds.map(
        embed_batch,
        batched=True,
        batch_size=batch_size,
        desc="Computing embeddings"
    )

    # 5) Write to a temp directory
    tmp_dir = ds_dir.parent / "contexts_dataset_tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    logger.info(f"Saving augmented dataset to temporary folder {tmp_dir}")
    ds.save_to_disk(str(tmp_dir))

    # 6) Swap folders
    logger.info("Replacing original dataset with augmented version")
    shutil.rmtree(ds_dir)
    tmp_dir.rename(ds_dir)

    logger.info("✅ Augmented dataset now saved at %s", ds_dir)
    
    # 7) Verify the embeddings
    logger.info("Verifying embeddings...")
    try:
        test_ds = load_from_disk(str(ds_dir))
        sample_emb = test_ds[0]["embeddings"]
        if sample_emb and len(sample_emb) > 0:
            logger.info("✅ Embeddings verified successfully")
            # Print sample embedding stats
            logger.info(f"Embedding dimension: {len(sample_emb)}")
            logger.info(f"Sample embedding mean: {np.mean(sample_emb):.4f}")
            logger.info(f"Sample embedding std: {np.std(sample_emb):.4f}")
        else:
            logger.error("❌ Embeddings verification failed")
    except Exception as e:
        logger.error(f"❌ Error verifying embeddings: {str(e)}")

if __name__ == "__main__":
    main()
