# add_embeddings_to_ds.py

import torch
import shutil
from pathlib import Path
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer

def main():
    # 1) Locate the existing HF dataset
    project_root = Path(__file__).parents[1]
    ds_dir       = project_root / "Dataset" / "processed" / "contexts_dataset"
    assert ds_dir.exists(), f"{ds_dir} not found"

    print(f"Loading dataset from {ds_dir} …")
    ds = load_from_disk(str(ds_dir))

    # 2) Initialize embedder
    device   = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=device)

    # 3) Add 'title' column
    print("Adding 'title' column …")
    ds = ds.map(lambda ex: {"title": ex["text"]}, batched=False)

    # 4) Compute embeddings in batches
    print("Computing 'embeddings' column …")
    def embed_batch(batch):
        embs = embedder.encode(batch["text"], show_progress_bar=False)
        return {"embeddings": [emb.tolist() for emb in embs]}

    ds = ds.map(embed_batch, batched=True, batch_size=64)

    # 5) Write to a temp directory
    tmp_dir = ds_dir.parent / "contexts_dataset_tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    print(f"Saving augmented dataset to temporary folder {tmp_dir} …")
    ds.save_to_disk(str(tmp_dir))

    # 6) Swap folders
    print("Replacing original dataset with augmented version …")
    shutil.rmtree(ds_dir)
    tmp_dir.rename(ds_dir)

    print("✅ Augmented dataset now saved at", ds_dir)

if __name__ == "__main__":
    main()
