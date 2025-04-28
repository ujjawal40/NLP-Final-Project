# Build_Index.py

import json
import time
import shutil
import torch
import nltk
import pandas as pd

from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings

from dataloader import PubMedQADataLoader

nltk.download("punkt", quiet=True)

# size of each chunk and overlap stride
CHUNK_SIZE = 512
STRIDE     = 256

def chunk_text(text: str, tokenizer) -> list[str]:
    """
    Split `text` into overlapping chunks of CHUNK_SIZE tokens,
    moving forward by STRIDE each time.
    """
    tokens = tokenizer.tokenize(text)
    chunks = []
    for start in range(0, len(tokens), STRIDE):
        slice_ = tokens[start : start + CHUNK_SIZE]
        chunk = tokenizer.convert_tokens_to_string(slice_)
        chunks.append(chunk)
        if start + CHUNK_SIZE >= len(tokens):
            break
    return chunks

def main():
    project_root = Path(__file__).parent.parent
    dataset_dir  = project_root / "Dataset"
    processed    = dataset_dir / "processed"
    contexts_f   = processed / "contexts.jsonl"
    vector_db    = processed / "vector_db"

    # 1) Remove old index folder
    if vector_db.exists():
        print("Removing old vector_db folder...")
        shutil.rmtree(vector_db)
    processed.mkdir(parents=True, exist_ok=True)
    vector_db.mkdir(parents=True, exist_ok=True)

    # 2) Load PubMedQA data
    print("1) Loading PubMedQA data…")
    loader = PubMedQADataLoader()
    df = loader.load_parquets()  # expects columns: question, context, pubid, final_decision
    print(f"   Loaded {len(df)} examples\n")

    # 3) Initialize embedding model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"2) Initializing SentenceTransformer on {device}…")
    embedder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=device)
    tokenizer = embedder.tokenizer
    print("   Done.\n")

    # 4) Chunk texts and write to contexts.jsonl
    print(f"3) Writing chunks to {contexts_f}…")
    idx = 0
    with contexts_f.open("w", encoding="utf-8") as fout:
        for _, row in tqdm(df.iterrows(), total=len(df)):
            question = str(row.get("question", ""))
            raw_ctx = row.get("context", [])
            if not isinstance(raw_ctx, list):
                raw_ctx = [raw_ctx]

            # coerce each context item to string
            contexts: list[str] = []
            for c in raw_ctx:
                if isinstance(c, str):
                    contexts.append(c)
                elif isinstance(c, dict):
                    # adjust key if needed
                    text = c.get("text") or c.get("page_content") or ""
                    contexts.append(str(text))
                else:
                    contexts.append(str(c))

            full_text = question + " " + " ".join(contexts)
            for chunk in chunk_text(full_text, tokenizer):
                record = {
                    "id": idx,
                    "text": chunk,
                    "meta": {
                        "pubid": row.get("pubid"),
                        "decision": row.get("final_decision")
                    }
                }
                fout.write(json.dumps(record) + "\n")
                idx += 1

    print(f"   Wrote {idx} chunks\n")

    # 5) Build & save FAISS index using LangChain from_texts
    print("4) Building FAISS index…")
    # load texts and metadata back
    texts, metadatas = [], []
    with contexts_f.open("r", encoding="utf-8") as fin:
        for line in fin:
            rec = json.loads(line)
            texts.append(rec["text"])
            metadatas.append(rec["meta"])

    # wrap embedder for LangChain
    lc_embed = SentenceTransformerEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": device}
    )

    # build index
    vectorstore = FAISS.from_texts(texts, lc_embed, metadatas=metadatas)

    # save to disk
    vectorstore.save_local(folder_path=str(vector_db), index_name="index")
    print(f"   FAISS index built and saved to {vector_db}")

if __name__ == "__main__":
    main()
