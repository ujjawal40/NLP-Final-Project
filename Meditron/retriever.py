import torch
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
from langchain_community.vectorstores import FAISS
from pathlib import Path
from tqdm import tqdm
import os


class PubMedRetriever:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vector_db = None
        self._init_models()
        self.index_path = Path(config["index_path"])
        os.makedirs(self.index_path.parent, exist_ok=True)

    def _init_models(self):
        """Initialize embedding models"""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["embedding_model"])
        self.model = AutoModel.from_pretrained(
            self.config["embedding_model"]).to(self.device)

    def embed(self, texts, batch_size=64):
        """Batch embedding generation"""
        self.model.eval()
        embeddings = []

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
            for i in tqdm(range(0, len(texts), batch_size),
                          desc="Generating embeddings",
                          unit="batch"):
                batch = texts[i:i + batch_size]
                inputs = self.tokenizer(
                    batch,
                    padding="max_length",
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(self.device)
                outputs = self.model(**inputs)
                embeddings.append(outputs.last_hidden_state[:, 0].cpu().numpy())

        return np.concatenate(embeddings).astype(np.float32)

    def build_index(self, documents):
        """Build and save FAISS index"""
        try:
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]

            # Generate embeddings
            print("Generating embeddings...")
            embeddings = self.embed(texts)

            # Create and save FAISS index
            print("Building FAISS index...")
            self.vector_db = FAISS.from_embeddings(
                text_embeddings=list(zip(texts, embeddings)),
                embedding=self.embed,
                metadatas=metadatas
            )
            self.vector_db.save_local(self.index_path)
            print(f"Index saved to {self.index_path}")

        except Exception as e:
            print(f"Error building index: {str(e)}")
            raise

    def retrieve(self, query, k=5):
        """Safe retrieval with auto-loading"""
        if not self.vector_db:
            self.vector_db = FAISS.load_local(
                self.index_path,
                self.embed,
                allow_dangerous_deserialization=True
            )
        return self.vector_db.similarity_search(query, k=k)