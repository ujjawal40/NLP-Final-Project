import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from langchain_community.vectorstores import FAISS
from pathlib import Path
from tqdm import tqdm


class PubMedRetriever:
    def __init__(self):
        # 1. Hardcoded paths matching your EXACT files
        self.index_dir = Path("/home/ubuntu/NLP-Final-Project/Dataset/processed/vector_db")
        self.index_name = "index"  # Matches your index.faiss/index.pkl

        # 2. Initialize model and tokenizer directly
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        self.model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2").to(self.device)
        self.model.eval()

        # 3. Load index immediately
        self.vector_db = self._load_faiss_index()

    def _load_faiss_index(self):
        """Load existing FAISS index with proper embedding handling"""
        return FAISS.load_local(
            folder_path=str(self.index_dir),
            index_name=self.index_name,
            embeddings=self._embed_query,  # Only needs the query embedding function
            allow_dangerous_deserialization=True
        )

    def _embed_query(self, text):
        """Embed single query (required by FAISS)"""
        with torch.no_grad():
            inputs = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0].float().cpu().numpy()[0]  # Return numpy array

    def retrieve(self, query, k=3):
        """Simplified retrieval"""
        return self.vector_db.similarity_search(query, k=k)