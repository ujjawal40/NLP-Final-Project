# retriever.py

import torch
from pathlib import Path
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer

class PubMedRetriever:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        project_root = Path(__file__).parents[1]
        processed = project_root / "Dataset" / "processed"

        dataset_dir = processed / "contexts_dataset"
        index_path = processed / "vector_db" / "index.faiss"

        self.dataset = load_from_disk(str(dataset_dir))

        # **IMPORTANT: Load FAISS index manually**
        self.dataset.load_faiss_index("embeddings", str(index_path))

        self.embed_model = SentenceTransformer("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract", device=self.device)

    def retrieve(self, query, top_k=5):
        encoded_input = self.embed_model.encode(
            [query],  # <-- Notice the []
            convert_to_tensor=True,
            device=self.device,
            normalize_embeddings=True
        ).cpu().numpy()

        scores, samples = self.dataset.get_nearest_examples_batch(
            "embeddings",
            queries=encoded_input,
            k=top_k,
        )

        return samples
