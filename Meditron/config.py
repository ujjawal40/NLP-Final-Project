import os
from pathlib import Path
import yaml

class Config:
    def __init__(self):
        self.project_root = Path("/home/ubuntu/NLP-Final-Project")  # Absolute path

        self.paths = {
            "data": {
                "raw": {
                    "artificial": str(self.project_root / "Dataset/pubmedqa_artificial.parquet"),
                    "labeled": str(self.project_root / "Dataset/pubmedqa_labeled.parquet")
                },
                "processed": str(self.project_root / "Dataset/processed"),
                # Remove .faiss extension here since the actual files are index.faiss/index.pkl
                "vector_db": str(self.project_root / "Dataset/processed/vector_db")
            },
            "models": {
                # Keep this if you're using both models
                "embedding": "sentence-transformers/all-mpnet-base-v2",
                "meditron": "epfl-llm/meditron-7b"
            }
        }

        # Create directories if they don't exist
        os.makedirs(self.project_root / "Dataset/processed", exist_ok=True)

config = Config()