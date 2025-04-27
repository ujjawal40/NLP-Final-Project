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
                "vector_db": str(self.project_root / "Dataset/processed/vector_db.faiss")
            },
            "models": {
                "embedding": "sentence-transformers/all-mpnet-base-v2"
            }
        }


config = Config()