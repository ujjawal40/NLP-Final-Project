import os
from pathlib import Path
import yaml

# Base paths
PROJECT_ROOT = Path(__file__).parent

# Remote settings (EC2)
REMOTE_HOST =  "44.197.245.147"
REMOTE_USER = "ubuntu"  # Replace with your EC2 username
REMOTE_PORT = 22
REMOTE_PROJECT_PATH = "/home/ubuntu/NLP-Final-Project/Meditron"

# Local settings
LOCAL_PROJECT_PATH = PROJECT_ROOT
LOCAL_PORT = 8888  # Changed from 8501 to 8888

# Model paths
DATASET_PATH = os.path.join(PROJECT_ROOT, "Dataset", "processed", "contexts_dataset")
INDEX_PATH = os.path.join(PROJECT_ROOT, "Dataset", "processed", "vector_db", "index.faiss")

# Model settings
MODEL_NAME = "epfl-llm/meditron-7b"
DEVICE = "cuda" if os.environ.get("USE_GPU", "true").lower() == "true" else "cpu"

# Streamlit settings
STREAMLIT_SERVER_PORT = 8888  # Changed from 8501 to 8888
STREAMLIT_SERVER_ADDRESS = "localhost"
STREAMLIT_SERVER_HEADLESS = True
STREAMLIT_BROWSER_GATHER_USAGE_STATS = False

# SSH Configuration
SSH_HOST = "44.197.245.147"  # Your EC2 instance IP
SSH_USER = "ubuntu"
SSH_KEY_PATH = "Ujjawal_AWS_NLP.pem"  # Updated to use the correct key file
SSH_PORT = 22

class Config:
    def __init__(self):
        self.project_root = PROJECT_ROOT  # Absolute path

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
                "meditron": MODEL_NAME
            }
        }

        # Create directories if they don't exist
        os.makedirs(self.project_root / "Dataset/processed", exist_ok=True)

config = Config()