import pandas as pd
from langchain.schema import Document
from pathlib import Path
import os


class PubMedQADataLoader:
    def __init__(self):
        # Absolute path to Dataset directory
        self.data_dir = Path.home() / "NLP-Final-Project" / "Dataset"

        # Set exact filenames with extensions
        self.artificial_path = self.data_dir / "pubmedqa_artificial.parquet"
        self.labeled_path = self.data_dir / "pubmedqa_labeled.parquet"

        # Verify files exist
        if not self.artificial_path.exists():
            available = list(self.data_dir.glob("*"))
            raise FileNotFoundError(
                f"Artificial dataset not found at: {self.artificial_path}\n"
                f"Available files: {[f.name for f in available]}"
            )
        if not self.labeled_path.exists():
            available = list(self.data_dir.glob("*"))
            raise FileNotFoundError(
                f"Labeled dataset not found at: {self.labeled_path}\n"
                f"Available files: {[f.name for f in available]}"
            )

    def load_parquets(self):
        """Load datasets with exact filenames"""
        try:
            print(f"Loading: {self.artificial_path}")
            print(f"Loading: {self.labeled_path}")

            artificial = pd.read_parquet(self.artificial_path)
            labeled = pd.read_parquet(self.labeled_path)

            return pd.concat([artificial, labeled])

        except Exception as e:
            raise RuntimeError(f"Failed to load parquet files: {str(e)}")

    def to_documents(self, df):
        """Convert DataFrame to LangChain documents"""
        return [
            Document(
                page_content=str(row["question"]),
                metadata={
                    "answer": str(row.get("answer", "")),
                    "context": str(row.get("context", "")),
                    "source": "artificial" if "artificial" in str(row) else "labeled"
                }
            )
            for _, row in df.iterrows()
        ]