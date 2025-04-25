import pandas as pd
from langchain.schema import Document
from config import Config


class PubMedQADataLoader:
    def __init__(self):
        self.config = Config()

    def load_parquets(self):
        """Load both dataset files"""
        df_art = pd.read_parquet(self.config.artificial_path)
        df_lbl = pd.read_parquet(self.config.labeled_path)
        return pd.concat([df_art, df_lbl])

    def to_documents(self, df):
        return [
            Document(
                page_content=f"Q: {row['question']}\nContext: {row['context']}",
                metadata={
                    "answer": row['long_answer'],
                    "label": row['final_decision']
                }
            ) for _, row in df.iterrows()
        ]