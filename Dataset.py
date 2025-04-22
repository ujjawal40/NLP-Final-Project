import numpy as np
import pandas as pd
import re
from collections import Counter
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
import os
import random
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModel

nltk.download('stopwords')

class PubMedQADatasetHF(Dataset):
    def __init__(self, parquet_filename, tokenizer, max_len=256, show_graph = True):
        base_dir = os.getcwd()
        file_path = os.path.join(base_dir, "Dataset", parquet_filename)

        self.df = pd.read_parquet(file_path)
        self.tokenizer = tokenizer
        self.max_len = max_len

        # Combine question + context
        self.df["text"] = self.df.apply(
            lambda x: x["question"] + " " + " ".join(x["context"]),
            axis=1
        )

        # Label mapping
        self.label_map = {"yes": 0, "no": 1, "maybe": 2}
        self.df["label"] = self.df["final_decision"].map(self.label_map)

        # 📊 Class Distribution
        class_counts = self.df["final_decision"].value_counts()
        print("\n🔢 Number of classes:", len(class_counts))
        print("🧮 Class distribution:\n", class_counts)

        if show_graph:

            plt.figure(figsize=(6, 4))
            sns.barplot(x=class_counts.index, y=class_counts.values, palette="magma")
            plt.title("Class Distribution in PubMedQA")
            plt.xlabel("Class")
            plt.ylabel("Count")
            plt.grid(axis='y', linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.show()

        # Pre-tokenize
        self.encodings = self.tokenizer(
            self.df["text"].tolist(),
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.df.iloc[idx]["label"], dtype=torch.long)
        return item



def main():
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  # or any other HF model

    # Load dataset and trigger analysis
    _ = PubMedQADatasetHF("pubmedqa_artificial.parquet", tokenizer)


if __name__ == "__main__":
    main()