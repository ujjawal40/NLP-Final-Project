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
nltk.download('stopwords')
import os
import random
# import matplotlib.pyplot as plt
# import seaborn as
from transformers import AutoTokenizer, AutoModel



#GPU check
is_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if is_cuda else "cpu")
print("Device:", device)

class PubMedQADatasetHF(Dataset):
    def __init__(self, parquet_filename, tokenizer, max_len=256):
        # Dynamic path using os
        base_dir = os.getcwd()
        file_path = os.path.join(base_dir, parquet_filename)

        # Read parquet
        self.df = pd.read_parquet(file_path)
        self.tokenizer = tokenizer
        self.max_len = max_len

        # Combine question + context
        self.df["text"] = self.df.apply(
            lambda x: x["question"] + " " + " ".join(x["context"]),
            axis=1
        )

        # Map labels to integers
        self.label_map = {"yes": 0, "no": 1, "maybe": 2}
        self.df["label"] = self.df["final_decision"].map(self.label_map)

        # Pre-tokenize all inputs
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
        # Return tokenized input and label
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.df.iloc[idx]["label"], dtype=torch.long)
        return item