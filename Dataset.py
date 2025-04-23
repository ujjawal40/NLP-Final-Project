import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import random
from typing import List


class PubMedQADatasetHF(Dataset):
    def __init__(self, parquet_filename, tokenizer, max_len=256, augment=False, show_graph=True):
        # Load data
        self.df = pd.read_parquet(os.path.join("Dataset", parquet_filename))

        # Validate and clean data
        self._clean_data()

        self.tokenizer = tokenizer
        self.max_len = max_len
        self.augment = augment
        self.show_graph = show_graph

        # Data augmentation if enabled
        if self.augment:
            self._augment_data()

        # Initialize encodings and labels
        self.encodings = None
        self.labels = None
        self._preprocess_data()

    def _clean_data(self):
        """Ensure all text fields are strings and contexts are properly formatted"""
        self.df["question"] = self.df["question"].astype(str)

        # Convert context list elements to strings
        self.df["context"] = self.df["context"].apply(
            lambda ctx: [str(x) for x in ctx] if isinstance(ctx, list) else [str(ctx)]
        )

    def _preprocess_data(self):
        """Tokenize and prepare all samples"""
        texts = []
        for _, row in self.df.iterrows():
            # Combine question and context with proper separation
            text = row["question"] + " " + " ".join(row["context"])
            texts.append(text.strip())

        # Tokenize all texts
        self.encodings = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        # Convert labels to tensor
        self.labels = torch.tensor(
            self.df["final_decision"].map({"yes": 0, "no": 1, "maybe": 2}).values,
            dtype=torch.long
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': self.labels[idx]
        }

    def _augment_data(self):
        """Medical-specific data augmentation"""
        new_rows = []
        for _, row in self.df.iterrows():
            # Skip augmentation for maybe class to avoid confusion
            if row["final_decision"] == "maybe":
                continue

            # Question paraphrasing
            if random.random() < 0.3:
                new_q = self._paraphrase_question(row['question'])
                new_rows.append({
                    'question': new_q,
                    'context': row['context'],
                    'final_decision': row['final_decision']
                })

        if new_rows:
            self.df = pd.concat([self.df, pd.DataFrame(new_rows)], ignore_index=True)

    def _paraphrase_question(self, question):
        """Simple question paraphrasing for medical text"""
        paraphrases = {
            "Does": "Is there evidence that",
            "Is": "Does the evidence show that",
            "Are": "Do the findings indicate that",
            "Can": "Is it possible that",
            "Should": "Is it recommended that"
        }
        for k, v in paraphrases.items():
            if question.startswith(k):
                return question.replace(k, v, 1)
        return question

    def get_class_weights(self):
        """Calculate class weights for imbalance handling"""
        class_counts = self.df["final_decision"].value_counts().sort_index()
        return 1. / class_counts.values