import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import random
import re
from typing import List, Dict, Union
from collections import Counter


class PubMedQADatasetHF(Dataset):
    def __init__(self, parquet_filename: str, tokenizer, max_len: int = 512, augment: bool = False,
                 show_graph: bool = True):
        """
        Enhanced PubMedQA Dataset with medical-specific augmentation

        Args:
            parquet_filename: Path to the parquet file
            tokenizer: HuggingFace tokenizer
            max_len: Maximum sequence length
            augment: Whether to apply data augmentation
            show_graph: Show class distribution graph
        """
        # Load and validate data
        self.df = self._load_and_validate_data(parquet_filename)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.augment = augment
        self.label_map = {"yes": 0, "no": 1, "maybe": 2}

        # Data processing
        if self.augment:
            self._apply_augmentation()

        self._preprocess_data()

        if show_graph:
            self._plot_class_distribution()

    def _load_and_validate_data(self, filename: str) -> pd.DataFrame:
        """Load and validate the parquet file"""
        df = pd.read_parquet(os.path.join("Dataset", filename))

        # Validate required columns
        required_columns = {"question", "context", "final_decision"}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"Missing required columns: {required_columns - set(df.columns)}")

        return df

    def _clean_text(self, text: Union[str, List[str], dict]) -> str:
        """Enhanced text cleaning that handles multiple input types"""
        if isinstance(text, dict):
            # Handle dictionary input (extract values and join)
            return ' '.join(str(v) for v in text.values()).strip()
        elif isinstance(text, list):
            # Handle list input
            return ' '.join(str(t) for t in text).strip()
        elif isinstance(text, str):
            # Handle string input
            return text.strip()
        else:
            # Fallback for other types
            return str(text).strip()

    def _apply_augmentation(self):
        """Medical-specific data augmentation"""
        augmented_rows = []

        for _, row in self.df.iterrows():
            # Skip augmentation for maybe class
            if row["final_decision"] == "maybe":
                continue

            # Apply augmentation with 30% probability
            if random.random() < 0.3:
                augmented_rows.append({
                    'question': self._paraphrase_medical_question(row['question']),
                    'context': row['context'],
                    'final_decision': row['final_decision']
                })

        if augmented_rows:
            self.df = pd.concat([self.df, pd.DataFrame(augmented_rows)], ignore_index=True)

    def _paraphrase_medical_question(self, question: str) -> str:
        """Enhanced medical question paraphrasing"""
        paraphrase_patterns = [
            (r'^Does (.+)', ['Is there evidence that \\1', 'Can we conclude that \\1']),
            (r'^Is (.+)', ['Does \\1', 'Would you say \\1']),
            (r'^Can (.+)', ['Is it possible to \\1', 'Could \\1']),
            (r'^Should (.+)', ['Is it recommended to \\1', 'Would you advise to \\1'])
        ]

        for pattern, replacements in paraphrase_patterns:
            if re.match(pattern, question):
                return random.choice(replacements)
        return question

    def _preprocess_data(self):
        """Tokenize and prepare all samples"""
        texts = [
            f"{self._clean_text(row['question'])} [SEP] {self._clean_text(row['context'])}"
            for _, row in self.df.iterrows()
        ]

        # Tokenize with proper special tokens
        self.encodings = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )

        # Convert labels to tensor
        self.labels = torch.tensor(
            self.df['final_decision'].map(self.label_map).values,
            dtype=torch.long
        )

    def _plot_class_distribution(self):
        """Plot class distribution (optional)"""
        import matplotlib.pyplot as plt

        counts = self.df['final_decision'].value_counts()
        counts.plot(kind='bar')
        plt.title('Class Distribution')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.show()

    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate inverse class frequency weights
        Returns:
            Tensor of shape (num_classes,) containing weights
        """
        counts = self.df['final_decision'].value_counts().reindex(
            list(self.label_map.keys()),
            fill_value=1e-6  # Smoothing for missing classes
        )
        weights = 1.0 / counts
        return torch.tensor(weights.values, dtype=torch.float32)

    def __len__(self) -> int:
        """Returns number of samples in dataset"""
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample by index
        Returns:
            Dictionary with:
                - input_ids: Tokenized text
                - attention_mask: Attention mask
                - labels: Target class
        """
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': self.labels[idx]
        }

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, tokenizer, **kwargs):
        """Alternative constructor from existing DataFrame"""
        dataset = cls.__new__(cls)
        dataset.df = df
        dataset.tokenizer = tokenizer
        dataset.max_len = kwargs.get('max_len', 512)
        dataset.label_map = {"yes": 0, "no": 1, "maybe": 2}
        dataset._preprocess_data()
        return dataset