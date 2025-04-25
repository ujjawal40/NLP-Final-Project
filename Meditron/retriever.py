from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F


class PubMedRetriever:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
        self.model = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")

    def embed(self, texts):
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return F.normalize(outputs.last_hidden_state[:, 0], p=2, dim=1)  # CLS token pooling