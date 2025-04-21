import torch

# Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tokenizer / model checkpoint
TOKENIZER_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"

# File paths
TRAIN_FILE = "pubmedqa_artificial.parquet"
VAL_FILE = "pubmedqa_labeled.parquet"

# Training hyperparameters
EPOCHS = 10
BATCH_SIZE = 16
MAX_LEN = 256
LEARNING_RATE = 1e-3

# Model architecture
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
NUM_CLASSES = 3
NUM_LAYERS = 2
DROPOUT = 0.4
USE_ATTENTION = True
