import torch
from transformers import TrainingArguments

# Device and model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKENIZER_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
MODEL_NAME = TOKENIZER_NAME

# Data configuration
TRAIN_FILE = "pubmedqa_artificial.parquet"
VAL_FILE = "pubmedqa_labeled.parquet"
MAX_LEN = 256  # Reduced from 512 if using LSTM
AUGMENT_DATA = True

# Training hyperparameters
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 1e-3  # Increased for LSTM
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1

# Model architecture
EMBEDDING_DIM = 128  # Size of word embeddings
HIDDEN_DIM = 256     # LSTM hidden state size
NUM_LAYERS = 2       # Number of LSTM layers
DROPOUT = 0.4        # Dropout probability
USE_ATTENTION = True  # Whether to use attention mechanism
NUM_CLASSES = 3      # yes/no/maybe
BIDIRECTIONAL = True # Whether LSTM is bidirectional


# Remove or update the TrainingArguments if not using Transformers trainer
# Simple version compatible with older Transformers:
TRAINING_ARGS = {
    'output_dir': './results',
    'num_train_epochs': EPOCHS,
    'per_device_train_batch_size': BATCH_SIZE,
    'per_device_eval_batch_size': BATCH_SIZE*2,
    'warmup_ratio': WARMUP_RATIO,
    'weight_decay': WEIGHT_DECAY,
    'logging_dir': './logs',
    'logging_steps': 10,
    'save_total_limit': 2,
    'fp16': torch.cuda.is_available(),
}