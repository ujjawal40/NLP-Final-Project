# Enhanced config.py
import torch
from transformers import TrainingArguments

# Device and model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKENIZER_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
MODEL_NAME = TOKENIZER_NAME  # Can be different for hybrid models

# Data configuration
TRAIN_FILE = "pubmedqa_artificial.parquet"
VAL_FILE = "pubmedqa_labeled.parquet"
TEST_FILE = None  # Add if available
MAX_LEN = 512  # Increased for medical context
AUGMENT_DATA = True

# Training hyperparameters
EPOCHS = 10
BATCH_SIZE = 16  # Reduced for BERT
GRADIENT_ACCUMULATION_STEPS = 2  # Simulates larger batch size
LEARNING_RATE = 2e-5  # Smaller for transformer models
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 500

# Model architecture
DROPOUT = 0.1  # Reduced for BERT
USE_FREEZING = True  # Freeze early layers
NUM_CLASSES = 3

# Evaluation
METRICS = ['accuracy', 'f1', 'precision', 'recall']

# TrainingArguments for Transformers
TRAINING_ARGS = TrainingArguments(
    output_dir='./results',
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE*2,
    warmup_steps=WARMUP_STEPS,
    weight_decay=WEIGHT_DECAY,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    fp16=torch.cuda.is_available(),
