import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class PubMedQAConfig:
    # Device configuration
    DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tokenizer settings
    TOKENIZER_NAME: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"

    # Data paths and processing
    TRAIN_FILE: str = "pubmedqa_artificial.parquet"
    VAL_FILE: str = "pubmedqa_labeled.parquet"
    MAX_LEN: int = 384  # Optimized for LSTM efficiency
    AUGMENT_DATA: bool = True

    # Training hyperparameters
    EPOCHS: int = 12  # Increased for progressive training
    BATCH_SIZE: int = 48  # Increased with gradient accumulation
    GRAD_ACCUM_STEPS: int = 2  # New - for effective batch size of 96
    LEARNING_RATE: float = 2e-3  # Adjusted for LSTM
    WEIGHT_DECAY: float = 0.02  # Increased regularization
    WARMUP_STEPS: int = 500  # More precise than ratio

    # Model architecture
    EMBEDDING_DIM: int = 256  # Increased for better representations
    HIDDEN_DIM: int = 512  # Scaled with embedding dim
    NUM_LAYERS: int = 2
    DROPOUT: float = 0.3  # Slightly reduced
    USE_ATTENTION: bool = True
    BIDIRECTIONAL: bool = True
    NUM_CLASSES: int = 3

    # Training monitoring
    LOGGING_DIR: str = "./logs"
    SAVE_DIR: str = "./checkpoints"
    LOG_STEPS: int = 25  # More frequent logging

    # Early stopping
    EARLY_STOPPING_PATIENCE: int = 3
    EARLY_STOPPING_DELTA: float = 0.001

    # New - Gradient clipping
    MAX_GRAD_NORM: float = 1.0

    # New - Class weights (adjust based on your dataset)
    CLASS_WEIGHTS: Optional[list] = None  # Will be calculated automatically

    def __post_init__(self):
        """Calculate derived parameters"""
        self.EFFECTIVE_BATCH_SIZE = self.BATCH_SIZE * self.GRAD_ACCUM_STEPS
        if torch.cuda.is_available():
            self.FP16 = True
            self.BATCH_SIZE = min(self.BATCH_SIZE, 64)  # Safe upper limit
        else:
            self.FP16 = False

    def get_training_args(self):
        """Generate Transformers-compatible training arguments"""
        return {
            'output_dir': self.SAVE_DIR,
            'per_device_train_batch_size': self.BATCH_SIZE,
            'per_device_eval_batch_size': self.BATCH_SIZE * 2,
            'num_train_epochs': self.EPOCHS,
            'gradient_accumulation_steps': self.GRAD_ACCUM_STEPS,
            'learning_rate': self.LEARNING_RATE,
            'weight_decay': self.WEIGHT_DECAY,
            'warmup_steps': self.WARMUP_STEPS,
            'logging_dir': self.LOGGING_DIR,
            'logging_steps': self.LOG_STEPS,
            'save_total_limit': 2,
            'fp16': self.FP16,
            'gradient_clipping': self.MAX_GRAD_NORM
        }


# Instantiate config
config = PubMedQAConfig()

# Example usage:
# training_args = config.get_training_args()
# print(config.HIDDEN_DIM)