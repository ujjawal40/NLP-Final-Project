import os
import time
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch import nn, optim
from tqdm import tqdm
import wandb
from typing import Dict, Any, Tuple
from config import config

# Disable warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_SILENT"] = "true"

# Import config
DEVICE = config.DEVICE
TOKENIZER_NAME = config.TOKENIZER_NAME
TRAIN_FILE = config.TRAIN_FILE
VAL_FILE = config.VAL_FILE
EPOCHS = config.EPOCHS
BATCH_SIZE = config.BATCH_SIZE
MAX_LEN = config.MAX_LEN
LEARNING_RATE = config.LEARNING_RATE  # Fix typo if present
EMBEDDING_DIM = config.EMBEDDING_DIM
HIDDEN_DIM = config.HIDDEN_DIM
NUM_CLASSES = config.NUM_CLASSES
NUM_LAYERS = config.NUM_LAYERS
DROPOUT = config.DROPOUT
USE_ATTENTION = config.USE_ATTENTION
BIDIRECTIONAL = config.BIDIRECTIONAL
GRAD_ACCUM_STEPS = config.GRAD_ACCUM_STEPS
MAX_GRAD_NORM = config.MAX_GRAD_NORM
from Dataset import PubMedQADatasetHF
from model import PubMedLSTMClassifier
from train import train_epochs  # Using the enhanced training function


def initialize_wandb() -> None:
    """Initialize Weights & Biases logging"""
    wandb.init(
        project="pubmed-qa",
        config={
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "architecture": "LSTM+Attention",
            "dataset": "PubMedQA",
            "embedding_dim": EMBEDDING_DIM,
            "hidden_dim": HIDDEN_DIM
        },
        mode="offline"
    )


def load_datasets(tokenizer: AutoTokenizer) -> Tuple[PubMedQADatasetHF, PubMedQADatasetHF]:
    """Load and validate training and validation datasets"""
    train_dataset = PubMedQADatasetHF(
        TRAIN_FILE,
        tokenizer,
        max_len=MAX_LEN,
        augment=True
    )
    val_dataset = PubMedQADatasetHF(
        VAL_FILE,
        tokenizer,
        max_len=MAX_LEN
    )

    print("\n=== Dataset Verification ===")
    print(f"Train samples: {len(train_dataset):,}")
    print(f"Validation samples: {len(val_dataset):,}")
    print("\nClass distribution:")
    print(train_dataset.df["final_decision"].value_counts())

    return train_dataset, val_dataset


def create_data_loaders(
        train_dataset: PubMedQADatasetHF,
        val_dataset: PubMedQADatasetHF
) -> Tuple[DataLoader, DataLoader]:
    """Create optimized data loaders"""
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE * 2,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader


def initialize_model(tokenizer: AutoTokenizer) -> PubMedLSTMClassifier:
    """Initialize model with verification"""
    model = PubMedLSTMClassifier(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=NUM_CLASSES,
        num_layers=NUM_LAYERS,
        drop_prob=DROPOUT,
        bidirectional=BIDIRECTIONAL,
        use_attention=USE_ATTENTION
    ).to(DEVICE)

    print("\n=== Model Architecture ===")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    return model


def get_class_weights(dataset: PubMedQADatasetHF) -> torch.Tensor:
    """Calculate and validate class weights"""
    weights = torch.tensor(dataset.get_class_weights(), dtype=torch.float32).to(DEVICE)
    assert len(weights) == NUM_CLASSES, \
        f"Class weights mismatch! Expected {NUM_CLASSES}, got {len(weights)}"
    print("\nClass weights:", weights.cpu().numpy())
    return weights


def main():
    # Initialize tracking
    initialize_wandb()
    print(f"\nUsing device: {DEVICE}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    # Prepare data
    train_dataset, val_dataset = load_datasets(tokenizer)
    train_loader, val_loader = create_data_loaders(train_dataset, val_dataset)

    # Initialize model
    model = initialize_model(tokenizer)

    # Loss function with class weights
    criterion = nn.NLLLoss(weight=get_class_weights(train_dataset))

    # Optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01
    )

    total_steps = len(train_loader) * EPOCHS // GRAD_ACCUM_STEPS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )

    # Training with enhanced loop
    history = train_epochs(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=DEVICE,
        epochs=EPOCHS,
        scheduler=scheduler,
        grad_clip=MAX_GRAD_NORM,
        accumulation_steps=GRAD_ACCUM_STEPS
    )

    # Save final model
    torch.save(model.state_dict(), "final_model.pt")
    wandb.finish()


if __name__ == "__main__":
    main()