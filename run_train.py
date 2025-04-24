import os
import time
from datetime import timedelta
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch import nn, optim
from tqdm import tqdm
import wandb

# Disable warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_SILENT"] = "true"

# Import config
from config import (
    DEVICE,
    TOKENIZER_NAME,
    TRAIN_FILE,
    VAL_FILE,
    EPOCHS,
    BATCH_SIZE,
    MAX_LEN,
    LEARNING_RATE,
    EMBEDDING_DIM,
    HIDDEN_DIM,
    NUM_CLASSES,
    NUM_LAYERS,
    DROPOUT,
    USE_ATTENTION,
    BIDIRECTIONAL
)

from Dataset import PubMedQADatasetHF
from model import PubMedLSTMClassifier
from train import train, evaluate


def main():
    # Initialize wandb
    wandb.init(
        project="pubmed-qa",
        config={
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "architecture": "LSTM+Attention",
            "dataset": "PubMedQA"
        },
        mode="offline"
    )

    print(f"🖥️ Using device: {DEVICE}")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    # Load and verify datasets
    train_dataset = PubMedQADatasetHF(TRAIN_FILE, tokenizer, max_len=MAX_LEN)
    val_dataset = PubMedQADatasetHF(VAL_FILE, tokenizer, max_len=MAX_LEN)

    print("\nDataset Verification:")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print("Class distribution in training set:")
    print(train_dataset.df["final_decision"].value_counts())

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE * 2,
        num_workers=4,
        pin_memory=True
    )

    # Initialize model with verification
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

    print("\nModel Architecture:")
    print(model)

    # Class weights with validation
    class_weights = torch.tensor(train_dataset.get_class_weights(), dtype=torch.float32).to(DEVICE)
    print(f"\nClass weights: {class_weights.cpu().numpy()}")
    assert len(class_weights) == NUM_CLASSES, \
        f"Class weights mismatch! Expected {NUM_CLASSES}, got {len(class_weights)}"

    criterion = nn.NLLLoss(weight=class_weights)

    # Optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01
    )

    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )

    # Training loop with robust error handling
    best_val_acc = 0
    for epoch in range(EPOCHS):
        try:
            print(f"\n📚 Epoch {epoch + 1}/{EPOCHS}")
            start_time = time.time()

            # Training phase
            train_loss, train_acc = train(
                model=model,
                dataloader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=DEVICE,
                scheduler=scheduler
            )

            # Validation phase
            val_loss, val_acc, val_report = evaluate(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                device=DEVICE
            )

            epoch_time = timedelta(seconds=time.time() - start_time)

            print(f"\n⏱ Time: {epoch_time}")
            print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.2%}")
            print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.2%}")
            print("Classification Report:\n", val_report)

            wandb.log({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "epoch": epoch,
                "lr": scheduler.get_last_lr()[0]
            })

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), "best_model.pt")
                print("🔥 New best model saved!")

        except Exception as e:
            print(f"❌ Error in epoch {epoch + 1}: {str(e)}")
            break

    wandb.finish()


if __name__ == "__main__":
    main()