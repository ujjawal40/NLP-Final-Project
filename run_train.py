import os
import time
from datetime import timedelta
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch import nn, optim
from tqdm import tqdm
import wandb
from config import *
from Dataset import PubMedQADatasetHF
from model import PubMedLSTMClassifier
from train import train, evaluate


def main():
    # 🚀 Experiment Tracking
    wandb.init(project="pubmed-qa", config={
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "architecture": "LSTM+Attention",
        "dataset": "PubMedQA"
    })

    print(f"🖥️ Using device: {DEVICE}")

    # 🏗️ Initialize components
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    # 📊 Datasets
    train_dataset = PubMedQADatasetHF(TRAIN_FILE, tokenizer, max_len=MAX_LEN, augment=True)
    val_dataset = PubMedQADatasetHF(VAL_FILE, tokenizer, max_len=MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2, num_workers=4)

    # 🧠 Model
    model = PubMedLSTMClassifier(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=NUM_CLASSES,
        num_layers=NUM_LAYERS,
        drop_prob=DROPOUT,
        use_attention=USE_ATTENTION
    ).to(DEVICE)


    class_weights = torch.tensor(train_dataset.get_class_weights()).to(DEVICE)
    criterion = nn.NLLLoss(weight=class_weights)


    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)


    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )


    best_val_acc = 0
    for epoch in range(EPOCHS):
        print(f"\n📚 Epoch {epoch + 1}/{EPOCHS}")
        start_time = time.time()


        train_loader_tqdm = tqdm(train_loader, desc=f"Train Epoch {epoch + 1}", leave=False)
        train_loss, train_acc = train(
            model, train_loader_tqdm, criterion, optimizer, DEVICE, scheduler
        )


        val_loader_tqdm = tqdm(val_loader, desc=f"Validate Epoch {epoch + 1}", leave=False)
        val_loss, val_acc, val_report = evaluate(model, val_loader_tqdm, criterion, DEVICE)


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
            "epoch": epoch
        })


        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pt")
            print("🔥 New best model saved!")

    wandb.finish()


if __name__ == "__main__":
    main()