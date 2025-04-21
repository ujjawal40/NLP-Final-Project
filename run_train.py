import os
from datetime import time

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from torch import nn
import tqdm

from config import (
    DEVICE, TOKENIZER_NAME, TRAIN_FILE, VAL_FILE,
    EPOCHS, BATCH_SIZE, MAX_LEN, LEARNING_RATE,
    EMBEDDING_DIM, HIDDEN_DIM, NUM_CLASSES, NUM_LAYERS,
    DROPOUT, USE_ATTENTION
)

from Dataset import PubMedQADatasetHF
from model import PubMedLSTMClassifier
from train import train, evaluate


print(f"🖥️  Using device: {DEVICE}")

# 🌐 Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

# 📂 Load datasets
train_dataset = PubMedQADatasetHF(TRAIN_FILE, tokenizer, max_len=MAX_LEN)
val_dataset = PubMedQADatasetHF(VAL_FILE, tokenizer, max_len=MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# 🏗️ Initialize model
model = PubMedLSTMClassifier(
    vocab_size=tokenizer.vocab_size,
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    output_dim=NUM_CLASSES,
    num_layers=NUM_LAYERS,
    drop_prob=DROPOUT,
    use_attention=USE_ATTENTION
).to(DEVICE)

# 🎯 Loss & optimizer
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 🔁 Training loop
for epoch in range(EPOCHS):
    print(f"\n📚 Epoch {epoch + 1}/{EPOCHS}")

    start_time = time.time()

    # Wrap DataLoader in tqdm
    train_loader_tqdm = tqdm(train_loader, desc="Training", leave=False)
    train_loss, train_acc = train(model, train_loader_tqdm, criterion, optimizer, DEVICE)

    val_loader_tqdm = tqdm(val_loader, desc="Validating", leave=False)
    val_loss, val_acc, val_report = evaluate(model, val_loader_tqdm, criterion, DEVICE)

    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)

    print(f"⏱️  Epoch Duration: {minutes}m {seconds}s")
    print(f"✅ Train Loss: {train_loss:.4f} | Accuracy: {train_acc:.2%}")
    print(f"🧪 Val   Loss: {val_loss:.4f} | Accuracy: {val_acc:.2%}")
    print("📊 Validation Classification Report:\n", val_report)