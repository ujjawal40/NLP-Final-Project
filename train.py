import torch
import numpy as np
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm
from typing import Dict, Tuple



def train(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        grad_clip: float = 1.0,
        accumulation_steps: int = 1
) -> Dict[str, float]:
    """
    Enhanced training function with gradient accumulation and clipping

    Args:
        model: The model to train
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Target device (cuda/cpu)
        scheduler: Optional learning rate scheduler
        grad_clip: Gradient clipping value
        accumulation_steps: Number of gradient accumulation steps

    Returns:
        Dictionary containing:
            - train_loss: Average training loss
            - train_acc: Training accuracy
            - train_f1: Macro F1 score
            - lr: Current learning rate
    """
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training")):
        inputs = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels) / accumulation_steps

        # Backward pass
        loss.backward()

        # Gradient accumulation
        if (batch_idx + 1) % accumulation_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad()

            if scheduler:
                scheduler.step()

        # Metrics calculation
        total_loss += loss.item() * accumulation_steps
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    # Calculate metrics
    train_loss = total_loss / len(dataloader)
    train_acc = np.mean(np.array(all_preds) == np.array(all_labels))
    train_f1 = f1_score(all_labels, all_preds, average='macro')

    return {
        'train_loss': train_loss,
        'train_acc': train_acc,
        'train_f1': train_f1,
        'lr': optimizer.param_groups[0]['lr']
    }


def evaluate(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module,
        device: torch.device,
        detailed_report: bool = True
) -> Tuple[float, float, str]:
    """
    Enhanced evaluation function with detailed metrics

    Args:
        model: The model to evaluate
        dataloader: Validation data loader
        criterion: Loss function
        device: Target device (cuda/cpu)
        detailed_report: Whether to generate full classification report

    Returns:
        Tuple containing:
            - val_loss: Average validation loss
            - val_acc: Validation accuracy
            - report: Classification report string
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            inputs = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    val_loss = total_loss / len(dataloader)
    val_acc = np.mean(np.array(all_preds) == np.array(all_labels))

    # Generate report
    report = ""
    if detailed_report:
        report = classification_report(
            all_labels, all_preds,
            target_names=["yes", "no", "maybe"],
            digits=4,
            output_dict=False
        )

    return val_loss, val_acc, report


def train_epochs(
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        epochs: int,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        grad_clip: float = 1.0,
        accumulation_steps: int = 1
) -> Dict[str, list]:
    """
    Complete training loop with epoch-level tracking

    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Target device
        epochs: Number of epochs
        scheduler: Optional LR scheduler
        grad_clip: Gradient clipping value
        accumulation_steps: Gradient accumulation steps

    Returns:
        Dictionary containing training history:
            - train_loss: List of training losses
            - train_acc: List of training accuracies
            - val_loss: List of validation losses
            - val_acc: List of validation accuracies
            - lrs: List of learning rates
    """
    history = {
        'train_loss': [],
        'train_acc': [],
        'train_f1': [],
        'val_loss': [],
        'val_acc': [],
        'lrs': []
    }

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        # Training phase
        train_metrics = train(
            model, train_loader, criterion, optimizer,
            device, scheduler, grad_clip, accumulation_steps
        )

        # Validation phase
        val_loss, val_acc, report = evaluate(model, val_loader, criterion, device)

        # Update history
        history['train_loss'].append(train_metrics['train_loss'])
        history['train_acc'].append(train_metrics['train_acc'])
        history['train_f1'].append(train_metrics['train_f1'])
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lrs'].append(train_metrics['lr'])

        # Print metrics
        print(
            f"Train Loss: {train_metrics['train_loss']:.4f} | Acc: {train_metrics['train_acc']:.4f} | F1: {train_metrics['train_f1']:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")
        print("\nClassification Report:")
        print(report)

    return history