import torch
from sklearn.metrics import classification_report


def train(model, dataloader, criterion, optimizer, device, scheduler=None):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for batch in dataloader:
        inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = criterion(outputs, labels)

        # Gradient accumulation for larger effective batch size
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
        optimizer.step()
        if scheduler:
            scheduler.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(dataloader), correct / total
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids, labels = batch["input_ids"].to(device), batch["labels"].to(device)
            output = model(input_ids)
            loss = criterion(output, labels)

            total_loss += loss.item()
            preds = torch.argmax(output, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    report = classification_report(all_labels, all_preds, target_names=["yes", "no", "maybe"])
    return total_loss / len(dataloader), correct / total, report
