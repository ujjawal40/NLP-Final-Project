import torch
from sklearn.metrics import classification_report

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for batch in dataloader:
        input_ids, labels = batch["input_ids"].to(device), batch["labels"].to(device)
        optimizer.zero_grad()
        output = model(input_ids)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(output, dim=1)
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
