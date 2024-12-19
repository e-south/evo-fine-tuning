# Training and evaluation logic

import torch

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    for input_ids, labels in dataloader:
        input_ids, labels = input_ids.to(device), labels.to(device)
        outputs = model(input_ids)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for input_ids, labels in dataloader:
            input_ids, labels = input_ids.to(device), labels.to(device)
            outputs = model(input_ids)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    return total_loss / len(dataloader)
