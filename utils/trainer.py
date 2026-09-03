import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.metrics import dice_score
from config import DEVICE, LR, LOCAL_EPOCHS


def train_one_epoch(model, loader: DataLoader, optimizer, criterion, device=DEVICE):
    model.train()
    total_loss = 0.0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader: DataLoader, device=DEVICE):
    model.eval()
    all_dice = []
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)
            all_dice.append(dice_score(logits, masks))
    return sum(all_dice) / len(all_dice) if all_dice else 0.0


def train_local(model, train_loader, val_loader=None,
                epochs=LOCAL_EPOCHS, lr=LR, device=DEVICE):
    """Full local training loop used by single-site baseline."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    history = {"train_loss": [], "val_dice": []}

    for epoch in range(epochs):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        history["train_loss"].append(loss)

        if val_loader is not None:
            dice = evaluate(model, val_loader, device)
            history["val_dice"].append(dice)

    return model, history
