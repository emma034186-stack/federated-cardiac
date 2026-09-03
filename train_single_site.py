"""
Single-site baseline: train U-Net on Hospital A's data only.
Saves model checkpoint and per-epoch metrics to results/.
"""
import os
import json
import torch
import torch.nn as nn
from tqdm import tqdm

from models.unet import build_model
from data.split import build_single_site
from utils.trainer import train_one_epoch, evaluate
from config import DEVICE, LR, RESULTS_DIR

NUM_EPOCHS = 50
HOSPITAL_ID = 0   # Hospital A as the single site


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=== Single-Site Baseline (Hospital A) ===")
    train_loader, val_loader, test_loader = build_single_site(hospital_id=HOSPITAL_ID)
    print(f"Train slices: {len(train_loader.dataset)} | "
          f"Val slices: {len(val_loader.dataset)} | "
          f"Test slices: {len(test_loader.dataset)}")

    model = build_model().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    history = {"train_loss": [], "val_dice": []}
    best_dice = 0.0

    for epoch in tqdm(range(1, NUM_EPOCHS + 1), desc="Epochs"):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        dice = evaluate(model, val_loader, DEVICE)
        scheduler.step()

        history["train_loss"].append(loss)
        history["val_dice"].append(dice)

        if dice > best_dice:
            best_dice = dice
            torch.save(model.state_dict(),
                       os.path.join(RESULTS_DIR, "single_site_best.pt"))

    test_dice = evaluate(model, test_loader, DEVICE)
    history["test_dice"] = test_dice
    print(f"\nBest Val Dice: {best_dice:.4f} | Test Dice: {test_dice:.4f}")

    with open(os.path.join(RESULTS_DIR, "single_site_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print("Saved results/single_site_history.json")


if __name__ == "__main__":
    main()
