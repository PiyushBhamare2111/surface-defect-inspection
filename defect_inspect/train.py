"""Fine-tune ResNet18 on NEU-CLS with early stopping.

Usage:  python -m defect_inspect.train [--epochs 30] [--patience 6]
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import DEFAULT_WEIGHTS, IMAGE_DIR, MODELS_DIR, SEED
from .data import (DefectDataset, discover, eval_transform, save_split,
                   stratified_split, train_transform)
from .model import build_model, get_device, save_checkpoint


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None) -> tuple[float, float]:
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=scaler is not None):
            logits = model(x)
            loss = criterion(logits, y)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        loss_sum += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        total += len(y)
    return loss_sum / total, correct / total


@torch.no_grad()
def evaluate_loader(model, loader, criterion, device) -> tuple[float, float]:
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += criterion(logits, y).item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        total += len(y)
    return loss_sum / total, correct / total


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=6, help="early-stopping patience (epochs)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", type=Path, default=DEFAULT_WEIGHTS)
    args = ap.parse_args(argv)

    seed_everything()
    device = get_device()
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    samples = discover()
    train, val, test = stratified_split(samples)
    MODELS_DIR.mkdir(exist_ok=True)
    save_split({"train": train, "val": val, "test": test}, MODELS_DIR / "split.json", IMAGE_DIR)
    print(f"Split: train={len(train)} val={len(val)} test={len(test)}")

    g = torch.Generator().manual_seed(SEED)
    pin = device.type == "cuda"
    train_dl = DataLoader(DefectDataset(train, train_transform()), batch_size=args.batch_size,
                          shuffle=True, num_workers=args.workers, pin_memory=pin, generator=g,
                          persistent_workers=args.workers > 0)
    val_dl = DataLoader(DefectDataset(val, eval_transform()), batch_size=64,
                        num_workers=args.workers, pin_memory=pin,
                        persistent_workers=args.workers > 0)

    model = build_model(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    best_loss, best_epoch, bad, history = float("inf"), 0, 0, []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_dl, optimizer, criterion, device, scaler)
        va_loss, va_acc = evaluate_loader(model, val_dl, criterion, device)
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                        "val_loss": va_loss, "val_acc": va_acc})
        flag = ""
        if va_loss < best_loss - 1e-4:
            best_loss, best_epoch, bad = va_loss, epoch, 0
            save_checkpoint(model, args.out, {"epoch": epoch, "val_acc": va_acc, "val_loss": va_loss})
            flag = "  * saved"
        else:
            bad += 1
        print(f"epoch {epoch:2d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f}{flag}")
        if bad >= args.patience:
            print(f"Early stopping: no val-loss improvement for {args.patience} epochs.")
            break
    elapsed = time.time() - t0
    print(f"Best epoch {best_epoch} (val loss {best_loss:.4f}); training took {elapsed/60:.1f} min")
    (MODELS_DIR / "train_history.json").write_text(json.dumps(
        {"history": history, "best_epoch": best_epoch, "seconds": elapsed, "device": str(device)},
        indent=1))


if __name__ == "__main__":
    main()
