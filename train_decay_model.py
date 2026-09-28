"""
Edge-Vision: Decay Regressor Training Script
=============================================
Trains the PyTorch DecayRegressorNN on labelled produce images and saves
a checkpoint to models/decay_regressor.pt which the app will load
automatically on next startup.

USAGE
-----
1. Add your images to sample_images/ (or anywhere accessible)
2. Edit data/training_labels.csv -- one row per image:
       image_path,category,freshness,days_left
       sample_images/fresh_tomato.jpg,Tomato,94.0,8
       sample_images/rotten_tomato.jpg,Tomato,7.0,1
3. Run:
       .venv\\Scripts\\python train_decay_model.py
   Optional flags:
       --csv   path/to/labels.csv   (default: data/training_labels.csv)
       --epochs 150                 (default: 100)
       --lr    0.001                (default: 0.001)
       --val   0.2                  (default: 20% held out for validation)
       --aug   5                    (default: augment each image 5x with jitter)

OUTPUT
------
  models/decay_regressor.pt   <- loaded automatically by the app
  training_report.txt         <- loss curve + per-class accuracy summary
"""

import os
import sys
import csv
import math
import random
import argparse
import textwrap
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ── Ensure project root is on PYTHONPATH so edge_vision imports work ──────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from edge_vision.cv.decay_model import DecayRegressorNN, PyTorchDecayModel, BASELINE_SHELF_LIFE
from edge_vision.config import MODELS_DIR, SAMPLE_IMAGES_DIR

CHECKPOINT_PATH = MODELS_DIR / "decay_regressor.pt"
REPORT_PATH     = PROJECT_ROOT / "training_report.txt"

# ── Colour-jitter augmentation helpers ────────────────────────────────────────

def augment_frame(img: np.ndarray) -> np.ndarray:
    """Apply random brightness, contrast, saturation and horizontal-flip."""
    # Random brightness / contrast
    alpha = random.uniform(0.75, 1.25)   # contrast
    beta  = random.randint(-20, 20)       # brightness
    img   = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    # Random hue / saturation shift in HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int32)
    hsv[:, :, 0] = np.clip(hsv[:, :, 0] + random.randint(-8, 8), 0, 179)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] + random.randint(-30, 30), 0, 255)
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # Random horizontal flip
    if random.random() > 0.5:
        img = cv2.flip(img, 1)

    # Random crop-and-resize (±10%)
    h, w = img.shape[:2]
    cx = random.randint(0, max(1, w // 10))
    cy = random.randint(0, max(1, h // 10))
    img = img[cy:h - cy, cx:w - cx]
    if img.size > 0:
        img = cv2.resize(img, (w, h))

    return img


# ── Dataset ───────────────────────────────────────────────────────────────────

class ProduceDecayDataset(Dataset):
    """
    Loads rows from training_labels.csv and returns (feature_vector, freshness_target, days_target).
    Each original image is augmented `aug_factor` times.
    """

    def __init__(self, rows: list, feature_extractor: PyTorchDecayModel, aug_factor: int = 5):
        self.samples = []
        extractor = feature_extractor

        for row in rows:
            img_path  = row["image_path"].strip()
            category  = row["category"].strip()
            freshness = float(row["freshness"]) / 100.0   # normalise 0-1
            days      = float(row["days_left"])

            if not os.path.exists(img_path):
                # Try relative to project root
                alt = PROJECT_ROOT / img_path
                if alt.exists():
                    img_path = str(alt)
                else:
                    print(f"  [WARN] Image not found, skipping: {img_path}")
                    continue

            img = cv2.imread(img_path)
            if img is None:
                print(f"  [WARN] Could not read image, skipping: {img_path}")
                continue

            # Original image
            feats = extractor.extract_decay_features(img, category)
            self.samples.append((feats, freshness, days))

            # Augmented copies
            for _ in range(aug_factor):
                aug = augment_frame(img.copy())
                feats_aug = extractor.extract_decay_features(aug, category)
                self.samples.append((feats_aug, freshness, days))

        random.shuffle(self.samples)
        print(f"  Dataset: {len(rows)} images × {aug_factor + 1} = {len(self.samples)} samples total")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feats, freshness, days = self.samples[idx]
        x = torch.tensor(feats, dtype=torch.float32)
        y_fresh = torch.tensor([freshness], dtype=torch.float32)
        y_days  = torch.tensor([days], dtype=torch.float32)
        return x, y_fresh, y_days


# ── Training loop ──────────────────────────────────────────────────────────────

def train(csv_path: str, epochs: int, lr: float, val_split: float, aug_factor: int):
    print("\n+======================================================+")
    print("|   Edge-Vision  -  Decay Regressor Training Script  |")
    print("+======================================================+\n")

    # ── Load CSV ──────────────────────────────────────────────────────────────
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"[ERROR] Labels CSV not found: {csv_path}")
        print("  Create data/training_labels.csv with columns: image_path,category,freshness,days_left")
        sys.exit(1)

    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if len(rows) == 0:
        print("[ERROR] No rows found in CSV. Add at least a few labelled images.")
        sys.exit(1)

    print(f"[1/5] Loaded {len(rows)} labelled samples from {csv_path}")

    # ── Build feature extractor (CPU only, used for dataset prep) ─────────────
    extractor = PyTorchDecayModel()

    # ── Build dataset & split ─────────────────────────────────────────────────
    random.shuffle(rows)
    n_val   = max(1, int(len(rows) * val_split))
    n_train = max(1, len(rows) - n_val)
    train_rows = rows[:n_train]
    val_rows   = rows[n_train:]

    print(f"[2/5] Building dataset  (train={n_train} | val={n_val} | aug×{aug_factor + 1})")
    train_ds = ProduceDecayDataset(train_rows, extractor, aug_factor)
    val_ds   = ProduceDecayDataset(val_rows,   extractor, aug_factor=0)  # no aug on val

    # Use a batch size that works even with tiny datasets
    batch_size = min(32, max(1, len(train_ds)))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False)

    # ── Model, optimiser, loss ────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[3/5] Training on: {device}")

    model     = DecayRegressorNN(in_features=16).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs, eta_min=lr * 0.01)
    mse_loss  = nn.MSELoss()

    # Max-days normalisation factor for loss balancing
    max_days = max(BASELINE_SHELF_LIFE.values())  # 18 days (apple)

    history = []
    best_val_loss = float("inf")
    best_epoch    = 0

    print(f"[4/5] Training for {epochs} epochs …\n")
    bar_width = 30

    for epoch in range(1, epochs + 1):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        train_loss_fresh = 0.0
        train_loss_days  = 0.0
        n_batches = 0

        for x, y_fresh, y_days in train_loader:
            x, y_fresh, y_days = x.to(device), y_fresh.to(device), y_days.to(device)
            optimiser.zero_grad()

            pred_fresh, pred_days = model(x)
            loss_fresh = mse_loss(pred_fresh, y_fresh)
            loss_days  = mse_loss(pred_days / max_days, y_days / max_days)  # normalised
            loss       = loss_fresh + 0.5 * loss_days

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()

            train_loss_fresh += loss_fresh.item()
            train_loss_days  += loss_days.item()
            n_batches        += 1

        scheduler.step()
        avg_train_fresh = train_loss_fresh / max(1, n_batches)
        avg_train_days  = train_loss_days  / max(1, n_batches)

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        val_loss_fresh = 0.0
        val_loss_days  = 0.0
        val_mae_fresh  = 0.0   # mean absolute error in freshness %
        val_mae_days   = 0.0   # mean absolute error in days
        n_val_batches  = 0

        with torch.no_grad():
            for x, y_fresh, y_days in val_loader:
                x, y_fresh, y_days = x.to(device), y_fresh.to(device), y_days.to(device)
                pred_fresh, pred_days = model(x)

                val_loss_fresh += mse_loss(pred_fresh, y_fresh).item()
                val_loss_days  += mse_loss(pred_days / max_days, y_days / max_days).item()
                val_mae_fresh  += torch.mean(torch.abs(pred_fresh - y_fresh)).item()
                val_mae_days   += torch.mean(torch.abs(pred_days  - y_days)).item()
                n_val_batches  += 1

        avg_val_fresh = val_loss_fresh / max(1, n_val_batches)
        avg_val_days  = val_loss_days  / max(1, n_val_batches)
        avg_val_loss  = avg_val_fresh + 0.5 * avg_val_days
        avg_mae_fresh = (val_mae_fresh / max(1, n_val_batches)) * 100.0  # back to %
        avg_mae_days  = val_mae_days   / max(1, n_val_batches)

        # Save best checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch    = epoch
            MODELS_DIR.mkdir(exist_ok=True, parents=True)
            torch.save(model.state_dict(), str(CHECKPOINT_PATH))

        history.append({
            "epoch": epoch,
            "train_fresh": avg_train_fresh,
            "train_days":  avg_train_days,
            "val_fresh":   avg_val_fresh,
            "val_days":    avg_val_days,
            "mae_fresh":   avg_mae_fresh,
            "mae_days":    avg_mae_days,
        })

        # Progress bar
        filled = int(bar_width * epoch / epochs)
        bar    = "#" * filled + "-" * (bar_width - filled)
        star   = " *BEST*" if epoch == best_epoch else ""
        print(
            f"  Epoch {epoch:>4}/{epochs}  [{bar}]  "
            f"val_fresh_loss={avg_val_fresh:.4f}  "
            f"mae_fresh={avg_mae_fresh:.1f}%  "
            f"mae_days={avg_mae_days:.2f}d{star}",
            end="\r" if epoch < epochs else "\n"
        )

    # ── Report ────────────────────────────────────────────────────────────────
    print(f"\n[5/5] Training complete.")
    print(f"      Best checkpoint: epoch {best_epoch}  val_loss={best_val_loss:.5f}")
    print(f"      Saved: {CHECKPOINT_PATH}\n")

    report_lines = [
        "Edge-Vision Decay Regressor - Training Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"CSV:       {csv_path}  ({len(rows)} images)",
        f"Epochs:    {epochs}   LR: {lr}   Aug: {aug_factor}x   Val split: {val_split:.0%}",
        f"Device:    {device}",
        f"Best:      epoch {best_epoch}  val_loss={best_val_loss:.5f}",
        "",
        f"{'Epoch':>6}  {'TrFresh':>9}  {'TrDays':>8}  {'ValFresh':>9}  {'ValDays':>8}  {'MAE Fresh':>10}  {'MAE Days':>10}",
        "-" * 75,
    ]
    for h in history:
        report_lines.append(
            f"{h['epoch']:>6}  {h['train_fresh']:>9.5f}  {h['train_days']:>8.5f}  "
            f"{h['val_fresh']:>9.5f}  {h['val_days']:>8.5f}  "
            f"{h['mae_fresh']:>9.1f}%  {h['mae_days']:>9.2f}d"
        )

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"      Full loss log: {REPORT_PATH}\n")

    # Human-readable final accuracy
    last = history[-1]
    print(f"  Final val freshness MAE : {last['mae_fresh']:5.1f}%")
    print(f"  Final val days-left MAE : {last['mae_days']:5.2f} days")
    print()
    print("  The model is now loaded automatically when the server starts.")
    print("  Restart the server to apply it:  run_server.bat\n")


# ── CLI entry-point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the Edge-Vision PyTorch Decay Regressor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            QUICKSTART
            ----------
            1. Add images to sample_images/  (fresh and rotten for each fruit)
            2. Add rows to data/training_labels.csv:
                   image_path,category,freshness,days_left
                   sample_images/fresh_tomato.jpg,Tomato,95.0,8
                   sample_images/rotten_tomato.jpg,Tomato,8.0,1
            3. Run:
                   .venv\\Scripts\\python train_decay_model.py

            TIP: The more labelled images you add, the better the model gets.
                 Aim for 10-30 images per category at various decay stages.
        """)
    )
    parser.add_argument("--csv",    default="data/training_labels.csv", help="Path to training labels CSV")
    parser.add_argument("--epochs", type=int,   default=100,  help="Number of training epochs (default: 100)")
    parser.add_argument("--lr",     type=float, default=0.001, help="Learning rate (default: 0.001)")
    parser.add_argument("--val",    type=float, default=0.2,  help="Validation split fraction (default: 0.2)")
    parser.add_argument("--aug",    type=int,   default=5,    help="Augmentation multiplier per image (default: 5)")
    args = parser.parse_args()

    train(
        csv_path  = args.csv,
        epochs    = args.epochs,
        lr        = args.lr,
        val_split = args.val,
        aug_factor= args.aug,
    )
