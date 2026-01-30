import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import SpineDataset       # Re-using your existing dataset
from model import SwinUNETR_Nano # Importing the new Swin model
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# --- Configuration ---
BATCH_SIZE = 4
LR = 0.001
EPOCHS = 100  # Keeping 50 for fair comparison
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if torch.backends.mps.is_available():
    DEVICE = "mps"

# Separate folders to avoid overwriting ResUNet results
MODEL_SAVE_DIR = "models"
CHECKPOINT_DIR = "checkpoints_swin"
VIS_DIR = "visualizations_swin"

def dice_coeff(pred, target, smooth=1.0):
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=2).sum(dim=2).sum(dim=2)
    union = pred.sum(dim=2).sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2).sum(dim=2)
    return ((2.0 * intersection + smooth) / (union + smooth)).mean()

def dice_loss(pred, target):
    return 1 - dice_coeff(pred, target)

def save_visualization(epoch, img_tensor, mask_tensor, pred_tensor, save_dir):
    mid_slice = 32
    img_slice = img_tensor[0, 0, :, :, mid_slice].cpu().numpy()
    mask_slice = mask_tensor[0, 0, :, :, mid_slice].cpu().numpy()
    pred_slice = pred_tensor[0, 0, :, :, mid_slice].detach().cpu().numpy()

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(img_slice, cmap="gray")
    ax[0].set_title(f"Input (Epoch {epoch})")
    ax[1].imshow(mask_slice, cmap="gray")
    ax[1].set_title("Ground Truth")
    ax[2].imshow(pred_slice, cmap="gray")
    ax[2].set_title("Swin Prediction")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"epoch_{epoch:03d}.png"))
    plt.close(fig)

def train():
    print(f"--- Starting SwinUNETR Training on {DEVICE} ---")
    
    # 1. Setup Directories
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(VIS_DIR, exist_ok=True)

    # 2. Setup Data
    train_ds = SpineDataset(split="train", transform=True)
    val_ds = SpineDataset(split="val", transform=False)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # Visualization Sample
    vis_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    fixed_img, fixed_mask = next(iter(vis_loader))
    fixed_img, fixed_mask = fixed_img.to(DEVICE), fixed_mask.to(DEVICE)

    # 3. Initialize Swin Model
    print("Initializing SwinUNETR Micro (~1.8M params)...")
    model = SwinUNETR_Nano(in_channels=1, out_channels=1).to(DEVICE)

    # 4. Optimizer & Loss
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=5)
    bce = nn.BCELoss()
    sigmoid = nn.Sigmoid() # SwinUNETR output is logits, unlike your ResUNet which had sigmoid inside

    best_val_loss = float("inf")

    # 5. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        epoch_train_loss = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Train]", leave=True)

        for imgs, masks in loop:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

            # Forward pass
            logits = model(imgs)
            preds = sigmoid(logits) # Apply sigmoid explicitly here

            loss_d = dice_loss(preds, masks)
            loss_b = bce(preds, masks)
            loss = loss_d + loss_b

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            loop.set_postfix(batch_loss=loss.item())

        # Validation
        model.eval()
        epoch_val_loss = 0
        epoch_val_dice = 0

        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                
                logits = model(imgs)
                preds = sigmoid(logits)

                loss_d = dice_loss(preds, masks)
                loss_b = bce(preds, masks)

                epoch_val_loss += (loss_d + loss_b).item()
                epoch_val_dice += dice_coeff(preds, masks).item()

            # Save Visualization
            vis_logits = model(fixed_img)
            vis_pred = sigmoid(vis_logits)
            save_visualization(epoch + 1, fixed_img, fixed_mask, vis_pred, VIS_DIR)

        # Logging
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_val_loss = epoch_val_loss / len(val_loader)
        avg_val_dice = epoch_val_dice / len(val_loader)

        print(f"Epoch {epoch + 1} Results (Swin):")
        print(f"  Train Loss : {avg_train_loss:.4f}")
        print(f"  Val Loss   : {avg_val_loss:.4f}")
        print(f"  Val Dice   : {avg_val_dice:.4f}")
        print("-" * 30)

        # Scheduler
        scheduler.step(avg_val_loss)

        # Save Best Model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, "best_swin_model.pth"))
            print("  --> New Best Swin Model Saved!")
        
        # Periodic Checkpoint
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"swin_epoch_{epoch + 1}.pth"))

if __name__ == "__main__":
    train()