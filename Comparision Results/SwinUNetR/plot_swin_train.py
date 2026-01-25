import matplotlib.pyplot as plt
import re

# File path
log_file_path = './swin_train_plot.txt'

# Lists to store data
epochs = []
train_losses = []
val_losses = []
val_dice_scores = []

# Regex patterns to find specific lines
# Matches "Epoch 1 Results"
epoch_pattern = re.compile(r"Epoch\s+(\d+)\s+Results")
# Matches "Train Loss : 0.1234"
train_loss_pattern = re.compile(r"Train Loss\s*:\s*([\d.]+)")
# Matches "Val Loss : 0.1234"
val_loss_pattern = re.compile(r"Val Loss\s*:\s*([\d.]+)")
# Matches "Val Dice : 0.1234"
val_dice_pattern = re.compile(r"Val Dice\s*:\s*([\d.]+)")

try:
    with open(log_file_path, 'r') as f:
        lines = f.readlines()
        
        # Temporary variables to hold current epoch data
        current_epoch = None
        
        for line in lines:
            line = line.strip()
            
            # check for Epoch number
            epoch_match = epoch_pattern.search(line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                epochs.append(current_epoch)
                continue

            # Check for Train Loss
            t_loss_match = train_loss_pattern.search(line)
            if t_loss_match:
                train_losses.append(float(t_loss_match.group(1)))
                continue

            # Check for Val Loss
            v_loss_match = val_loss_pattern.search(line)
            if v_loss_match:
                val_losses.append(float(v_loss_match.group(1)))
                continue

            # Check for Val Dice
            dice_match = val_dice_pattern.search(line)
            if dice_match:
                val_dice_scores.append(float(dice_match.group(1)))
                continue

    # Verification: Ensure all lists are the same length
    min_len = min(len(epochs), len(train_losses), len(val_losses), len(val_dice_scores))
    epochs = epochs[:min_len]
    train_losses = train_losses[:min_len]
    val_losses = val_losses[:min_len]
    val_dice_scores = val_dice_scores[:min_len]

    # --- Plotting ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Graph 1: Superimposed Losses
    ax1.plot(epochs, train_losses, label='Train Loss', color='blue', linewidth=2)
    ax1.plot(epochs, val_losses, label='Val Loss', color='red', linestyle='--', linewidth=2)
    ax1.set_title('Training vs Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.6)

    # Graph 2: Validation Dice Score
    ax2.plot(epochs, val_dice_scores, label='Val Dice', color='green', linewidth=2)
    ax2.set_title('Validation Dice Score')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Dice Score')
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.6)

    # Add text for the best dice score
    if val_dice_scores:
        max_dice = max(val_dice_scores)
        max_epoch = epochs[val_dice_scores.index(max_dice)]
        ax2.annotate(f'Best: {max_dice:.4f} (Ep {max_epoch})', 
                     xy=(max_epoch, max_dice), 
                     xytext=(max_epoch, max_dice - (max_dice * 0.1)),
                     arrowprops=dict(facecolor='black', shrink=0.05))

    plt.tight_layout()
    plt.show()

    print(f"Successfully plotted data for {len(epochs)} epochs.")

except FileNotFoundError:
    print(f"Error: Could not find file '{log_file_path}'. Please check the path.")
except Exception as e:
    print(f"An error occurred: {e}")