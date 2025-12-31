import re
import matplotlib.pyplot as plt

LOG_FILE = 'logs/training_log.txt'
TARGET_DICE = 0.85

def parse_log(filename):
    """Parses the training log to extract metrics."""
    epochs = []
    train_losses = []
    val_losses = []
    val_dices = []

    epoch_pattern = re.compile(r"Epoch (\d+) Results:")
    train_loss_pattern = re.compile(r"Train Loss\s*:\s*([\d.]+)")
    val_loss_pattern = re.compile(r"Val Loss\s*:\s*([\d.]+)")
    val_dice_pattern = re.compile(r"Val Dice\s*:\s*([\d.]+)")

    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            epoch_match = epoch_pattern.search(line)
            if epoch_match:
                epochs.append(int(epoch_match.group(1)))
            
            train_match = train_loss_pattern.search(line)
            if train_match:
                train_losses.append(float(train_match.group(1)))
            
            val_match = val_loss_pattern.search(line)
            if val_match:
                val_losses.append(float(val_match.group(1)))
                
            dice_match = val_dice_pattern.search(line)
            if dice_match:
                val_dices.append(float(dice_match.group(1)))
                
        min_len = min(len(epochs), len(train_losses), len(val_losses), len(val_dices))
        return {
            'epochs': epochs[:min_len],
            'train_loss': train_losses[:min_len],
            'val_loss': val_losses[:min_len],
            'val_dice': val_dices[:min_len]
        }

    except FileNotFoundError:
        print(f"Error: Could not find file '{filename}'.")
        return None

def plot_metrics(data):
    """Plots the metrics side by side."""
    if not data:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    ax1.plot(data['epochs'], data['train_loss'], label='Train Loss', color='#1f77b4', linewidth=2)
    ax1.plot(data['epochs'], data['val_loss'], label='Validation Loss', color='#d62728', linewidth=2, linestyle='--')
    
    ax1.set_title('Training vs. Validation Loss', fontsize=14)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2.plot(data['epochs'], data['val_dice'], label='Validation Dice', color='#2ca02c', linewidth=2)
    
    ax2.axhline(y=TARGET_DICE, color='gray', linestyle=':', linewidth=1.5, label=f'Target ({TARGET_DICE})')
    
    ax2.set_title('Validation Dice Score', fontsize=14)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Dice Coefficient', fontsize=12)
    ax2.legend(fontsize=10, loc='lower right')
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    
    save_path = 'training_metrics.png'
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    metrics = parse_log(LOG_FILE)
    if metrics:
        print(f"Found {len(metrics['epochs'])} epochs. Plotting...")
        plot_metrics(metrics)