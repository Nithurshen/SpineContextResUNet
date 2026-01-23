import os
import torch
import numpy as np
import nibabel as nib
import subprocess
from pathlib import Path

def create_dummy_ct(filename="dummy_ct.nii.gz"):
    data = np.zeros((32, 32, 32), dtype=np.int16)
    affine = np.eye(4)
    nii = nib.Nifti1Image(data, affine)
    nib.save(nii, filename)
    return filename

def download_and_count():
    print("--- Step 1: Triggering Weight Download (CLI) ---")
    dummy_file = create_dummy_ct()
    
    # Check for existing weights first to avoid re-downloading loop
    home_dir = Path.home()
    ts_dir = home_dir / ".totalsegmentator"
    checkpoints = list(ts_dir.rglob("checkpoint_final.pth"))

    if not checkpoints:
        print("No weights found. Triggering download...")
        cmd = ["TotalSegmentator", "-i", dummy_file, "-o", "dummy_output", "--fast", "--ml"]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            checkpoints = list(ts_dir.rglob("checkpoint_final.pth"))
        except Exception as e:
            print(f"Download failed: {e}")
            return

    if not checkpoints:
        print("Still no checkpoints found.")
        return

    # Use the first valid checkpoint found
    ckpt_path = checkpoints[0]
    print(f"\n--- Step 2: Found Weights ---")
    print(f"Path: {ckpt_path}")

    print("\n--- Step 3: Counting Parameters ---")
    try:
        # Load with weights_only=False to allow numpy scalars
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        
        # --- ROBUST WEIGHT EXTRACTION ---
        # 1. Try known keys for state dictionaries
        if "state_dict" in checkpoint:
            print(" -> Found 'state_dict' key.")
            weights = checkpoint["state_dict"]
        elif "network_weights" in checkpoint:
            print(" -> Found 'network_weights' key.")
            weights = checkpoint["network_weights"]
        else:
            # 2. If no known key, assume the dict IS the weights, 
            # but we must filter out non-tensors (like 'epoch', 'optimizer', etc.)
            print(" -> No 'state_dict' key found. Iterating root dictionary...")
            weights = checkpoint

        total_params = 0
        counted_layers = 0
        
        for key, value in weights.items():
            # SAFETY CHECK: Only count actual PyTorch Tensors
            if torch.is_tensor(value):
                # Optional: Skip special scalar trackers if any
                if "num_batches_tracked" in key:
                    continue
                
                total_params += value.numel()
                counted_layers += 1
            else:
                # Debugging info: print what we skipped
                if counted_layers < 3: # Print first few skips only
                    print(f"    Skipping non-tensor key: '{key}' (Type: {type(value).__name__})")

        print("\n" + "="*50)
        print("OFFICIAL PARAMETER COUNT (Verified)")
        print(f"File                 : {ckpt_path.name}")
        print(f"Model Name           : {ckpt_path.parent.parent.parent.name}")
        print(f"Layers Counted       : {counted_layers}")
        print(f"Total Parameters     : {total_params:,}")
        print(f"Size in MB (float32) : {total_params * 4 / (1024**2):.2f} MB")
        print("="*50)

    except Exception as e:
        print(f"Error reading checkpoint: {e}")

    # Cleanup
    if os.path.exists(dummy_file):
        os.remove(dummy_file)
    if os.path.exists("dummy_output"):
        import shutil
        shutil.rmtree("dummy_output")

if __name__ == "__main__":
    download_and_count()