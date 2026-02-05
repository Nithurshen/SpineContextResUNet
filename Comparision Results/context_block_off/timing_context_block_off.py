import os
# Must be set before torch is imported
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch
import nibabel as nib
import numpy as np
import time
import sys

# Update path to find the ablation model
sys.path.append('Comparision Results/context_block_off')
from model import SpineResUNet_cotext_off

# --- CONFIGURATION ---
PATCH_SIZE = (128, 128, 64)
OVERLAP = 0.25
VOL_PATH = "data/raw/dataset-03test/rawdata/sub-verse714/sub-verse714_dir-iso_ct.nii.gz"
MSK_PATH = "data/raw/dataset-03test/derivatives/sub-verse714/sub-verse714_dir-iso_seg-vert_msk.nii.gz"
MODEL_PATH = "models/best_model_context_off.pth"

def compute_dice(pred, gt):
    intersection = np.sum(pred * gt)
    return (2.0 * intersection) / (np.sum(pred) + np.sum(gt) + 1e-6)

def predict_sliding_window(model, vol, device):
    d, h, w = vol.shape
    pd, ph, pw = PATCH_SIZE
    prob_map = np.zeros(vol.shape, dtype=np.float32)
    weight_map = np.zeros(vol.shape, dtype=np.float32)
    
    stride_d, stride_h, stride_w = [int(p * (1 - OVERLAP)) for p in PATCH_SIZE]
    
    z_steps = sorted(list(set(list(range(0, d - pd + stride_d, stride_d)) + [max(0, d - pd)])))
    y_steps = sorted(list(set(list(range(0, h - ph + stride_h, stride_h)) + [max(0, h - ph)])))
    x_steps = sorted(list(set(list(range(0, w - pw + stride_w, stride_w)) + [max(0, w - pw)])))

    model.to(device)
    model.eval()
    
    with torch.no_grad():
        for z in z_steps:
            for y in y_steps:
                for x in x_steps:
                    patch_actual = vol[z:z+pd, y:y+ph, x:x+pw]
                    curr_d, curr_h, curr_w = patch_actual.shape
                    
                    if (curr_d, curr_h, curr_w) != PATCH_SIZE:
                        patch_input = np.pad(patch_actual, (
                            (0, pd - curr_d), (0, ph - curr_h), (0, pw - curr_w)
                        ))
                    else:
                        patch_input = patch_actual

                    patch_t = torch.from_numpy(patch_input).float().unsqueeze(0).unsqueeze(0).to(device)
                    output = model(patch_t)
                    
                    pred_patch = output.squeeze().cpu().numpy()
                    prob_map[z:z+curr_d, y:y+curr_h, x:x+curr_w] += pred_patch[:curr_d, :curr_h, :curr_w]
                    weight_map[z:z+curr_d, y:y+curr_h, x:x+curr_w] += 1.0

    return prob_map / np.maximum(weight_map, 1e-6)

def run_test():
    # Load Data
    vol_nii = nib.load(VOL_PATH)
    vol_data = np.clip(vol_nii.get_fdata(), -1000, 2000)
    vol_data = (vol_data + 1000) / 3000
    gt_data = (nib.load(MSK_PATH).get_fdata() > 0).astype(np.float32)

    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")

    for dev_type in devices:
        device = torch.device(dev_type)
        print(f"\n--- Testing SpineResUNet_cotext_off on {dev_type.upper()} ---")
        
        # Initialize model with base_filters=16 as per your architecture
        model = SpineResUNet_cotext_off(in_channels=1, out_channels=1, base_filters=16)
        
        # Load weights and move model to device
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        model.to(device)

        if dev_type == "mps":
            torch.mps.empty_cache()
            print("Warming up MPS device...")
            dummy = torch.randn(1, 1, 64, 64, 64).to(device)
            _ = model(dummy)

        start_time = time.time()
        pred_prob = predict_sliding_window(model, vol_data, device)
        elapsed = time.time() - start_time
        
        dice = compute_dice((pred_prob > 0.5).astype(np.float32), gt_data)
        
        print(f"Results for {dev_type.upper()}:")
        print(f"  Time Taken: {elapsed:.2f} seconds")
        print(f"  Dice Score: {dice:.4f}")

if __name__ == "__main__":
    run_test()