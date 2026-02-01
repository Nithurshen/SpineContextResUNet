import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import nibabel as nib
import os
import glob
import gc
import time
from tqdm import tqdm
from scipy.ndimage import label


from src.model import SpineResUNet

# --- CONFIGURATION ---
DEVICE = "cpu"  # Force CPU as requested

MODEL_PATH = "models/best_model.pth"

# Path Configuration (Adjust to your local paths)
TEST_RAW_DIR = "data/raw/dataset-03test/rawdata"
TEST_DERIV_DIR = "data/raw/dataset-03test/derivatives"
RESULTS_DIR = "results/verse2020_fast"
CSV_PATH = os.path.join(RESULTS_DIR, "full_fast_metrics.csv")

# Performance Settings
BATCH_SIZE = 4            # Process 4 chunks at once (Good balance for CPU)
PATCH_SIZE = (128, 128, 64) 
STRIDE = (128, 128, 64)   # Zero Overlap = Fastest way to process full scan

os.makedirs(RESULTS_DIR, exist_ok=True)

def keep_largest_blob(mask):
    """Post-processing: Keeps only the largest connected component (spine)."""
    labeled_mask, num_features = label(mask)
    if num_features == 0: return mask
    counts = np.bincount(labeled_mask.ravel())
    counts[0] = 0
    largest_label = counts.argmax()
    return (labeled_mask == largest_label).astype(np.uint8)

def compute_metrics(pred, gt):
    """Computes Dice and Recall efficiently."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    
    intersection = np.logical_and(pred, gt).sum()
    pred_sum = pred.sum()
    gt_sum = gt.sum()
    
    dice = (2.0 * intersection) / (pred_sum + gt_sum + 1e-6)
    recall = (intersection) / (gt_sum + 1e-6)
    return dice, recall

def predict_entire_scan(model, vol_path):
    """
    Processes the FULL volume using a sliding window with zero overlap.
    Memory Efficient: Loads and predicts in streams (no huge lists).
    """
    # 1. Load Volume
    img_nii = nib.as_closest_canonical(nib.load(vol_path))
    vol_data = img_nii.get_fdata().astype(np.float32)
    
    # Normalize
    vol_data = np.clip(vol_data, -1000, 2000)
    vol_data = (vol_data + 1000) / 3000
    
    d, h, w = vol_data.shape
    pd, ph, pw = PATCH_SIZE
    sd, sh, sw = STRIDE

    # 2. Allocate Output Mask (UINT8 = 1 byte/voxel for memory efficiency)
    final_mask = np.zeros((d, h, w), dtype=np.uint8)

    # 3. Generate Coordinates covering the ENTIRE scan
    coords = []
    for z in range(0, d, sd):
        for y in range(0, h, sh):
            for x in range(0, w, sw):
                coords.append((z, y, x))

    # 4. Batched Inference Loop
    model.eval()
    with torch.no_grad():
        for i in range(0, len(coords), BATCH_SIZE):
            batch_coords = coords[i : i + BATCH_SIZE]
            batch_tensors = []
            valid_infos = []

            # Prepare Batch
            for (z, y, x) in batch_coords:
                # Extract patch
                sub_vol = vol_data[z : z + pd, y : y + ph, x : x + pw]
                cd, ch, cw = sub_vol.shape

                # Pad if patch is at the edge and smaller than model input
                if (cd, ch, cw) != PATCH_SIZE:
                    temp = np.zeros(PATCH_SIZE, dtype=np.float32)
                    temp[:cd, :ch, :cw] = sub_vol
                    sub_vol = temp

                # Convert to Tensor (N, C, D, H, W)
                tensor = torch.from_numpy(sub_vol).unsqueeze(0).unsqueeze(0)
                batch_tensors.append(tensor)
                valid_infos.append((z, y, x, cd, ch, cw))

            if not batch_tensors: continue
            
            # Run Model
            input_batch = torch.cat(batch_tensors).to(DEVICE)
            preds = model(input_batch)
            
            # Binarize immediately to save memory
            preds_bin = (preds > 0.5).byte().cpu().numpy()

            # Paste Result
            for idx, (z, y, x, cd, ch, cw) in enumerate(valid_infos):
                mask_slice = preds_bin[idx, 0, :cd, :ch, :cw]
                # Direct assignment (Zero Overlap means no blending needed)
                final_mask[z:z+cd, y:y+ch, x:x+cw] = mask_slice

            # Explicit Cleanup
            del input_batch, preds, preds_bin, batch_tensors

    return final_mask, img_nii.affine, img_nii.header

def run_evaluation():
    print(f"--- Running Full Scan Inference (Fast Mode) on {DEVICE} ---")
    
    model = SpineResUNet().to(DEVICE)
    if os.path.exists(MODEL_PATH):
        # Load weights (map_location handles CPU/GPU automatically)
        state = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state)
        print(f"✅ Model loaded from {MODEL_PATH}")
    else:
        print(f"❌ Model not found at {MODEL_PATH}")
        return

    # Find Files
    vol_files = sorted(glob.glob(os.path.join(TEST_RAW_DIR, "**", "*ct.nii.gz"), recursive=True))
    
    if not vol_files:
        print(f"❌ No files found in {TEST_RAW_DIR}")
        return

    results = []
    print(f"Processing {len(vol_files)} volumes...")
    print(f"{'ID':<10} | {'Time(s)':<8} | {'Dice':<6} | {'Recall':<6}")
    print("-" * 40)

    for vol_path in tqdm(vol_files):
        subject_id = os.path.basename(vol_path).split("_")[0]
        
        try:
            start_t = time.time()
            
            # 1. Predict
            pred_mask, affine, header = predict_entire_scan(model, vol_path)
            
            # 2. Post-Process (Largest Blob)
            pred_mask = keep_largest_blob(pred_mask)
            
            duration = time.time() - start_t
            
            # 3. Find GT and Calculate Metrics
            # Fallback search for GT in case file naming varies slightly
            gt_candidates = glob.glob(os.path.join(TEST_DERIV_DIR, "**", f"{subject_id}*_seg-vert_msk.nii.gz"), recursive=True)
            
            dice, recall = 0.0, 0.0
            status = ""
            
            if gt_candidates:
                gt_nii = nib.load(gt_candidates[0])
                gt_data = (gt_nii.get_fdata() > 0).astype(np.uint8)
                dice, recall = compute_metrics(pred_mask, gt_data)
                status = "✅"
            else:
                status = "⚠️ (No GT)"

            print(f"{subject_id:<10} | {duration:<8.2f} | {dice:<6.3f} | {recall:<6.3f} | {status}")
            
            results.append({
                "ID": subject_id, "Time": duration, 
                "Dice": dice, "Recall": recall
            })

            # 4. Save Result
            save_path = os.path.join(RESULTS_DIR, f"{subject_id}_pred.nii.gz")
            nib.save(nib.Nifti1Image(pred_mask, affine, header), save_path)

            # Cleanup
            del pred_mask, affine, header
            gc.collect()

        except Exception as e:
            print(f"❌ Error on {subject_id}: {e}")
            gc.collect()

    # Final Summary
    if results:
        df = pd.DataFrame(results)
        df.to_csv(CSV_PATH, index=False)
        print("\n=== SUMMARY ===")
        print(f"Avg Time: {df['Time'].mean():.2f} s")
        if df['Dice'].sum() > 0:
            print(f"Avg Dice: {df[df['Dice']>0]['Dice'].mean():.4f}")
            print(f"Avg Recall: {df[df['Recall']>0]['Recall'].mean():.4f}")

if __name__ == "__main__":
    run_evaluation()