import torch
import numpy as np
import pandas as pd
import nibabel as nib
import os
import glob
import tempfile
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage import label
from totalsegmentator.python_api import totalsegmentator

# --- CONFIGURATION ---
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

# TotalSegmentator settings
TS_FAST_MODE = False  # Set to True for faster (3mm) low-res inference

# CTSpine1K specific paths
TEST_RAW_DIR = "data/raw/CTSpine1k/volumes/test"
TEST_DERIV_DIR = "data/raw/CTSpine1k/labels/test"
RESULTS_DIR = "results/totalsegmentator/ctspine1k"

os.makedirs(RESULTS_DIR, exist_ok=True)


def keep_largest_blob(mask):
    """
    Retains only the largest connected component in a binary mask.
    """
    labeled_mask, num_features = label(mask)
    
    if num_features == 0:
        return mask
    
    # Calculate size of each component (excluding background 0)
    component_sizes = [np.sum(labeled_mask == i) for i in range(1, num_features + 1)]
    
    if not component_sizes:
        return mask

    # Identify the label of the largest component
    largest_label = np.argmax(component_sizes) + 1
    
    # Return binary mask of only the largest component
    return (labeled_mask == largest_label).astype(np.float32)


def compute_metrics(pred, gt):
    """
    Computes Dice, IoU, Recall, and Precision.
    """
    pred_f = pred.flatten()
    gt_f = gt.flatten()

    intersection = np.sum(pred_f * gt_f)
    sum_pred = np.sum(pred_f)
    sum_gt = np.sum(gt_f)
    
    # Dice: 2 * TP / (TP + FP + TP + FN)
    dice = (2.0 * intersection) / (sum_pred + sum_gt + 1e-6)
    
    # IoU: TP / (TP + FP + FN) -> Intersection / Union
    union = sum_pred + sum_gt - intersection
    iou = intersection / (union + 1e-6)
    
    # Recall (Sensitivity): TP / (TP + FN) -> Intersection / GT
    recall = intersection / (sum_gt + 1e-6)
    
    # Precision: TP / (TP + FP) -> Intersection / Prediction
    precision = intersection / (sum_pred + 1e-6)
    
    return {
        "Dice": dice,
        "IoU": iou,
        "Recall": recall,
        "Precision": precision
    }


def save_instance_visual(ct_vol, pred_mask, subject_id, output_dir):
    """Saves a mid-sagittal 2D overlay of the localization result."""
    mid_idx = ct_vol.shape[0] // 2
    ct_slice = ct_vol[mid_idx, :, :].T
    mask_slice = pred_mask[mid_idx, :, :].T

    fig, ax = plt.subplots(figsize=(10, 20))
    ax.imshow(ct_slice, cmap="gray", origin="lower")

    masked_mask = np.ma.masked_where(mask_slice == 0, mask_slice)
    ax.imshow(masked_mask, cmap="winter", alpha=0.4, origin="lower")

    ax.set_title(f"TotalSeg Prediction: {subject_id}", color="white", fontsize=14, backgroundcolor="black")
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"{subject_id}_ts_seg.png"), facecolor="black"
    )
    plt.close()


def predict_totalsegmentator(ct_path, target_shape):
    """
    Runs TotalSegmentator on the input file path.
    Aggregates all 'vertebrae_*' classes into a single binary mask.
    """
    with tempfile.TemporaryDirectory() as tmp_out_dir:
        try:
            # We use task="total" to get all vertebrae
            totalsegmentator(
                ct_path, 
                tmp_out_dir, 
                task="total", 
                fast=TS_FAST_MODE, 
                ml=False, 
                device=DEVICE,
                quiet=True
            )
        except Exception as e:
            print(f"\n[Error] TotalSegmentator failed for {ct_path}: {e}")
            return np.zeros(target_shape)

        # TotalSegmentator outputs individual nifti files for each vertebrae
        # We need to find all vertebrae files and combine them.
        vert_files = glob.glob(os.path.join(tmp_out_dir, "vertebrae_*.nii.gz"))
        
        # If no explicit vertebrae files, sometimes they are just in the main segmentation
        # but usually task="total" splits them. If not found, check for 'spine.nii.gz' if using older versions
        if not vert_files:
             # Fallback: check if 'spine' class exists specifically (rare in 'total' task but possible in others)
             spine_file = os.path.join(tmp_out_dir, "spine.nii.gz")
             if os.path.exists(spine_file):
                 vert_files = [spine_file]
             else:
                print(f"\n[Warning] No vertebrae found for {ct_path}")
                return np.zeros(target_shape)

        combined_mask = None

        for v_file in vert_files:
            nii = nib.as_closest_canonical(nib.load(v_file))
            data = nii.get_fdata()
            
            if combined_mask is None:
                combined_mask = data
            else:
                combined_mask += data
        
        binary_pred = (combined_mask > 0).astype(np.float32)

        # Simple resizing or cropping check would go here if shapes allow it, 
        # but TS usually preserves input header.
        if binary_pred.shape != target_shape:
            print(f"\n[Warning] Shape mismatch: Pred {binary_pred.shape} vs GT {target_shape}")
            # In a real pipeline, you might want to resample here using torch or scipy.zoom
            # For now, we return zeros to avoid crashing metric computation
            return np.zeros(target_shape)
        
        return binary_pred


def run_evaluation():
    # Use the flat file structure typical of CTSpine1K
    ct_files = sorted(glob.glob(os.path.join(TEST_RAW_DIR, "*.nii.gz")))
    
    detailed_results = []

    print(f"--- Evaluating {len(ct_files)} CTSpine1K Volumes (TotalSegmentator) ---")
    print(f"Mode: {'FAST (3mm)' if TS_FAST_MODE else 'NORMAL (1.5mm)'}")
    print(f"{'Subject ID':<20} | {'Dice':<8} | {'IoU':<8} | {'Recall':<8} | {'Prec':<8}")
    print("-" * 70)

    for ct_path in tqdm(ct_files, desc="Processing"):
        file_name = os.path.basename(ct_path)
        subject_id = file_name.replace(".nii.gz", "")
        
        # CTSpine1K naming logic: look for matching file in label directory
        # Adjust pattern if your specific download has different naming (e.g. _seg vs same name)
        # Standard CTSpine1K often has same name or _seg suffix.
        potential_labels = glob.glob(os.path.join(TEST_DERIV_DIR, f"{subject_id}*.nii.gz"))

        if not potential_labels:
            tqdm.write(f"Skip: Ground truth not found for {subject_id}")
            continue
            
        gt_path = potential_labels[0]

        # Load GT
        gt_nii = nib.as_closest_canonical(nib.load(gt_path))
        gt_vol = (gt_nii.get_fdata() > 0).astype(np.float32)
        
        # Load CT (for visual saving)
        ct_nii = nib.as_closest_canonical(nib.load(ct_path))
        ct_vol = np.clip(ct_nii.get_fdata(), -1000, 2000)
        ct_vol = (ct_vol + 1000) / 3000

        # Predict
        pred_vol = predict_totalsegmentator(ct_path, gt_vol.shape)

        # --- POST PROCESSING: Keep Largest Blob ---
        # TotalSegmentator can sometimes find floating bits of bone elsewhere
        pred_vol = keep_largest_blob(pred_vol)

        # Metrics
        metrics = compute_metrics(pred_vol, gt_vol)
        metrics["ID"] = subject_id
        detailed_results.append(metrics)
        
        # Visualization
        save_instance_visual(ct_vol, pred_vol, subject_id, RESULTS_DIR)
        
        tqdm.write(
            f"{subject_id:<20} | {metrics['Dice']:<8.4f} | {metrics['IoU']:<8.4f} | "
            f"{metrics['Recall']:<8.4f} | {metrics['Precision']:<8.4f}"
        )

    if not detailed_results:
        print("No matching ground truth files found/processed.")
        return

    # Create DataFrame
    df = pd.DataFrame(detailed_results)

    print("\n" + "=" * 60)
    print("FINAL CTSpine1K PERFORMANCE SUMMARY (TotalSegmentator)")
    print("-" * 60)
    print(f"Mean Dice      : {df['Dice'].mean():.4f} ± {df['Dice'].std():.4f}")
    print(f"Mean IoU       : {df['IoU'].mean():.4f} ± {df['IoU'].std():.4f}")
    print(f"Mean Recall    : {df['Recall'].mean():.4f} ± {df['Recall'].std():.4f}")
    print(f"Mean Precision : {df['Precision'].mean():.4f} ± {df['Precision'].std():.4f}")
    
    best_case = df.loc[df['Dice'].idxmax()]
    worst_case = df.loc[df['Dice'].idxmin()]
    
    print("-" * 60)
    print(f"Best Dice      : {best_case['Dice']:.4f} ({best_case['ID']})")
    print(f"Worst Dice     : {worst_case['Dice']:.4f} ({worst_case['ID']})")
    print("=" * 60)

    csv_name = "test_metrics_ctspine1k_totalsegmentator.csv"
    df.to_csv(os.path.join(RESULTS_DIR, csv_name), index=False)
    print(f"Results saved to {os.path.join(RESULTS_DIR, csv_name)}")


if __name__ == "__main__":
    run_evaluation()