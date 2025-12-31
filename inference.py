import torch
import numpy as np
import pandas as pd
import nibabel as nib
import os
import glob
from tqdm import tqdm
from src.model import SpineResUNet
import matplotlib.pyplot as plt

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)
MODEL_PATH = "models/best_model.pth"
PATCH_SIZE = (128, 128, 64)
OVERLAP = 0.5
TEST_RAW_DIR = "data/raw/dataset-03test/rawdata"
TEST_DERIV_DIR = "data/raw/dataset-03test/derivatives"
RESULTS_DIR = "results/test"
os.makedirs(RESULTS_DIR, exist_ok=True)


def compute_dice(pred, gt):
    """Simple Dice calculation"""
    intersection = np.sum(pred * gt)
    dice = (2.0 * intersection) / (np.sum(pred) + np.sum(gt) + 1e-6)
    return dice


def save_instance_visual(ct_vol, pred_mask, subject_id, output_dir):
    """Saves a mid-sagittal 2D overlay of the localization result."""
    mid_idx = ct_vol.shape[0] // 2
    ct_slice = ct_vol[mid_idx, :, :].T
    mask_slice = pred_mask[mid_idx, :, :].T

    fig, ax = plt.subplots(figsize=(10, 20))
    ax.imshow(ct_slice, cmap="gray", origin="lower")

    # Blue mask overlay for binary localization
    masked_mask = np.ma.masked_where(mask_slice == 0, mask_slice)
    ax.imshow(masked_mask, cmap="winter", alpha=0.4, origin="lower")

    ax.set_title(f"Localization: {subject_id}", color="white", fontsize=14)
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"{subject_id}_localization.png"), facecolor="black"
    )
    plt.close()


def predict_sliding_window(model, vol):
    """Memory-efficient sliding window using CPU for accumulation"""
    d, h, w = vol.shape
    pd, ph, pw = PATCH_SIZE
    prob_map = torch.zeros(vol.shape).cpu()
    count_map = torch.zeros(vol.shape).cpu()

    stride_d, stride_h, stride_w = [int(p * (1 - OVERLAP)) for p in PATCH_SIZE]
    vol_t = torch.from_numpy(vol).float()

    z_steps = sorted(
        list(set(list(range(0, d - pd + stride_d, stride_d)) + [max(0, d - pd)]))
    )
    y_steps = sorted(
        list(set(list(range(0, h - ph + stride_h, stride_h)) + [max(0, h - ph)]))
    )
    x_steps = sorted(
        list(set(list(range(0, w - pw + stride_w, stride_w)) + [max(0, w - pw)]))
    )

    with torch.no_grad():
        for z in z_steps:
            for y in y_steps:
                for x in x_steps:
                    patch = (
                        vol_t[z : z + pd, y : y + ph, x : x + pw]
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .to(DEVICE)
                    )
                    pred = model(patch)
                    prob_map[z : z + pd, y : y + ph, x : x + pw] += pred.squeeze().cpu()
                    count_map[z : z + pd, y : y + ph, x : x + pw] += 1.0
                    if DEVICE == "mps":
                        torch.mps.empty_cache()

    return ((prob_map / count_map) > 0.5).float().numpy()


def run_evaluation():
    model = SpineResUNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    ct_files = sorted(
        glob.glob(os.path.join(TEST_RAW_DIR, "**/*ct.nii.gz"), recursive=True)
    )
    detailed_results = []

    print(f"--- Evaluating {len(ct_files)} Test Volumes (Dice) ---")
    print(f"{'Subject ID':<15} | {'Dice Score':<12}")
    print("-" * 35)

    for ct_path in tqdm(ct_files, desc="Overall Progress"):
        subject_id = os.path.basename(ct_path).split("_")[0]
        mask_pattern = os.path.join(TEST_DERIV_DIR, subject_id, "*_seg-vert_msk.nii.gz")
        mask_files = glob.glob(mask_pattern)

        if not mask_files:
            continue

        ct_nii = nib.as_closest_canonical(nib.load(ct_path))
        gt_nii = nib.as_closest_canonical(nib.load(mask_files[0]))

        ct_vol = np.clip(ct_nii.get_fdata(), -1000, 2000)
        ct_vol = (ct_vol + 1000) / 3000
        gt_vol = (gt_nii.get_fdata() > 0).astype(np.float32)

        pred_vol = predict_sliding_window(model, ct_vol)

        dice = compute_dice(pred_vol, gt_vol)

        detailed_results.append({"ID": subject_id, "Dice": dice})
        save_instance_visual(ct_vol, pred_vol, subject_id, RESULTS_DIR)
        print(f"{subject_id:<15} | {dice:<12.4f}")

    dices = [r["Dice"] for r in detailed_results]

    print("\n" + "=" * 40)
    print("FINAL TEST SET PERFORMANCE SUMMARY")
    print(f"Mean Dice Score : {np.mean(dices):.4f} ± {np.std(dices):.4f}")
    print(
        f"Best Case       : {np.max(dices):.4f} ({detailed_results[np.argmax(dices)]['ID']})"
    )
    print(
        f"Worst Case      : {np.min(dices):.4f} ({detailed_results[np.argmin(dices)]['ID']})"
    )
    print("=" * 40)

    df = pd.DataFrame(detailed_results)
    df.to_csv("test_metrics_dice.csv", index=False)
    print("Results saved to test_metrics_dice.csv")


if __name__ == "__main__":
    run_evaluation()
