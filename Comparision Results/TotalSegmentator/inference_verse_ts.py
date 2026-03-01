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

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

TS_FAST_MODE = False

TEST_RAW_DIR = "data/raw/dataset-03test/rawdata"
TEST_DERIV_DIR = "data/raw/dataset-03test/derivatives"
RESULTS_DIR = "results/totalsegmentator/verse"
os.makedirs(RESULTS_DIR, exist_ok=True)


def keep_largest_blob(mask):
    labeled_mask, num_features = label(mask)

    if num_features == 0:
        return mask

    component_sizes = [np.sum(labeled_mask == i) for i in range(1, num_features + 1)]

    if not component_sizes:
        return mask

    largest_label = np.argmax(component_sizes) + 1

    return (labeled_mask == largest_label).astype(np.float32)


def compute_metrics(pred, gt):
    pred_f = pred.flatten()
    gt_f = gt.flatten()

    intersection = np.sum(pred_f * gt_f)
    sum_pred = np.sum(pred_f)
    sum_gt = np.sum(gt_f)

    dice = (2.0 * intersection) / (sum_pred + sum_gt + 1e-6)

    union = sum_pred + sum_gt - intersection
    iou = intersection / (union + 1e-6)

    recall = intersection / (sum_gt + 1e-6)

    precision = intersection / (sum_pred + 1e-6)

    return {"Dice": dice, "IoU": iou, "Recall": recall, "Precision": precision}


def save_instance_visual(ct_vol, pred_mask, subject_id, output_dir):
    mid_idx = ct_vol.shape[0] // 2
    ct_slice = ct_vol[mid_idx, :, :].T
    mask_slice = pred_mask[mid_idx, :, :].T

    fig, ax = plt.subplots(figsize=(10, 20))
    ax.imshow(ct_slice, cmap="gray", origin="lower")

    masked_mask = np.ma.masked_where(mask_slice == 0, mask_slice)
    ax.imshow(masked_mask, cmap="winter", alpha=0.4, origin="lower")

    ax.set_title(f"Localization (TotalSeg): {subject_id}", color="white", fontsize=14)
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"{subject_id}_localization.png"), facecolor="black"
    )
    plt.close()


def predict_totalsegmentator(ct_path, target_shape):
    with tempfile.TemporaryDirectory() as tmp_out_dir:
        try:
            totalsegmentator(
                ct_path,
                tmp_out_dir,
                task="total",
                fast=TS_FAST_MODE,
                ml=False,
                device=DEVICE,
                quiet=True,
            )
        except Exception as e:
            print(f"\n[Error] TotalSegmentator failed for {ct_path}: {e}")
            return np.zeros(target_shape)

        vert_files = glob.glob(os.path.join(tmp_out_dir, "vertebrae_*.nii.gz"))

        if not vert_files:
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

        if binary_pred.shape != target_shape:
            print(
                f"\n[Warning] Shape mismatch: Pred {binary_pred.shape} vs GT {target_shape}"
            )

        return binary_pred


def run_evaluation():
    ct_files = sorted(
        glob.glob(os.path.join(TEST_RAW_DIR, "**/*ct.nii.gz"), recursive=True)
    )
    detailed_results = []

    print(f"--- Evaluating {len(ct_files)} Test Volumes (TotalSegmentator) ---")
    print(f"Mode: {'FAST (3mm)' if TS_FAST_MODE else 'NORMAL (1.5mm)'}")
    print(
        f"{'Subject ID':<15} | {'Dice':<8} | {'IoU':<8} | {'Recall':<8} | {'Prec':<8}"
    )
    print("-" * 65)

    for ct_path in tqdm(ct_files, desc="Overall Progress"):
        subject_id = os.path.basename(ct_path).split("_")[0]
        mask_pattern = os.path.join(TEST_DERIV_DIR, subject_id, "*_seg-vert_msk.nii.gz")
        mask_files = glob.glob(mask_pattern)

        if not mask_files:
            continue

        gt_nii = nib.as_closest_canonical(nib.load(mask_files[0]))
        gt_vol = (gt_nii.get_fdata() > 0).astype(np.float32)

        ct_nii = nib.as_closest_canonical(nib.load(ct_path))
        ct_vol = np.clip(ct_nii.get_fdata(), -1000, 2000)
        ct_vol = (ct_vol + 1000) / 3000

        pred_vol = predict_totalsegmentator(ct_path, gt_vol.shape)

        pred_vol = keep_largest_blob(pred_vol)

        metrics = compute_metrics(pred_vol, gt_vol)
        metrics["ID"] = subject_id
        detailed_results.append(metrics)

        save_instance_visual(ct_vol, pred_vol, subject_id, RESULTS_DIR)

        tqdm.write(
            f"{subject_id:<15} | {metrics['Dice']:<8.4f} | {metrics['IoU']:<8.4f} | "
            f"{metrics['Recall']:<8.4f} | {metrics['Precision']:<8.4f}"
        )

    if not detailed_results:
        print("No matching ground truth files found.")
        return

    df = pd.DataFrame(detailed_results)

    print("\n" + "=" * 55)
    print("FINAL TEST SET PERFORMANCE SUMMARY (TotalSegmentator)")
    print("-" * 55)
    print(f"Mean Dice      : {df['Dice'].mean():.4f} ± {df['Dice'].std():.4f}")
    print(f"Mean IoU       : {df['IoU'].mean():.4f} ± {df['IoU'].std():.4f}")
    print(f"Mean Recall    : {df['Recall'].mean():.4f} ± {df['Recall'].std():.4f}")
    print(
        f"Mean Precision : {df['Precision'].mean():.4f} ± {df['Precision'].std():.4f}"
    )

    best_case = df.loc[df["Dice"].idxmax()]
    worst_case = df.loc[df["Dice"].idxmin()]

    print("-" * 55)
    print(f"Best Dice      : {best_case['Dice']:.4f} ({best_case['ID']})")
    print(f"Worst Dice     : {worst_case['Dice']:.4f} ({worst_case['ID']})")
    print("=" * 55)

    df.to_csv("test_metrics_full_totalsegmentator.csv", index=False)
    print("Results saved to test_metrics_full_totalsegmentator.csv")


if __name__ == "__main__":
    run_evaluation()
