# SpineContextResUNet: A Computationally Efficient Residual UNet for Spine CT Segmentation

## Overview

**SpineContextResUNet** is a lightweight 3D deep learning architecture designed for rapid spinal localization in Computed Tomography (CT) scans. While state-of-the-art models like Transformers or large-scale ensembles demand substantial GPU resources, our model is engineered for edge deployment and resource-constrained clinical environments.

### Key Features

* **Computationally Efficient:** Features a footprint of only ~1.7M parameters, making it ideal for edge platforms like the Nvidia Jetson Orin Nano.
* **3D Context Modeling:** Integrates a specialized Context Block using parallel multi-dilated convolutions to capture long-range anatomical dependencies without the memory overhead of Self-Attention or the latency of RNNs.
* **High Performance:** Achieves a Dice score of 88.17% on CTSpine1K and 88.13% on VerSe2020 datasets.
* **Hardware Agnostic:** Performs robust inference on standard clerical hardware (Intel Core i5, 8GB RAM) where heavier baselines like TotalSegmentator fail due to memory exhaustion.

---

## ## Architecture

The architecture follows a U-shaped encoder-decoder topology:

* **Backbone:** Built on Residual Blocks (two $3\times3\times3$ convolutions with BN and ReLU) to facilitate gradient flow.
* **Context Block (ASPP):** Positioned at the bottleneck, this module uses four parallel branches with dilation rates $r \in \{1, 2, 4, 8\}$ to aggregate multi-scale context.
* **Loss Function:** A composite $\mathcal{L}_{Total} = \mathcal{L}_{BCE} + \mathcal{L}_{Dice}$ to handle class imbalance and ensure boundary refinement.

---

## Benchmarks

### Segmentation Performance

| Architecture | Parameters | VerSe2020 (Dice) | CTSpine1K (Dice) |
| --- | --- | --- | --- |
| SwinUNETR (Constrained) | 3,746,536 | 0.7387 | 0.7285 |
| 3D U-Net | 1,788,274 | 0.8144 | 0.8132 |
| ResUNet | 1,424,545 | 0.8652 | 0.8644 |
| **SpineContextResUNet** | **1,703,841** | **0.8813** | **0.8817** |

### Inference Latency (Seconds)

| Model | NVidia T4 GPU | Intel Core i5 (8GB RAM) |
| --- | --- | --- |
| 3D U-Net | 51.01s | 348.25s |
| **SpineContextResUNet** | **86.66s** | **792.49s** |
| TotalSegmentator | 127.67s | **Crashed** |

---

### Training & Inference

1. **Preprocessing:** All volumes should be resampled to **1mm³ isotropic resolution**. Intensities are clipped to **[-1000, 2000] HU** and normalized to [-1, 1].
2. **Patch Size:** Training is performed on fixed patches of **$128\times128\times64$**.
3. **Inference:** Uses a sliding-window approach with a 0.5 stride overlap and Gaussian importance weighting for reconstructed volumes.
