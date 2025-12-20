# SpineContextResUNet: Efficient 3D Spinal Localization

**SpineContextResUNet** is a high-efficiency 3D deep learning framework designed to localize the human spine within Computed Tomography (CT) volumes. By leveraging a custom **1.2M parameter architecture**, this project achieves state-of-the-art efficiency, allowing for complex medical image analysis on consumer-grade hardware.

## Key Innovations

### 1. Architectural Efficiency

Most 3D medical segmentation models (like the standard 3D U-Net) contain 30M+ parameters. **SpineContextResUNet** uses only **1.25 Million parameters**, representing a ~25x reduction in size while maintaining high segmentation accuracy.

### 2. Multi-Dilated Context Bottleneck

To compensate for its small size, the model utilizes a specialized bottleneck layer with **dilated convolutions**. This allows the network to "see" a larger anatomical area (wider receptive field) to understand the spine's global structure without increasing the number of weights.

### 3. Apple Silicon Optimization

The project is built specifically to utilize **Metal Performance Shaders (MPS)**, enabling high-speed training and inference on Apple M-series chips (M1/M2/M3/M4).

---

## Project Structure

```bash
├── src/
│   ├── model.py
│   ├── dataset.py
│   └── train.py
├── models/
├── visualizations/
└── test_metrics_dice.csv

```

---

## Performance & Results

The model was rigorously tested across multiple datasets (VerSe and Global clinical scans), demonstrating high robustness to different imaging protocols.

### Metrics

| Metric | Result |
| --- | --- |
| **Mean Dice Score** | **0.8315** |
| **Parameter Count** | **1.25M** |

## Research Context

This project was developed to address the "Black Box" and "Computational Inefficiency" problems in modern clinical AI. By providing a lightweight yet accurate localization map, **SpineContextResUNet** serves as a robust foundation for automated vertebral fracture detection and spinal deformity analysis.
