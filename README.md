# SpineContextResUNet: Efficient 3D Spinal Localization

**SpineContextResUNet** is a high-efficiency 3D deep learning framework designed to localize the human spine within Computed Tomography (CT) volumes. By leveraging a custom **1.2M parameter architecture**, this project achieves state-of-the-art efficiency, allowing for complex medical image analysis on consumer-grade hardware.

![segmentation_sample](https://github.com/Nithurshen/SpineContextResUNet/blob/main/results/test/sub-verse758_localization.png)

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
├── checkpoints/             # Saved model states during training
├── data/                    # Dataset storage (Raw and Preprocessed)
├── logs/                    # Execution logs
│   ├── test_set_evaluation.txt
│   └── training_log.txt
├── models/                  # Best performing weights
│   └── best_model.pth
├── results/                 # Evaluation output
│   └── test/                # Visual results and subject-wise masks
├── src/                     # Core project source code
│   ├── dataset.py           # 3D NIfTI Loader and Normalization
│   ├── evaluate.py          # Full test set evaluation script
│   ├── model.py             # SpineResUNet Architecture
│   ├── param_count.py       # Model efficiency metrics
│   ├── preprocess.py        # Data preparation logic
│   └── train.py             # Training loop logic
├── visualizations/          # Stage 1 training visualizations
├── inference.py             # Single-instance prediction script
├── README.md                # Project documentation
├── requirements.txt         # Environment dependencies
└── test_metrics_dice.csv    # Quantified Dice performance logs
```

---

## Performance & Results

The model was rigorously tested across multiple datasets (VerSe and Global clinical scans), demonstrating high robustness to different imaging protocols.

### Metrics

| Metric | Result |
| --- | --- |
| **Mean Dice Score** | **0.8315** |
| **Parameter Count** | **1.25M** |

### Result Images

The evaluation files generated during testing, including subject-specific segmentation masks and sagittal visual overlays, are automatically stored in the `results/test/` directory for easy verification and analysis.

## Research Context

This project was developed to address the "Black Box" and "Computational Inefficiency" problems in modern clinical AI. By providing a lightweight yet accurate localization map, **SpineContextResUNet** serves as a robust foundation for automated vertebral fracture detection and spinal deformity analysis.
