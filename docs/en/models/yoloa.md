---
comments: true
description: YOLOA (YOLO Anomaly) is a training-free anomaly detection system that extends YOLO with memory-bank feature comparison and heatmap-guided detection for industrial defect inspection.
keywords: YOLOA, YOLO Anomaly, anomaly detection, industrial inspection, defect detection, MVTec, training-free, memory bank, heatmap fusion, Ultralytics
---

# YOLOA: Real-Time Anomaly Detection

## Introduction

YOLOA (YOLO Anomaly) extends the YOLO detection framework for **anomaly detection** — finding defects and irregularities in industrial images without needing labeled examples of every possible defect type. Built on top of YOLO and [YOLOE](yoloe.md) checkpoints, YOLOA adds a memory bank of normal-image features and a heatmap-guided fusion module that boosts the detector's sensitivity to anomalies.

Unlike traditional anomaly detection methods that require per-category training, YOLOA supports a **training-free workflow**: load a pretrained YOLO checkpoint, fit a memory bank on ~20 normal images, and immediately detect anomalies with pixel-level heatmaps. For maximum accuracy, YOLOA also supports full supervised fine-tuning with mask-prior augmentation, OOD cross-dataset evaluation, and configurable fusion architectures.

## Architecture Overview

YOLOA retains the standard YOLO structure — a convolutional **backbone** for feature extraction, a **PAN-FPN neck** for multi-scale fusion, and a **decoupled detection head** (Detect or Segment). Three novel components enable anomaly detection:

- **Memory Bank**: Stores backbone feature vectors extracted from normal (defect-free) images. At inference, each test image's features are compared against the bank via cosine similarity (or learned discriminators) to produce a 2D **anomaly heatmap** — bright where the image differs from normal.

- **Heatmap Fusion**: The anomaly heatmap is fused into the PAN features before the detection head. Four fusion strategies are available:
    - `bias` — additive per-pixel bias (simple, robust)
    - `film` — grouped FiLM modulation (learned channel-wise affine transform)
    - `soft` — temperature-softmax normalization + bias
    - `queryfilm` — K learned spatial queries with cross-attention

- **Mask-Prior Augmentation**: During supervised training, ground-truth bounding boxes are rendered as Gaussian masks and injected as the prior. Multiple augmentations (jitter, noise, dropout, fragment, distractor blobs) close the gap between clean GT masks and real memory-bank heatmaps at deploy time.

Optional components:

- **SegBranch**: A small auxiliary segmentation head that learns to predict the heatmap from backbone features, enabling self-contained inference without an external memory bank.
- **Two-Head**: Duplicates the detection head — head_a consumes raw PAN features (deployable honest detector), head_b consumes prior-fused features. Both trained jointly so the prior benefits from shared backbone gradients.
- **AnomalyMCDetect**: A decoupled detection head that separates binary anomaly detection (`is there a defect?`) from multi-class defect-type classification (`what kind?`).

## Available Models and Operating Modes

YOLOA v2 supports two head types across five model scales:

| Model Config | Head | Inference | Validation | Training | Export |
|-------------|------|:---------:|:----------:|:--------:|:------:|
| `yolo26n-anomaly.yaml` | Detect | ✅ | ✅ | ✅ | ✅ |
| `yolo26s-anomaly.yaml` | Detect | ✅ | ✅ | ✅ | ✅ |
| `yolo26m-anomaly.yaml` | Detect | ✅ | ✅ | ✅ | ✅ |
| `yolo26l-anomaly.yaml` | Detect | ✅ | ✅ | ✅ | ✅ |
| `yolo26x-anomaly.yaml` | Detect | ✅ | ✅ | ✅ | ✅ |
| `yolo26n-anomaly-seg.yaml` | Segment | ✅ | ✅ | ✅ | ✅ |
| `yolo26s-anomaly-seg.yaml` | Segment | ✅ | ✅ | ✅ | ✅ |
| `yolo26m-anomaly-seg.yaml` | Segment | ✅ | ✅ | ✅ | ✅ |
| `yolo26l-anomaly-seg.yaml` | Segment | ✅ | ✅ | ✅ | ✅ |
| `yolo26x-anomaly-seg.yaml` | Segment | ✅ | ✅ | ✅ | ✅ |

YOLOA configs are **not** distributed as pretrained `.pt` files. Instead, load a YOLO or YOLOE checkpoint as the backbone:

```python
from ultralytics import YOLO

# Load pretrained backbone with anomaly head
model = YOLO("yoloe-26m.pt", task="anomaly_v2")
model = YOLO("yolo26m-anomaly.yaml", task="anomaly_v2")
```

## Usage Flow

YOLOA provides three tiers, from zero-effort to full training. For detailed code examples, see the [Anomaly Detection task doc](../tasks/anomaly.md).

### 1. Out-of-Box (No Setup)

Load any YOLO checkpoint and predict immediately — no training, no memory bank. A quick baseline to see what vanilla YOLO detects before adding anomaly guidance.

### 2. Training-Free (Memory Bank)

Call `fit()` on a directory of normal images to build a feature memory bank. The model then produces anomaly heatmaps that guide detection. Still no gradient training — the backbone weights never change.

### 3. Supervised Training

Full fine-tuning with mask-prior augmentation, periodic OOD evaluation on MVTec AD categories, and configurable fusion architectures. Use this when you have labeled defect data and want maximum accuracy.

## Key Concepts

### Prior Modes

The **prior** is a spatial hint injected before the detection head. Deploy mode defaults to `heatmap` (memory bank); training uses `box` (GT-rendered masks).

| Mode | Source | Training? | Typical Use |
|------|--------|:---------:|-------------|
| `none` | Passthrough | ✅ | Honest baseline — no prior |
| `box` | GT bbox → gauss mask | Train only | Standard training prior |
| `heatmap` | Memory bank feature-distance | Deploy | Training-free anomaly detection |
| `mask` | External / user-provided mask | Both | Interactive prompt |
| `segment` | SegBranch predicted heatmap | Deploy | Self-contained inference |
| `heatmap_fused` | Bank + learned scorer | Deploy | Best ensemble accuracy |

### Memory Bank

The memory bank stores normal-image backbone features (from layers specified by `bb_layers`). At inference, each test image's features are compared to the bank via cosine similarity, producing a per-pixel anomaly score. The bank is built by `YOLOA.fit()` and cached to disk for reuse. Key parameters:

- **Bank size**: Subsample via coreset selection (`max_bank_size`) or greedy fill
- **Projection**: Optional PCA-like dimension reduction (`proj_dim`)
- **Calibration**: Per-channel compactness normalization

### Fusion Modes

How the heatmap modulates PAN features:

| Mode | Mechanism | When to Use |
|------|-----------|-------------|
| `bias` | Additive per-pixel bias (broadcast) | Default — simple, fast, robust |
| `film` | Grouped FiLM: channel-group affine (α·x + β) | When channel-wise modulation helps |
| `soft` | Softmax normalization + bias | Smooth, bounded response |
| `queryfilm` | K learned queries with cross-attention over P3 | Most expressive (experimental) |

### Mask Dropout (p_drop)

During training, with probability `p_drop` per sample, the mask prior is zeroed — forcing the model to also detect without hints. This prevents the model from becoming dependent on the prior and ensures robust deployment when the heatmap is imperfect.

## OOD Cross-Dataset Evaluation

YOLOA supports periodic out-of-distribution evaluation on MVTec AD during training. At each `test_val_freq` epoch, the trainer:

1. Builds per-category memory banks from normal MVTec training splits
2. Runs 3-mode evaluation (none / heatmap / mask prior) on test splits
3. Logs per-category AUROC, mAP, and fitness to wandb
4. Uses `test_metrics(heatmap_prior)/mAP10` as the OOD fitness for best.pt selection

Configure in the model YAML:

```yaml
anomaly_v2:
  test_val_freq: 3
  test_heatmap_prior: true
  test_categories: null  # all 15 MVTec categories
  test_fit_cfg: ../../yoloa_fit_default.yaml
```

## Performance

!!! note

    MVTec AD benchmark results are being prepared. Check back for per-category image-AUROC and pixel-AUROC tables.
    For preliminary results, see the [YOLOA experiments tracker](../reference/yoloa/experiments.md).

## Citations and Acknowledgments

!!! tip "YOLOA"

    YOLOA is an ongoing research project at Ultralytics. A formal publication is in preparation.

If you use YOLOA in your work, please cite:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @software{yoloa_ultralytics,
          author = {Louis Lyu and Ultralytics Team},
          title = {Ultralytics YOLOA: Real-Time Anomaly Detection},
          year = {2026},
          url = {https://github.com/ultralytics/ultralytics},
          license = {AGPL-3.0}
        }
        ```

## FAQ

### What makes YOLOA different from other anomaly detection methods?

YOLOA is built on YOLO's real-time detection backbone, not a separate anomaly-specific architecture. This means it inherits YOLO's speed, exportability, and ecosystem. The memory bank + heatmap fusion approach bridges the gap between training-free feature comparison and supervised detection — you can start with zero training and optionally fine-tune for higher accuracy.

### Can I use YOLOA without a memory bank?

Yes. Use `prior="none"` for passthrough (vanilla YOLO). The model still benefits from its pretrained features — it just won't have the anomaly heatmap guiding detection. This is the honest baseline for measuring how much the prior helps.

### How many normal images do I need for the memory bank?

Typically 20–200 images are sufficient. More images improve coverage of normal variation; fewer images work if the category is homogeneous (e.g., a single fabric texture). The bank is automatically subsampled to manage memory.

### Does YOLOA require a GPU?

Training-free mode runs on CPU (slower but functional — a few seconds per image on modern hardware). Supervised training benefits from a GPU. Inference with a fitted model also works on CPU. For edge deployment, export to ONNX or TensorRT — the exported model runs the detector standalone (memory bank inference is Python-only).

### What datasets does YOLOA support?

The memory-bank workflow works with any set of normal images — no labels needed for fitting. Supervised training uses standard YOLO detection format with binary labels. YOLOA is evaluated on [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) (15 industrial categories) and supports custom industrial datasets in YOLO format.
