---
comments: true
description: Detect and localize defects and anomalies in industrial images using YOLOA. Learn out-of-box inference, training-free memory-bank fitting, and full supervised training.
keywords: YOLOA, YOLO Anomaly, anomaly detection, defect detection, industrial inspection, MVTec, training-free, memory bank, heatmap, Ultralytics
model_name: yolo26-anomaly
---

# Anomaly Detection

<img width="1024" src="https://www.ultralytics.com/_next/image?url=https%3A%2F%2Fcdn.prod.website-files.com%2F64a2c1a4f4b4c4e7d0f8d5e7%2F66d2e2b3e1c3a3b5f7c3d8e2_anomaly-detection.avif&w=1920&q=75" alt="YOLOA anomaly detection">

Anomaly detection is the task of identifying images or regions that deviate from what is considered normal. Unlike [object detection](detect.md) (which finds known objects) or [classification](classify.md) (which assigns a single label), anomaly detection answers: **"is there anything unusual in this image, and where?"** — even for defect types never seen during setup.

YOLOA (YOLO Anomaly) provides three usage tiers, from zero-effort to full training:

1. **Out-of-box** — load any YOLO checkpoint and run `predict(prior="none")` as a baseline detector.
2. **Training-free** — `fit()` on a handful of normal images to build a memory bank of normal features, then detect anomalies via heatmap-guided fusion.
3. **Supervised training** — fine-tune a pretrained YOLO/YOLOE checkpoint with mask-prior augmentation and OOD evaluation.

The output of YOLOA is a set of bounding boxes (and optionally instance masks) with confidence scores, plus a pixel-level anomaly heatmap for visualization.

## Models

YOLOA v2 model configs are available in two variants:

| Model Config | Head | Description | Data Requirement |
|-------------|------|-------------|------------------|
| [yolo26-anomaly.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v2/yolo26-anomaly.yaml) | `Detect` | Bounding-box anomaly detection | Bounding boxes (binary: normal/anomaly) |
| [yolo26-anomaly-seg.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v2/yolo26-anomaly-seg.yaml) | `Segment` | Per-instance mask prediction | Polygon masks (binary: normal/anomaly) |

Available scales: **n**, **s**, **m**, **l**, **x** (appended to the model name, e.g., `yolo26m-anomaly.yaml`).

!!! tip

    YOLOA models are **not** distributed as pretrained `.pt` files. Instead, you load a pretrained YOLO or YOLOE checkpoint as the backbone, then use the anomaly config for the head. The model downloads automatically from the latest Ultralytics [release](https://github.com/ultralytics/assets/releases).

## Quick Start (3 Tiers)

### Tier 1: Out-of-Box Inference

Load any YOLO checkpoint and predict immediately — no training, no memory bank. This gives you a vanilla YOLO detector as a baseline.

!!! example

    === "Python"

        ```python
        from ultralytics.yoloa import YOLOA

        # Load any YOLO checkpoint
        model = YOLOA("yolo26n.pt")

        # Predict with no prior (passthrough — vanilla YOLO)
        results = model.predict("path/to/image.jpg", prior="none")

        # Visualize results
        results[0].show()
        ```

### Tier 2: Fit with Normal Images (Training-Free)

Build a memory bank from ~20–200 normal (defect-free) images. The model compares each test image's features against the bank to produce an anomaly heatmap, which is fused into the detector to boost sensitivity.

!!! example

    === "Python"

        ```python
        from ultralytics.yoloa import YOLOA

        # Load checkpoint
        model = YOLOA("yolo26n.pt")

        # Fit memory bank on normal images
        model.fit(
            "path/to/bottle/train/good",  # directory of normal images
            name="bottle",
            cfg="yoloa_fit_default.yaml",  # fit configuration
            cache="banks/",                # cache path
            device="mps",                  # or "cpu", "cuda:0"
        )

        # Predict with heatmap prior (memory bank → heatmap → fusion → detection)
        results = model.predict("path/to/test.png", prior="heatmap")

        # Save the fitted model for reuse (carries the memory bank)
        model.save("bottle_fitted.pt")
        ```

### Tier 3: Supervised Training

Fine-tune a pretrained checkpoint with heatmap-guided fusion, OOD cross-dataset evaluation, and mask-prior augmentation.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load pretrained checkpoint with anomaly_v2 task
        model = YOLO("yoloe-26m.pt", task="anomaly_v2")

        # Train with anomaly config
        results = model.train(
            data="mvtec_binary.yaml",
            model="yolo26m-anomaly.yaml",
            epochs=100,
            imgsz=640,
            batch=256,
        )
        ```

    === "CLI"

        ```bash
        # Train with anomaly v2 config
        yolo train \
            task=anomaly_v2 \
            model=yolo26m-anomaly.yaml \
            pretrained=yoloe-26m.pt \
            data=mvtec_binary.yaml \
            epochs=100 \
            imgsz=640
        ```

## Train

Training a YOLOA v2 model uses standard YOLO training with additional anomaly-specific behavior: mask-prior dropout for robustness, optional OOD cross-dataset evaluation, and heatmap-guided fusion.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yoloe-26m.pt", task="anomaly_v2")       # pretrained checkpoint
        model = YOLO("yolo26m-anomaly.yaml", task="anomaly_v2")  # from YAML config

        # Train
        results = model.train(data="mvtec_binary.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Train from pretrained checkpoint
        yolo train task=anomaly_v2 model=yolo26m-anomaly.yaml \
            pretrained=yoloe-26m.pt data=mvtec_binary.yaml epochs=100 imgsz=640

        # Train from config only
        yolo train task=anomaly_v2 model=yolo26m-anomaly.yaml \
            data=mvtec_binary.yaml epochs=100 imgsz=640
        ```

### Training Data Format

YOLOA v2 training uses standard YOLO detection or segmentation format with **binary labels** (1 class: anomaly). Normal images have empty label files (no boxes). See the [Detection Dataset](../datasets/detect/index.md) guide for format details.

```yaml
# Example data.yaml
path: /path/to/dataset
train: train/images
val: val/images

names:
  0: anomaly
nc: 1
```

### Key Training Parameters

All anomaly v2 settings live in the model YAML under the `anomaly_v2` block. Key parameters you may want to adjust:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `p_drop` | 0.6 | Probability of dropping the mask prior per sample (anti-shortcut) |
| `fusion_mode` | `bias` | How the heatmap fuses into features: `bias`, `film`, `soft`, `queryfilm` |
| `mask_mode` | `gauss` | Prior shape: `rect` (box), `gauss` (2D Gaussian blob) |
| `sigma_factor` | `[0.20, 0.40]` | Gaussian sigma range (fraction of box size) |
| `mask_aug_passes` | 2 | Number of prior augmentation passes |
| `seg_branch` | `false` | Enable learnable SegBranch heatmap predictor |
| `two_head` | `false` | Enable auxiliary prior head (head_b) |

For the full list, see the config file: [yolo26-anomaly.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v2/yolo26-anomaly.yaml).

### OOD Evaluation

During training, YOLOA can run periodic out-of-distribution evaluation on MVTec AD categories. This measures generalization to unseen defect types. Configure via the `anomaly_v2` block:

```yaml
test_val_freq: 3          # run every 3 epochs (0 = off)
test_heatmap_prior: true  # evaluate with memory-bank heatmap
test_none_prior: false    # also evaluate without prior
test_categories: null     # null = all 15 MVTec categories
```

## Val

Validate a trained or fitted YOLOA model.

!!! example

    === "Python (Trained Model)"

        ```python
        from ultralytics import YOLO

        model = YOLO("path/to/best.pt", task="anomaly_v2")
        metrics = model.val(data="mvtec_binary.yaml")
        ```

    === "Python (Training-Free)"

        ```python
        from ultralytics.yoloa import YOLOA

        model = YOLOA("bottle_fitted.pt")
        metrics = model.val(data="bottle.yaml", prior="heatmap", imgsz=640)
        ```

    === "CLI"

        ```bash
        yolo val task=anomaly_v2 model=path/to/best.pt data=mvtec_binary.yaml
        ```

Validation runs a two-pass evaluation (when using a trained model):
- **mask-off** — vanilla YOLO, no prior → honest detection floor
- **mask-on** — with GT mask prior → oracle upper bound

The validator also computes **image AUROC** and **pixel AUROC** from the model's anomaly heatmap.

## Predict

Run inference with a trained or fitted YOLOA model.

!!! example

    === "Python (Trained Model)"

        ```python
        from ultralytics import YOLO

        model = YOLO("path/to/best.pt", task="anomaly_v2")
        results = model.predict("path/to/image.jpg", imgsz=640)
        results[0].show()
        ```

    === "Python (Training-Free with Prior Modes)"

        ```python
        from ultralytics.yoloa import YOLOA

        model = YOLOA("bottle_fitted.pt")

        # No prior — vanilla YOLO baseline
        results = model.predict("test.png", prior="none")

        # Heatmap prior — memory bank anomaly map
        results = model.predict("test.png", prior="heatmap")

        # Mask prior — provide a region prompt
        results = model.predict("test.png", prior="mask", imgsz=640)
        ```

    === "CLI"

        ```bash
        yolo predict task=anomaly_v2 model=path/to/best.pt source=path/to/image.jpg
        ```

### Prior Modes

| Mode | Source | Use Case |
|------|--------|----------|
| `none` | No prior (passthrough) | Honest baseline — how well the model detects without hints |
| `heatmap` | Memory bank feature-distance map | Deploy: training-free anomaly detection |
| `mask` | External mask / user prompt | Interactive: "look at this region" |
| `segment` | SegBranch predicted heatmap | When SegBranch is trained, self-contained inference |
| `heatmap_fused` | Bank + learned scorer ensemble | Best accuracy when both are available |

## Export

Export a trained YOLOA model to standard formats for deployment.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("path/to/best.pt", task="anomaly_v2")
        model.export(format="onnx")
        ```

    === "CLI"

        ```bash
        yolo export model=path/to/best.pt format=onnx
        ```

{% include "macros/export-table.md" %}

!!! note

    Exported models carry the trained detector but not the memory bank. For training-free heatmap-guided inference, use the Python `YOLOA` wrapper which manages the bank separately.

## FAQ

### What is the difference between anomaly detection and regular object detection?

Object detection finds **known** object classes (person, car, etc.) — you train on labeled examples of every class you want to detect. Anomaly detection finds **anything unusual** — you only need normal (defect-free) examples. The model learns what "normal" looks like and flags deviations, including defect types it has never seen. This is critical for industrial inspection where defect types are unpredictable.

### Do I need to train YOLOA to use it?

No. You can use YOLOA in **training-free mode**: load any YOLO checkpoint with `YOLOA()`, call `fit()` on ~20 normal images to build a memory bank, then `predict(prior="heatmap")`. No gradient training required. See [Tier 2](#tier-2-fit-with-normal-images-training-free) above.

### When should I use supervised training instead of training-free mode?

Training-free mode works well when normal images are homogeneous and defects are visually distinct (e.g., MVTec texture categories like carpet, leather). Supervised training is better when:
- You have labeled defect examples and want maximum accuracy
- Normal images are highly varied (different lighting, orientations, backgrounds)
- You need to deploy on resource-constrained devices (trained model runs standalone)

### What data format does YOLOA expect?

For training-free mode: a directory of normal images (any format PIL can read). For supervised training: standard [YOLO detection format](../datasets/detect/index.md) with binary labels (0 = anomaly). Normal images have empty label files. For segmentation: [YOLO segmentation format](../datasets/segment/index.md) with polygon masks.

### What is a prior, and which mode should I use?

A **prior** is a spatial hint that tells the model "pay attention to this region." At deploy time, the `heatmap` prior is the recommended default — it's computed automatically from the memory bank and requires no user input. Use `none` to measure the model's unaided detection performance. Use `mask` for interactive prompting (e.g., "check this scratched area").
