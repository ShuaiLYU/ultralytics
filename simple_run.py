#!/usr/bin/env python3
"""Minimal YOLOA script: fit on normal images → predict on test images → save results."""

import cv2
import torch
import numpy as np
from pathlib import Path
from ultralytics.yoloa import YOLOA
from ultralytics.utils import YAML

# ============================================================
# CONFIG
# ============================================================
CKPT = "/Users/louis/workspace/ultra_louis_work/expman/data/pulled/yoloa_clean/26m_yoloav2_v8.2_clsgrad_flr01_perscale_mul_wd2_lr08_v1/weights/best.pt"
FIT_CFG = "ultralytics/cfg/yoloa_fit_spatial.yaml"
DEVICE = "mps"

BOTTLE_ROOT = Path("/Users/louis/workspace/ultra_louis_work/buffer/AnomalyData/bottle")
VARIANTS = ["bottle_full_640"]
# VARIANTS = ["bottle_original", "bottle_full_640", "bottle_top_640", "bottle_bottom_640"]
# ============================================================


def concat_result(img_path, model, pred_result, out_path):
    """Save a 3-panel image: original | heatmap overlay | prediction boxes.
    Heatmap is resized to match the original image and annotated with min/max."""
    # Original image
    orig = cv2.imread(str(img_path))
    if orig is None:
        return

    # Heatmap from model
    hm = getattr(model.model, "_last_heatmap", None)
    if hm is None:
        return
    if isinstance(hm, torch.Tensor):
        hm = hm.detach().cpu().numpy()
    hm = hm.squeeze()
    hmin, hmax = float(hm.min()), float(hm.max())

    # Resize to original image size
    hm_rs = cv2.resize(hm, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_LINEAR)
    # Normalise to [0, 255] for JET
    hm_norm = np.clip((hm_rs - hmin) / (hmax - hmin + 1e-8), 0, 1)
    hm_u8 = (hm_norm * 255).astype(np.uint8)
    hm_jet = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)

    # Overlay heatmap on original (alpha blend)
    alpha = 0.5
    overlay = cv2.addWeighted(orig, 1 - alpha, hm_jet, alpha, 0)

    # Annotate min/max on heatmap overlay
    cv2.putText(overlay, f"min={hmin:.4f}", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(overlay, f"max={hmax:.4f}", (8, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Prediction boxes
    pred_img = pred_result.plot()

    # Concatenate 3 panels side by side
    h = max(orig.shape[0], overlay.shape[0], pred_img.shape[0])
    panels = []
    for panel in [orig, overlay, pred_img]:
        ph, pw = panel.shape[:2]
        if ph < h:
            pad = np.zeros((h - ph, pw, 3), dtype=np.uint8)
            panel = np.vstack([panel, pad])
        panels.append(panel)
    out = np.hstack(panels)

    # Title bar
    title = f"  original  |  heatmap overlay (min={hmin:.4f} max={hmax:.4f})  |  predictions  "
    bar = np.zeros((24, out.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, title, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
    out = np.vstack([bar, out])

    cv2.imwrite(str(out_path), out)


def main():
    fit_args = YAML.load(FIT_CFG)
    imgsz = int(fit_args["imgsz"])

    infer = {
        "heat_norm": fit_args.get("heat_norm", "none"),
        "heat_edge_sigma": fit_args.get("heat_edge_sigma", 1.0),
        "heat_edge": bool(fit_args.get("heat_edge", False)),
    }

    for v in VARIANTS:
        normal_dir = BOTTLE_ROOT / v / "images" / "train"
        val_dir = BOTTLE_ROOT / v / "images" / "val"
        out_dir = Path(f"runs/temp/{v}/predict")
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"  {v}")
        print(f"  train: {normal_dir}")
        print(f"  val:   {val_dir}")
        print(f"  out:   {out_dir}")
        print(f"{'='*60}")

        # 1. Load + fit
        model = YOLOA(CKPT)
        model.fit(str(normal_dir), cfg=FIT_CFG, device=DEVICE, name=v)

        # 2. Predict on all val images (recursive into good/defective)
        imgs = sorted(p for p in val_dir.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        for img_path in imgs:
            rel = img_path.relative_to(val_dir)  # e.g. good/xxx.jpg
            subdir = out_dir / rel.parent
            subdir.mkdir(parents=True, exist_ok=True)

            r = model.predict(
                str(img_path), prior="heatmap", imgsz=imgsz,
                conf=0.2, iou=0.2, device=DEVICE, verbose=False, **infer,
            )[0]
            concat_result(img_path, model, r, subdir / f"{img_path.stem}_result.jpg")

        print(f"  [{v}] {len(imgs)} images -> {out_dir}")

        del model
        if DEVICE == "mps":
            torch.mps.empty_cache()

    print(f"\n✅ All {len(VARIANTS)} variants done!")


if __name__ == "__main__":
    main()
