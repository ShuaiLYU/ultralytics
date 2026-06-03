"""Minimal demo: compare for v2 anomaly detection mask prompt.

Run from the repo root:

    cd /Users/louis/workspace/ultra_louis_work/ultralytics
    python docs_yoloa_v2/demo_mask_prompt.py

Writes ONE fixed-path PNG (overwritten each run, so VSCode can keep it open):
    [ original | mask_off | seg_pred | seg_bbox | gt overlay | mask_on | mb_prior(in) | mb_det ]

Columns 3-4 require a SegBranch in the checkpoint (else blank).
Columns 5-6 require a GT annotation mask (else blank).
Columns 7-8 require normal (good) images to build the memory bank.

ultralytics quirks:
  * ``end2end`` / ``max_det`` are MODEL-level -- only baked in on the FIRST
    ``predict()`` call (when ``setup_model`` runs). Pass them on the warmup.
  * ``conf`` / ``iou`` are PER-CALL NMS args.
"""

from pathlib import Path

import cv2
import numpy as np
import torch

from ultralytics import YOLO
from ultra_ext.yoloa import get_random_sample, get_mvtec_raw_support
from ultra_ext.im import concat_samh


# ============================================================================
# EDIT
# ============================================================================
# experiment = "26m_yoloav2_v5_binary_cm20_gauss_pd50_v1"
# experiment = "26m_yoloav2seg_v5_binary_cm20_rect_pd50_a1_v1"
# experiment = "26m_yoloav2_softhint_rect_pd50_v1"
experiment = "26m_yoloav2_softhint_rect_pd50_seg_a1_v1"


MODEL_PATH = f"/Users/louis/workspace/ultra_louis_work/ultra6/runs/yoloa_v2/{experiment}/weights/best.pt"
SAVE_PATH  = "../runs/temp/demo_mask_prompt/compare.png"
SAVE_MB_FLOW_PATH = "../runs/temp/demo_mask_prompt/mb_prior_flow.png"
SAVE_MB_STAGE_PATH = "../runs/temp/demo_mask_prompt/mb_prior_stage_concat.png"
CATEGORY   = "leather"  # from MVTec AD (or "all" for random across all categories)

CONF, IOU, END2END, MAX_DET = 0.05, 0.05, False, 9
GOOD=False  # if True, sample from "good" (non-anomalous) images; else from "bad" (anomalous) ones

# Memory-bank settings
MB_LAYER_IDX         = 4      # backbone tap layer (stride-8 output for yolo26-family)
MB_MAX_IMGS          = 200     # cap support-set size for speed
MB_K                 = 5
MB_TEMPERATURE       = 3.0
MB_IMGSZ             = 224
MB_ACCUMULATE_THRESH = 0.3    # OBMA novelty filter: drop features with cosine-dist < thresh
MB_CACHE_DIR         = "../runs/temp/mb_cache"  # bank tensors cached here by (model, category)
MB_REBUILD           = False  # set True to force a fresh bank build (ignore cache)
MB_DIRECT_HEATMAP_PRIOR = True  # True: feed resized heatmap directly as external prior
MB_DIRECT_PRIOR_SCALE   = 0.8    # amplitude scale when direct mode is enabled

def overlay_mask(img_bgr, mask_path, color=(0, 0, 255), alpha=0.45):
    """Red-tint a BGR image wherever the binary mask is non-zero."""
    out = img_bgr.copy()
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return out
    m = cv2.resize(m, (out.shape[1], out.shape[0]), interpolation=cv2.INTER_NEAREST)
    sel = m > 0
    out[sel] = (alpha * np.array(color) + (1 - alpha) * out[sel]).astype(np.uint8)
    return out

from ultralytics.nn.modules.anomaly_v2 import BboxMaskRenderer

_RENDERER = None

def load_mask_as_prior(path, size=80, scale=0.99, sigma_factor=0.25):
    global _RENDERER
    if _RENDERER is None or _RENDERER.mask_size != size:
        _RENDERER = BboxMaskRenderer(mask_size=size, mode="gauss", sigma_factor=sigma_factor)

    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    H, W = img.shape
    ys, xs = np.where(img > 0)
    if len(xs) == 0:
        return torch.zeros(1, 1, size, size)

    x1, x2 = xs.min(), xs.max() + 1
    y1, y2 = ys.min(), ys.max() + 1
    cx = ((x1 + x2) / 2) / W
    cy = ((y1 + y2) / 2) / H
    w = (x2 - x1) / W
    h = (y2 - y1) / H

    bboxes = torch.tensor([[cx, cy, w, h]], dtype=torch.float32)
    batch_idx = torch.zeros(1, dtype=torch.long)
    mask = _RENDERER(bboxes, batch_idx, batch_size=1)  # (1,1,size,size) in [0,1]
    return mask * scale

def memorybank_heatmap_to_prior(
    heatmap,
    size=80,
    thresh=None,
    scale=0.8,
    direct_heatmap_prior=False,
    direct_scale=0.8,
):
    """Convert memory-bank heatmap to prior and keep intermediate states for analysis.

    Args:
        heatmap (np.ndarray): input MB heatmap in [0, 1].
        size (int): target prior size.
        thresh (float | None): threshold used by rect/CC mode. If None,
            uses adaptive threshold from current heatmap.
        scale (float): amplitude scaling for rect/CC output prior.
        direct_heatmap_prior (bool): if True, feed resized heatmap directly;
            if False, use threshold + peak-CC + bounding-rect conversion.
        direct_scale (float): amplitude scaling for direct-heatmap prior.
    """
    h = cv2.resize(heatmap.astype("float32"), (size, size), interpolation=cv2.INTER_LINEAR)
    h = np.clip(h, 0.0, 1.0)

    if direct_heatmap_prior:
        prior_mask = (h * float(direct_scale)).astype("float32")
        prior = torch.from_numpy(prior_mask)[None, None]
        debug = {
            "resized_heatmap": h,
            "binary_mask": (h > 0).astype("float32"),
            "peak_mask": np.zeros_like(h, dtype="float32"),
            "rect_mask": h,
            "prior_mask": prior_mask,
            "peak_xy": tuple(int(v) for v in np.unravel_index(int(np.argmax(h)), h.shape)[::-1]),
            "peak_score": float(np.max(h)),
            "threshold": 0.0,
            "num_components": 0,
            "peak_label": 0,
        }
        return prior, debug

    if thresh is None:
        thresh = float(np.clip(np.quantile(h, 0.995) * 0.7, 0.12, 0.45))

    rect = np.zeros((size, size), dtype="float32")
    peak_mask = np.zeros((size, size), dtype="float32")
    fg = (h > thresh).astype("uint8")
    labels = np.zeros((size, size), dtype="int32")

    # Keep only the connected component containing the global peak to avoid
    # one huge box caused by sparse high-score speckles.
    peak_idx = int(np.argmax(h))
    py, px = divmod(peak_idx, size)
    peak_label = 0
    num = 0

    if h[py, px] > thresh:
        num, labels = cv2.connectedComponents(fg, connectivity=8)
        if num > 1:
            peak_label = int(labels[py, px])
            if peak_label > 0:
                peak_mask = (labels == peak_label).astype("float32")
                ys, xs = np.where(labels == peak_label)
                if len(xs):
                    rect[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1] = 1.0

    prior = torch.from_numpy(rect * scale)[None, None]
    debug = {
        "resized_heatmap": h,
        "binary_mask": fg.astype("float32"),
        "peak_mask": peak_mask,
        "rect_mask": rect,
        "prior_mask": (rect * scale).astype("float32"),
        "peak_xy": (px, py),
        "peak_score": float(h[py, px]),
        "threshold": float(thresh),
        "num_components": int(num),
        "peak_label": int(peak_label),
    }
    return prior, debug


def _mask_panel(mask_01, out_shape):
    m = cv2.resize(mask_01.astype("float32"), (out_shape[1], out_shape[0]), interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor((np.clip(m, 0.0, 1.0) * 255).astype("uint8"), cv2.COLOR_GRAY2BGR)


def heatmap_stats(heatmap):
    """Return compact scalar stats for labeling/debugging."""
    h = heatmap.astype("float32")
    return {
        "min": float(np.min(h)),
        "max": float(np.max(h)),
        "mean": float(np.mean(h)),
        "p99": float(np.quantile(h, 0.99)),
    }


def save_mb_prior_flow_figure(img_bgr, mb_heat, debug, save_path):
    """Save a separate figure showing MB heatmap -> prior conversion steps."""
    h_img, w_img = img_bgr.shape[:2]
    heat_rs = debug["resized_heatmap"]

    panels = [
        title(img_bgr, "input"),
        title(heatmap_overlay(img_bgr, mb_heat), "mb_heat"),
        title(_mask_panel(heat_rs, (h_img, w_img)), f"resized heat (peak={debug['peak_score']:.3f})"),
        title(_mask_panel(debug["binary_mask"], (h_img, w_img)), f"binary > thr ({debug['threshold']:.3f})"),
        title(_mask_panel(debug["peak_mask"], (h_img, w_img)), f"peak CC (n={debug['num_components']})"),
        title(_mask_panel(debug["rect_mask"], (h_img, w_img)), "bbox rect"),
        title(_mask_panel(debug["prior_mask"], (h_img, w_img)), "prior mask (scaled)"),
    ]
    out = concat_samh(panels, gap=12, gap_color=(255, 255, 255))
    save_path = Path(save_path).resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), out)
    return save_path


def save_mb_prior_stage_concat(img_bgr, mb_heat, debug, save_path):
    """Save a compact stage image: heatmap, intermediate masks, final prior."""
    h_img, w_img = img_bgr.shape[:2]
    stats = heatmap_stats(mb_heat)
    prior_overlay = cv2.addWeighted(
        _mask_panel(debug["prior_mask"], (h_img, w_img)),
        0.5,
        img_bgr,
        0.5,
        0,
    )
    panels = [
        title(
            heatmap_overlay(img_bgr, mb_heat),
            f"1) mb_heat (max={stats['max']:.3f}, mean={stats['mean']:.3f}, p99={stats['p99']:.3f})",
        ),
        title(_mask_panel(debug["binary_mask"], (h_img, w_img)), "2) intra: binary"),
        title(_mask_panel(debug["peak_mask"], (h_img, w_img)), "3) intra: peak_cc"),
        title(_mask_panel(debug["rect_mask"], (h_img, w_img)), "4) intra: rect"),
        title(prior_overlay, "5) final prior"),
    ]
    out = concat_samh(panels, gap=12, gap_color=(255, 255, 255))
    save_path = Path(save_path).resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), out)
    return save_path


def heatmap_overlay(img_bgr, heatmap, alpha=0.45):
    """JET-colormap overlay of a [0,1] heatmap onto a BGR image."""
    h = cv2.resize(heatmap.astype("float32"), (img_bgr.shape[1], img_bgr.shape[0]),
                   interpolation=cv2.INTER_LINEAR)
    h = np.clip(h, 0.0, 1.0)
    cmap = cv2.applyColorMap((h * 255).astype("uint8"), cv2.COLORMAP_JET)
    return cv2.addWeighted(cmap, alpha, img_bgr, 1 - alpha, 0)


def title(img, text, h=72, scale=1.6, thickness=4):
    bar = np.full((h, img.shape[1], 3), 30, np.uint8)  # near-black bar
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x = max(8, (bar.shape[1] - tw) // 2)
    y = (h + th) // 2
    cv2.putText(bar, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return np.vstack([bar, img])


def blank_like(img):
    """Light-gray placeholder matching ``img`` shape."""
    return np.full_like(img, 230)


def _preprocess_img(path: Path, imgsz: int = 640) -> torch.Tensor:
    """Letterbox + normalize a single image to (1, 3, H, W) float32 in [0, 1]."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    # Simple letterbox resize (no padding math needed — just resize to square)
    img = cv2.resize(img, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # (3, H, W)
    return t.unsqueeze(0)  # (1, 3, H, W)


def build_mb_from_category(model, category: str) -> bool:
    """Install backbone tap and build memory bank from MVTec good images.

    The bank is cached under ``MB_CACHE_DIR`` keyed by model stem, category,
    layer index, and image count; subsequent runs skip the forward-pass build.

    Args:
        model: the underlying YOLOAnomalyV2Model (``y.model``).
        category: MVTec category name.

    Returns:
        True if bank was built (or loaded from cache) successfully.
    """
    paths = get_mvtec_raw_support(category, cap=MB_MAX_IMGS)
    if not paths:
        print(f"[mb] no good images found for category={category!r}")
        return False

    model.install_backbone_tap(
        MB_LAYER_IDX, K=MB_K, temperature=MB_TEMPERATURE,
        accumulate_thresh=MB_ACCUMULATE_THRESH,
    )

    model_stem = Path(MODEL_PATH).stem
    cache_path = (
        Path(MB_CACHE_DIR)
        / f"{model_stem}_{category}_l{MB_LAYER_IDX}_n{MB_MAX_IMGS}_s{MB_IMGSZ}.pt"
    )

    if cache_path.exists() and not MB_REBUILD:
        print(f"[mb] loading cached bank ({cache_path.name})")
    else:
        print(f"[mb] building bank from {len(paths)} images (category={category!r})")

    batches = [_preprocess_img(p, MB_IMGSZ) for p in paths]
    model.build_memory_bank(batches, verbose=True, cache_path=cache_path, rebuild=MB_REBUILD)
    return True


def _predict_with_external_mask(y, image_path, external_mask, conf, iou):
    """Run one predict call with temporary external mask and restore state."""
    y.predictor.external_mask = external_mask
    result = y.predict(image_path, imgsz=320, save=False, verbose=False, conf=conf, iou=iou)
    y.predictor.external_mask = None
    return result


def main():
    image_path, mask_path = get_random_sample(CATEGORY, good=GOOD)
    has_anno = mask_path is not None and Path(mask_path).exists()

    y = YOLO(MODEL_PATH)
    print(f"Loaded: {type(y.model).__name__}  mask_mode={y.model.mask_renderer.mode}")

    # Build memory bank (must happen before first predict so the hook is installed)
    mb_ready = build_mb_from_category(y.model, CATEGORY)

    # Warmup -- bakes end2end/max_det into the model on first call.
    y.predict(image_path, imgsz=320, save=False, verbose=False,
              end2end=END2END, max_det=MAX_DET, conf=CONF, iou=IOU)

    img = cv2.imread(str(image_path))

    # 1-2: original + mask-off detection. Hook also captures _mb_bb_feat here.
    y.predictor.external_mask = None
    y.predictor.bbox_prompt = None
    r_off = y.predict(image_path, imgsz=320, save=False, verbose=False, conf=CONF, iou=IOU)

    panels = [
        title(img, "original"),
        title(r_off[0].plot(), f"mask_off  ({len(r_off[0].boxes)} det)"),
    ]

    # 3-4: SegBranch heatmap + heatmap-guided detection (blank if no seg head).
    seg_branch = getattr(y.model, "seg_branch", None)
    if seg_branch is not None:
        logits = y.model._seg_logits_buf
        if isinstance(logits, tuple):
            logits = logits[0]
        seg_heat = logits.detach().float().sigmoid()[0, 0].cpu().numpy()

        seg_prior, _ = memorybank_heatmap_to_prior(
            seg_heat,
            size=y.model.mask_size,
            direct_heatmap_prior=True,
        )
        r_seg_bbox = _predict_with_external_mask(y, image_path, seg_prior, conf=CONF, iou=IOU)

        panels.append(title(heatmap_overlay(img, seg_heat), "seg_pred"))
        panels.append(title(r_seg_bbox[0].plot(), f"seg_bbox  ({len(r_seg_bbox[0].boxes)} det)"))
    else:
        panels.append(title(blank_like(img), "seg_pred (n/a)"))
        panels.append(title(blank_like(img), "seg_bbox (n/a)"))

    # 5-6: GT mask overlay + GT-mask-guided detection (blank if no annotation).
    if has_anno:
        gt_prior = load_mask_as_prior(mask_path)
        r_on = _predict_with_external_mask(y, image_path, gt_prior, conf=CONF, iou=IOU)

        panels.append(title(overlay_mask(img, mask_path), "gt mask overlay"))
        panels.append(title(r_on[0].plot(), f"mask_on   ({len(r_on[0].boxes)} det)"))
    else:
        panels.append(title(blank_like(img), "gt mask (n/a)"))
        panels.append(title(blank_like(img), "mask_on (n/a)"))

    # 7-8: Memory-bank heatmap + MB-guided detection.
    # After mask-off predict, the backbone hook captured _mb_bb_feat.
    # The predictor calls disable_mask_once() so auto-injection was skipped;
    # we extract the heatmap manually and pass it as external_mask.
    mb_bb_feat = getattr(y.model, "_mb_bb_feat", None)
    if mb_ready and mb_bb_feat is not None:
        mb_heat_t = y.model.memory_bank.heatmap(mb_bb_feat)  # (1,1,H,W)
        mb_heat = mb_heat_t[0, 0].cpu().numpy()
        mb_stats = heatmap_stats(mb_heat)
        mb_prior, mb_debug = memorybank_heatmap_to_prior(
            mb_heat,
            size=getattr(y.model, "mask_size", 80),
            direct_heatmap_prior=MB_DIRECT_HEATMAP_PRIOR,
            direct_scale=MB_DIRECT_PRIOR_SCALE,
        )
        mb_thresh = float(mb_debug["threshold"])
        r_mb = _predict_with_external_mask(y, image_path, mb_prior, conf=CONF, iou=IOU)

        flow_path = save_mb_prior_flow_figure(img, mb_heat, mb_debug, SAVE_MB_FLOW_PATH)
        stage_path = save_mb_prior_stage_concat(img, mb_heat, mb_debug, SAVE_MB_STAGE_PATH)

        panels.append(
            title(
                _mask_panel(mb_debug["prior_mask"], img.shape[:2]),
                f"mb_prior(in) (max={mb_stats['max']:.3f}, p99={mb_stats['p99']:.3f}, thr={mb_thresh:.2f}, direct={MB_DIRECT_HEATMAP_PRIOR})",
            )
        )
        panels.append(title(r_mb[0].plot(), f"mb_det    ({len(r_mb[0].boxes)} det)"))
        print(f"mb flow: {flow_path}")
        print(f"mb stage: {stage_path}")
        print(
            "mb heat stats: "
            f"min={mb_stats['min']:.6f}, max={mb_stats['max']:.6f}, "
            f"mean={mb_stats['mean']:.6f}, p99={mb_stats['p99']:.6f}, thr={mb_thresh:.6f}"
        )
    else:
        panels.append(title(blank_like(img), "mb_prior(in) (n/a)"))
        panels.append(title(blank_like(img), "mb_det (n/a)"))

    out = concat_samh(panels, gap=12, gap_color=(255, 255, 255))

    save_path = Path(SAVE_PATH).resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), out)

    print(f"image : {Path(image_path).resolve()}")
    print(f"mask  : {Path(mask_path).resolve() if has_anno else None}")
    print(f"saved : {save_path}")


if __name__ == "__main__":
    main()
