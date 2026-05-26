"""Predict on MVTec / DAGM_AD val images — mirror of ``scripts/val_yoloa.py``.

Model construction (resolver, build_overrides, support_cap, cache, --rebuild) is
identical to ``val_yoloa.py``.  The script differs only in what it does after
building: instead of ``.val()``, it runs ``.predict()`` on either a single random
val image or every val image, saving a composite per image to ``runs/temp/``.

Usage:
    # Random anomalous image, full GT | YOLO | YOLOAnomaly composite
    python scripts/predict_yoloa.py --category leather

    # Random good (normal) image
    python scripts/predict_yoloa.py --category leather --kind good

    # Predict on ALL val images (good + anomaly)
    python scripts/predict_yoloa.py --category leather --all

    # All anomalous val images only
    python scripts/predict_yoloa.py --category leather --all --kind anomaly

    # DAGM_AD class (names disjoint with MVTec — no prefix needed)
    python scripts/predict_yoloa.py --category Class1 --all

    # Force rebuild of the cached YOLOAnomaly bank
    python scripts/predict_yoloa.py --category leather --rebuild

    # Reproducible random pick
    python scripts/predict_yoloa.py --category leather --seed 42

    # Single model only (single-panel output)
    python scripts/predict_yoloa.py --category leather --mode yolo
    python scripts/predict_yoloa.py --category leather --mode yoloa

    # Explicit image path
    python scripts/predict_yoloa.py --category leather --source /abs/path.png
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

from ultra_ext.yoloa import (
	MVTEC_CATEGORIES, DAGM_CATEGORIES, get_mvtec_yolo_data,
	get_or_build_yoloa, read_yolo_label,
)

# BASE_MODEL = "/Users/louis/workspace/ultra_louis_work/ultra6/runs/yoloa/26m_yolo_v2_binary_cm20_v1/weights/best.pt"

BASE_MODEL= "/Users/louis/workspace/ultra_louis_work/ultra6/runs/yoloa/26m_yolo_v3_binary_cm20_v1/weights/best.pt"

OUT_ROOT   = Path("./runs/temp")


# ── Same dataset resolver as val_yoloa.py ──────────────────────────────────

def _resolve(cat: str) -> dict:
	all_cats = MVTEC_CATEGORIES + DAGM_CATEGORIES
	canon = {c.lower(): c for c in all_cats}.get(cat.lower())
	if canon is None:
		raise SystemExit(f"Unknown category '{cat}'. Valid: {all_cats}")
	data = get_mvtec_yolo_data(canon)
	return {
		"name":       canon,
		"yaml":       Path(data["data_yaml"]),
		"support":    data["train_im_list"],
		"test_im":    data["test_im_list"],
		"test_good":  data["test_good_im_list"],
		"test_anom":  data["test_anomaly_im_list"],
	}


# ── Pick image(s) ──────────────────────────────────────────────────────────

def _filter_by_kind(info: dict, kind: str) -> list[str]:
	if kind == "good":    return list(info["test_good"])
	if kind == "anomaly": return list(info["test_anom"])
	return list(info["test_im"])     # "any"


def _pick_one(info: dict, kind: str, seed: int | None) -> str:
	pool = _filter_by_kind(info, kind)
	if not pool:
		raise SystemExit(f"No images of kind='{kind}' in {info['name']}.")
	return random.Random(seed).choice(pool)


# ── Composite rendering ────────────────────────────────────────────────────

def _label_strip(panel: np.ndarray, text: str) -> np.ndarray:
	h, w = panel.shape[:2]
	strip = np.zeros((28, w, 3), dtype=np.uint8)
	cv2.putText(strip, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
	            (255, 255, 255), 1, cv2.LINE_AA)
	return np.vstack([strip, panel])


def _heatmap_to_numpy(heatmap) -> np.ndarray:
	"""Convert tensor/array heatmap to a 2-D float32 array clipped to [0, 1]."""
	hm = heatmap.detach().cpu().float().numpy() if hasattr(heatmap, "detach") else np.asarray(heatmap, dtype=np.float32)
	if hm.ndim == 3:
		hm = hm.squeeze()
	hm = np.nan_to_num(hm, nan=0.0, posinf=1.0, neginf=0.0)
	return np.clip(hm, 0.0, 1.0)


def _heatmap_overlay(img_bgr: np.ndarray, heatmap, alpha: float = 0.45) -> np.ndarray:
	"""Blend a [0,1] heatmap (tensor or array) over *img_bgr* using JET colormap."""
	hm = _heatmap_to_numpy(heatmap)
	hm_u8 = (hm * 255.0).astype(np.uint8)
	hm_u8 = cv2.resize(hm_u8, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
	color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
	return cv2.addWeighted(img_bgr, 1.0 - alpha, color, alpha, 0)


def _heatmap_to_boxes(heatmap, ad_conf: float, min_area: int = 25) -> list[tuple[int, int, int, int, float]]:
	"""Threshold heatmap at ad_conf, return connected-component bboxes with max-score.

	Returns list of (x1, y1, x2, y2, score).  Coordinates are in heatmap pixel space
	(which is upsampled to the original image size by ``AnomalyPredictor._attach_heatmaps``).
	"""
	hm = _heatmap_to_numpy(heatmap)
	binary = (hm >= ad_conf).astype(np.uint8)
	if binary.sum() == 0:
		return []
	n_lab, lab, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
	out = []
	for j in range(1, n_lab):
		area = int(stats[j, cv2.CC_STAT_AREA])
		if area < min_area:
			continue
		x1 = int(stats[j, cv2.CC_STAT_LEFT])
		y1 = int(stats[j, cv2.CC_STAT_TOP])
		x2 = x1 + int(stats[j, cv2.CC_STAT_WIDTH])
		y2 = y1 + int(stats[j, cv2.CC_STAT_HEIGHT])
		score = float(hm[lab == j].max())
		out.append((x1, y1, x2, y2, score))
	return out


def _result_with_heatmap_boxes(src_result, boxes: list):
	"""Return a *deep copy* of *src_result* with `.boxes` replaced by heatmap-derived bboxes.

	Use ``src_result.plot()`` on the returned object so styling matches the rest of the
	pipeline exactly (same colors, fonts, score formatting).  Each row is
	``[x1, y1, x2, y2, score, cls=0]``.
	"""
	import copy as _copy
	import torch
	from ultralytics.engine.results import Boxes
	new = _copy.copy(src_result)            # shallow copy is enough — only swap .boxes
	if not boxes:
		new.boxes = Boxes(torch.zeros((0, 6), dtype=torch.float32), src_result.orig_img.shape[:2])
		return new
	rows = torch.tensor([[x1, y1, x2, y2, s, 0.0] for x1, y1, x2, y2, s in boxes],
	                    dtype=torch.float32)
	new.boxes = Boxes(rows, src_result.orig_img.shape[:2])
	return new


def _draw_gt(img_bgr: np.ndarray, gts) -> np.ndarray:
	vis = img_bgr.copy()
	for cls, x1, y1, x2, y2 in gts:
		cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
		cv2.putText(vis, f"gt:{cls}", (x1, max(y1 - 6, 12)),
		            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
	return vis


def _predict_one(src_path: Path, *, mode: str, yolo_model, yoloa_model,
                 pred_kw: dict, ad_conf: float = 0.4) -> tuple[np.ndarray, int, bool]:
	"""Run model(s) on one image, return (stitched panel, num_gt_boxes, is_anomaly)."""
	img_bgr = cv2.imread(str(src_path))
	if img_bgr is None:
		raise RuntimeError(f"Cannot read image: {src_path}")
	H, W = img_bgr.shape[:2]
	gts = read_yolo_label(src_path.with_suffix(".txt"), W, H)
	is_anom = len(gts) > 0

	panels: list[tuple[str, np.ndarray]] = []
	if mode == "both":
		panels.append((f"GT ({len(gts)})", _draw_gt(img_bgr, gts)))
	if mode in ("yolo", "both"):
		r = yolo_model.predict(str(src_path), **pred_kw)[0]
		panels.append((f"YOLO ({len(r.boxes)})", r.plot()))
	if mode in ("yoloa", "both"):
		r = yoloa_model.predict(str(src_path), **pred_kw)[0]
		panels.append((f"YOLOAnomaly per-level ({len(r.boxes)})", r.plot()))
		hm = getattr(r, "heatmap", None)
		if hm is not None:
			panels.append((f"Heatmap (max={float(hm.max()):.3f})",
			               _heatmap_overlay(img_bgr, hm)))
			# Heatmap → threshold by ad_conf → connected-component bboxes.
			hb = _heatmap_to_boxes(hm, ad_conf=ad_conf)
			r_hm = _result_with_heatmap_boxes(r, hb)
			panels.append((f"Heatmap bbox @>{ad_conf} ({len(hb)})", r_hm.plot()))

	h = min(p_img.shape[0] for _, p_img in panels)
	def _resize(p_img):
		s = h / p_img.shape[0]
		return cv2.resize(p_img, (int(p_img.shape[1] * s), h))
	stitched = np.hstack([_label_strip(_resize(p_img), title) for title, p_img in panels])
	return stitched, len(gts), is_anom


# ── Main ───────────────────────────────────────────────────────────────────

def main():
	p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
	p.add_argument("--category",    required=True,
	               help="Category name, e.g. 'leather' or 'Class1' (MVTec/DAGM names are disjoint)")
	p.add_argument("--all",         action="store_true",
	               help="Predict on every val image (still filtered by --kind)")
	p.add_argument("--kind",        choices=["good", "anomaly", "any"], default="anomaly",
	               help="Which kind of val image(s) to use (default: anomaly)")
	p.add_argument("--mode",        choices=["yolo", "yoloa", "both"], default="both")
	p.add_argument("--source",      default=None, help="Explicit image path (skip random pick)")
	p.add_argument("--seed",        type=int, default=None, help="Seed for random pick")
	p.add_argument("--base-model",  default=BASE_MODEL)
	p.add_argument("--ad-conf",     type=float, default=0.60,
	               help="YOLOAnomaly per_level ad_conf threshold (default 0.60)")
	p.add_argument("--support-cap", type=int, default=200,
	               help="Max support images for YOLOAnomaly bank (default 200)")
	p.add_argument("--rebuild",     action="store_true",
	               help="Force rebuild of the cached YOLOAnomaly bank")
	p.add_argument("--out-dir",     default=None,
	               help="Output dir (default: runs/temp/predict_<category>/ for --all, "
	                    "runs/temp/ for single)")
	p.add_argument("--imgsz",       type=int, default=640,
	               help="Inference image size (match the imgsz used during build).")
	args = p.parse_args()

	# Strip 'mvtec-'/'dagm-' prefix for compatibility with val_yoloa.py syntax.
	cat = args.category
	if cat.lower().startswith(("mvtec-", "dagm-")):
		cat = cat.split("-", 1)[1]
	info = _resolve(cat)

	# Pick image list.
	if args.source is not None:
		sources = [args.source]
	elif args.all:
		sources = _filter_by_kind(info, args.kind)
		if not sources:
			raise SystemExit(f"No images of kind='{args.kind}' in {info['name']}.")
	else:
		sources = [_pick_one(info, args.kind, args.seed)]
	print(f"[predict] category={info['name']}  mode={args.mode}  "
	      f"kind={args.kind}  n_images={len(sources)}")

	# Build models once (shared across all source images).
	yolo_model = None
	yoloa_model = None
	if args.mode in ("yolo", "both"):
		from ultralytics import YOLO
		print(f"[predict] loading plain YOLO from {args.base_model}")
		yolo_model = YOLO(args.base_model)
	if args.mode in ("yoloa", "both"):
		print(f"[predict] preparing YOLOAnomaly (rebuild={args.rebuild}, "
		      f"support_cap={args.support_cap}, ad_conf={args.ad_conf}) …")
		build_overrides = dict(
			feature_mode  = "per_level",
			ad_conf       = args.ad_conf,
			active_layers = [0, 1, 2],
			return_heatmap= False,
		)
		yoloa_model = get_or_build_yoloa(
			category=info["name"],
			base_model=args.base_model,
			support_imgs=info["support"],
			rebuild=args.rebuild,
			build_overrides=build_overrides,
			support_cap=args.support_cap,
			verbose=True,
		)

	# Output dir.
	if args.out_dir:
		out_dir = Path(args.out_dir).resolve()
	elif args.all or len(sources) > 1:
		out_dir = (OUT_ROOT / f"predict_{info['name']}").resolve()
	else:
		out_dir = OUT_ROOT.resolve()
	out_dir.mkdir(parents=True, exist_ok=True)

	# Predict args — match val pipeline's NMS exactly via agnostic_nms=True.
	pred_kw = dict(imgsz=args.imgsz, conf=0.001, iou=0.001, max_det=1000,
	               agnostic_nms=True, verbose=False)

	saved = []
	for i, src in enumerate(sources):
		src_path = Path(src)
		if not src_path.exists():
			print(f"  [{i:03d}/{len(sources)}] MISSING {src_path}")
			continue
		try:
			stitched, n_gt, is_anom = _predict_one(
				src_path, mode=args.mode,
				yolo_model=yolo_model, yoloa_model=yoloa_model,
				pred_kw=pred_kw, ad_conf=args.ad_conf,
			)
		except Exception as e:  # noqa: BLE001
			print(f"  [{i:03d}/{len(sources)}] FAILED on {src_path}: {e!r}")
			continue
		tag    = "anomaly" if is_anom else "good"
		sub    = src_path.parent.name      # defect type (e.g. "poke", "color")
		stem   = src_path.stem
		fname  = (f"predict_{info['name']}_{tag}_{stem}.png" if len(sources) == 1
		          else f"{i:03d}_{tag}_{sub}_{stem}.png")
		out_path = (out_dir / fname).resolve()
		cv2.imwrite(str(out_path), stitched)
		saved.append(out_path)
		# Print just the full absolute path on its own line — easy to copy / pipe / grep.
		print(out_path)

	if len(saved) > 1:
		print(f"\n[predict] saved {len(saved)} composite(s) → {out_dir}")


if __name__ == "__main__":
	main()


'''
python scripts/predict_yoloa.py --category carpet --all --kind any --mode yoloa \
       --base-model /Users/louis/workspace/ultra_louis_work/ultra6/runs/yoloa/26m_mergedata_v3/weights/best.pt \
       --imgsz 320 --ad-conf 0.4

python scripts/predict_yoloa.py --category carpet --all --kind any --mode yoloa \
       --base-model yolo26m.pt --rebuild \
       --imgsz 320 --ad-conf 0.4



'''