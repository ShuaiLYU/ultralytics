"""Run model.predict() on cached YOLOAnomaly + save 3-panel composites.

Composite layout per image:  [per-level bbox | heatmap overlay | heatmap-bbox]
(plus GT and plain-YOLO panels when ``--mode both``).

LOAD-ONLY: expects ``tools/build_yoloa.py --config <yaml>`` to have populated the cache.

Usage:
    # All anomaly test images of one category
    python tools/predict_yoloa.py --config tools/configs/yoloa_l6_320.yaml \\
           --category mvtec-leather --kind anomaly

    # Both good + anomaly
    python tools/predict_yoloa.py --config tools/configs/yoloa_l6_320.yaml \\
           --category mvtec-cable --kind any

    # Single random image (reproducible)
    python tools/predict_yoloa.py --config tools/configs/yoloa_l6_320.yaml \\
           --category mvtec-leather --seed 42

    # Adjust heatmap-bbox threshold (does NOT trigger rebuild)
    python tools/predict_yoloa.py --config tools/configs/yoloa_l6_320.yaml \\
           --category mvtec-tile --ad-conf 0.3
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _anomaly_common import (
	add_category_args, cache_dir_for, load_config, parse_categories,
	predict_dir_for, resolve_category,
)

from ultra_ext.yoloa import load_yoloa, read_yolo_label


# ── Composite-building helpers ─────────────────────────────────────────────

def _label_strip(panel: np.ndarray, text: str) -> np.ndarray:
	"""Stack a 28-px black label strip on top of *panel* with *text* in white."""
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
	"""Blend a [0,1] heatmap over *img_bgr* with JET colormap."""
	hm = _heatmap_to_numpy(heatmap)
	hm_u8 = (hm * 255.0).astype(np.uint8)
	hm_u8 = cv2.resize(hm_u8, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
	color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
	return cv2.addWeighted(img_bgr, 1.0 - alpha, color, alpha, 0)


def _heatmap_to_boxes(heatmap, ad_conf: float, min_area: int = 25) -> list[tuple[int, int, int, int, float]]:
	"""Threshold heatmap at ad_conf, return connected-component bboxes (x1,y1,x2,y2,score)."""
	hm = _heatmap_to_numpy(heatmap)
	binary = (hm >= ad_conf).astype(np.uint8)
	if binary.sum() == 0:
		return []
	n_lab, lab, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
	out = []
	for j in range(1, n_lab):
		if int(stats[j, cv2.CC_STAT_AREA]) < min_area:
			continue
		x1 = int(stats[j, cv2.CC_STAT_LEFT])
		y1 = int(stats[j, cv2.CC_STAT_TOP])
		x2 = x1 + int(stats[j, cv2.CC_STAT_WIDTH])
		y2 = y1 + int(stats[j, cv2.CC_STAT_HEIGHT])
		score = float(hm[lab == j].max())
		out.append((x1, y1, x2, y2, score))
	return out


def _result_with_heatmap_boxes(src_result, boxes: list):
	"""Shallow-copy a Result and swap ``.boxes`` with heatmap-derived bboxes (cls=0)."""
	import copy as _copy
	import torch
	from ultralytics.engine.results import Boxes
	new = _copy.copy(src_result)
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


def _build_composite(src_path: Path, *, mode: str, yolo_model, yoloa_model,
                     pred_kw: dict, ad_conf: float) -> tuple[np.ndarray, bool]:
	"""Run model(s) on one image, return (stitched panel, is_anomaly)."""
	img_bgr = cv2.imread(str(src_path))
	if img_bgr is None:
		raise RuntimeError(f"Cannot read image: {src_path}")
	H, W = img_bgr.shape[:2]
	gts = read_yolo_label(src_path.with_suffix(".txt"), W, H)
	is_anom = len(gts) > 0

	panels: list[tuple[str, np.ndarray]] = []
	# Always include GT as the first panel for easy reference.
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
			hb = _heatmap_to_boxes(hm, ad_conf=ad_conf)
			r_hm = _result_with_heatmap_boxes(r, hb)
			panels.append((f"Heatmap bbox @>{ad_conf} ({len(hb)})", r_hm.plot()))

	h = min(p_img.shape[0] for _, p_img in panels)
	def _resize(p_img):
		s = h / p_img.shape[0]
		return cv2.resize(p_img, (int(p_img.shape[1] * s), h))
	stitched = np.hstack([_label_strip(_resize(p_img), title) for title, p_img in panels])
	return stitched, is_anom


def _filter_by_kind(info: dict, kind: str) -> list[str]:
	if kind == "good":
		return list(info["test_good"])
	if kind == "anomaly":
		return list(info["test_anom"])
	return list(info["test_im"])


def main():
	p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
	add_category_args(p)
	p.add_argument("--mode", choices=["yolo", "yoloa", "both"], default="yoloa")
	p.add_argument("--kind", choices=["good", "anomaly", "any"], default="anomaly",
	               help="Which test images to visualise (default: anomaly).")
	p.add_argument("--source", default=None,
	               help="Explicit single image path (skips dataset iteration).")
	p.add_argument("--seed", type=int, default=None,
	               help="If set, pick one random image from --kind pool.")
	p.add_argument("--out-dir", default=None,
	               help="Output dir (default: runs/temp/predict_<category>/).")
	p.add_argument("--ad-conf", type=float, default=None,
	               help="Override config's ad_conf for heatmap binarisation + per_level scoring.")
	args = p.parse_args()

	cfg = load_config(args.config)
	cache_dir = cache_dir_for(cfg)
	ad_conf = args.ad_conf if args.ad_conf is not None else cfg["anomaly_arg"]["ad_conf"]
	names = parse_categories(args)

	print(f"Config     : {cfg['_path']}")
	print(f"Cache dir  : {cache_dir}")
	print(f"Mode       : {args.mode}    imgsz={cfg['model_arg']['imgsz']}    ad_conf={ad_conf}")
	print(f"Categories : {names}")

	# Forward predict() with the full model_arg from YAML.
	pred_kw = dict(cfg["model_arg"])

	# Load plain YOLO once (if used by --mode).
	yolo_model = None
	if args.mode in ("yolo", "both"):
		from ultralytics import YOLO
		print(f"[predict] loading plain YOLO from {cfg['base_model']}")
		yolo_model = YOLO(cfg["base_model"])

	for name in names:
		try:
			info = resolve_category(name)
		except KeyError as e:
			print(f"[{name}] {e}"); continue

		tag = info["name"]

		# Select sources.
		if args.source is not None:
			sources = [args.source]
		else:
			sources = _filter_by_kind(info, args.kind)
			if args.seed is not None and sources:
				sources = [random.Random(args.seed).choice(sources)]
		if not sources:
			print(f"[{tag}] no images of kind={args.kind}, skip")
			continue

		# Load YOLOAnomaly from cache.
		yoloa_model = None
		if args.mode in ("yoloa", "both"):
			try:
				yoloa_model = load_yoloa(tag, cfg["base_model"], cache_dir=cache_dir)
				yoloa_model.set_anomaly_args(ad_conf=ad_conf)
			except FileNotFoundError as e:
				print(f"[{tag}] {e}")
				continue

		# Output dir.
		out_dir = Path(args.out_dir).resolve() if args.out_dir \
			else predict_dir_for(cfg, tag).resolve()
		out_dir.mkdir(parents=True, exist_ok=True)
		print(f"\n=== {tag}: {len(sources)} image(s) → {out_dir} ===")

		saved = 0
		for i, src in enumerate(sources):
			src_path = Path(src)
			if not src_path.exists():
				print(f"  [{i:03d}/{len(sources)}] MISSING {src_path}")
				continue
			try:
				stitched, is_anom = _build_composite(
					src_path, mode=args.mode,
					yolo_model=yolo_model, yoloa_model=yoloa_model,
					pred_kw=pred_kw, ad_conf=ad_conf,
				)
			except Exception as e:  # noqa: BLE001
				print(f"  [{i:03d}/{len(sources)}] FAILED on {src_path}: {e!r}")
				continue
			vis_tag = "anomaly" if is_anom else "good"
			sub = src_path.parent.name
			out_path = out_dir / f"{i:03d}_{vis_tag}_{sub}_{src_path.stem}.png"
			cv2.imwrite(str(out_path), stitched)
			saved += 1
		print(f"  → saved {saved} composite(s)")


if __name__ == "__main__":
	main()
