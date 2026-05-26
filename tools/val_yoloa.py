"""Anomaly val — plain YOLO and/or YOLOAnomaly, across MVTec or DAGM_yolo.

Categories are addressed with a ``<dataset>-<name>`` prefix so both datasets share one
script.  Filename stays ``val_yoloa.py`` for continuity, but it now handles both.

Usage:
    # Single category
    python scripts/val_yoloa.py --category mvtec-leather
    python scripts/val_yoloa.py --category dagm-class1

    # Mix categories across datasets
    python scripts/val_yoloa.py --category mvtec-leather,mvtec-bottle,dagm-class1

    # All categories of one dataset
    python scripts/val_yoloa.py --all mvtec
    python scripts/val_yoloa.py --all dagm

    # All of both
    python scripts/val_yoloa.py --all mvtec,dagm

    # Only YOLOAnomaly, all DAGM, override threshold
    python scripts/val_yoloa.py --all dagm --mode yoloa --ad-conf 0.40

    # Only plain YOLO baseline on a single MVTec category
    python scripts/val_yoloa.py --category mvtec-tile --mode yolo
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

from sympy import python

from ultra_ext.yoloa import (
	MVTEC_CATEGORIES, DAGM_CATEGORIES, get_mvtec_yolo_data,
	print_metric_table, save_metric_csv,
	val_plain_yolo, val_yoloa,
)

BASE_MODEL = "/Users/louis/workspace/ultra_louis_work/ultra6/runs/yoloa/26m_yolo_v2_binary_cm20_v1/weights/best.pt"

# BASE_MODEL= "/Users/louis/workspace/ultra_louis_work/ultra6/runs/yoloa/26m_yolo_v3_binary_cm20_v1/weights/best.pt"
BASE_MODEL= "/Users/louis/workspace/ultra_louis_work/ultra6/runs/yoloa/26m_mergedata_v3/weights/best.pt"
BASE_MODEL="yolo26m.pt"  # ← set to a local path or a YOLOv8m variant from the hub (e.g. "yolo26m.pt") to run with that backbone


# ========================================================================================================================
#   Anomaly val  (mode=yoloa, ad_conf=0.5, support_cap=2000)
# ========================================================================================================================
#   name                           n_val    mAP10    mAP25    mAP50   mAP50-95        P        R  img_auroc  pix_auroc
#   ------------------------------------------------------------------------------------------------------------------
#   leather/yoloa                    124   0.5495   0.4062   0.2469     0.1408   0.8622   0.4842     0.7857     0.8164
#   grid/yoloa                        78   0.1251   0.0658   0.0658     0.0322   0.2053   0.0408     0.8170     0.8767
#   tile/yoloa                       117   0.3683   0.3260   0.2440     0.1999   0.3823   0.4070     0.8438     0.6341
#   wood/yoloa                        79   0.5853   0.3968   0.2191     0.1059   0.7676   0.4720     0.9509     0.8424
#   carpet/yoloa                     117   0.5052   0.3011   0.2208     0.1295   0.6199   0.5213     0.8824     0.8267
#   cable/yoloa                      150   0.2972   0.2481   0.1632     0.0764   0.5495   0.3104     0.6904     0.6903
#   hazelnut/yoloa                   110   0.3787   0.3250   0.2542     0.1327   0.3956   0.4049     0.7907     0.9573
#   pill/yoloa                       167   0.2608   0.2061   0.1457     0.1022   0.3989   0.2678     0.7679     0.8866
#   screw/yoloa                      160   0.1441   0.1441   0.1110     0.0359   0.5404   0.0465     0.7186     0.9595
#   metal_nut/yoloa                  115   0.1989   0.1074   0.0587     0.0306   0.3380   0.3000     0.7757     0.8661
#   capsule/yoloa                    132   0.4216   0.2471   0.1147     0.0507   0.7010   0.3211     0.7734     0.9597
#   bottle/yoloa                      83   0.4015   0.3330   0.1349     0.0431   0.4971   0.4091     0.8587     0.9110
#   transistor/yoloa                 100   0.0800   0.0694   0.0068     0.0068   0.1867   0.1395     0.6504     0.6640
#   zipper/yoloa                     151   0.4215   0.3411   0.1415     0.0384   0.7784   0.2874     0.7088     0.9147
#   ------------------------------------------------------------------------------------------------------------------
#   AVERAGE                         1683   0.3384   0.2512   0.1519     0.0804   0.5159   0.3151     0.7867     0.8433





# BASE_MODEL="yolo26m.pt"
OUT_CSV    = Path("./runs/temp/val_anomaly.csv")


# ── Dataset registry ───────────────────────────────────────────────────────
# Both MVTec and DAGM_AD are in the same on-disk layout (train/good, test/{good,defect},
# sibling .txt labels, per-class .yaml), so they share one resolver via
# ``get_mvtec_yolo_data``.  Adding another MVTec-format dataset = add a list of
# category names and one entry here.

DATASETS: dict[str, list[str]] = {
	"mvtec": MVTEC_CATEGORIES,
	"dagm":  DAGM_CATEGORIES,
}


def _resolve(cat: str) -> dict:
	"""Resolve any category (case-insensitive) → paths via ``get_mvtec_yolo_data``."""
	all_cats = MVTEC_CATEGORIES + DAGM_CATEGORIES
	canon = {c.lower(): c for c in all_cats}.get(cat.lower())
	if canon is None:
		raise KeyError(f"Unknown category '{cat}'. Valid: {all_cats}")
	data = get_mvtec_yolo_data(canon)
	return {
		"name":    canon,
		"yaml":    Path(data["data_yaml"]),
		"support": data["train_im_list"],
		"n_val":   len(data["test_im_list"]),
		"n_anom":  len(data["test_anomaly_im_list"]),
	}


# ── CLI plumbing ──────────────────────────────────────────────────────────

def _parse_categories(args) -> list[str]:
	"""Return a flat list of category names.

	- ``--all <dataset>``  expands to every category in that dataset (mvtec / dagm).
	- ``--category <name>[,...]``  lists explicit entries.  Either bare names
	  (``leather``, ``Class1``) or dataset-prefixed (``mvtec-leather``, ``dagm-Class1``);
	  the prefix is stripped — disambiguation isn't needed because MVTec & DAGM names
	  are disjoint.
	"""
	out: list[str] = []
	if args.all:
		for ds in [s.strip().lower() for s in args.all.split(",") if s.strip()]:
			if ds not in DATASETS:
				raise SystemExit(f"Unknown dataset '{ds}'. Valid: {list(DATASETS)}")
			out.extend(DATASETS[ds])
	if args.category:
		for spec in [s.strip() for s in args.category.split(",") if s.strip()]:
			# Accept "mvtec-leather" or just "leather" — strip optional dataset prefix.
			name = spec.split("-", 1)[1] if spec.lower().startswith(("mvtec-", "dagm-")) else spec
			out.append(name)
	if not out:
		raise SystemExit("Pass either --all <dataset> or --category <name>.")
	return out


def main():
	p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
	p.add_argument("--category",  default=None,
	               help="Comma-separated <dataset>-<name>, e.g. 'mvtec-leather,dagm-class1'")
	p.add_argument("--all",       default=None,
	               help="Dataset key(s), e.g. 'mvtec' or 'dagm' or 'mvtec,dagm'")
	p.add_argument("--mode",      choices=["yolo", "yoloa", "both"], default="both")
	p.add_argument("--base-model", default=BASE_MODEL)
	p.add_argument("--csv",       default=str(OUT_CSV))
	p.add_argument("--ad-conf",   type=float, default=0.5,
	               help="YOLOAnomaly per_level ad_conf threshold (default 0.60)")
	p.add_argument("--support-cap", type=int, default=2000,
	               help="Max support images per category for YOLOAnomaly bank (default 200)")
	p.add_argument("--support-batch", type=int, default=1,
	               help="Batch size for memory-bank build (backbone forward); default 1")
	p.add_argument("--max-bank-size", type=int, default=None,
	               help="If set, switch to coreset fast-path: skip OBMA filtering and "
	                    "k-center compress bank to this size at freeze time.")
	p.add_argument("--bb-heatmap-layers", type=int, nargs="+", default=None,
	               help="Backbone layer indices to tap for fused anomaly heatmap (e.g. '4').")
	p.add_argument("--bb-heatmap-channels", type=int, nargs="+", default=None,
	               help="Channel counts for each tapped layer (must align with --bb-heatmap-layers).")
	p.add_argument("--accumulate-thresh", type=float, default=None,
	               help="OBMA accept threshold (default from get_arguments=0.3). Raise to 0.5-0.7 "
	                    "when using diverse backbone features so the bank doesn't blow up.")
	p.add_argument("--no-auto-temperature", action="store_true",
	               help="Disable auto temperature calibration; pin β at the initial value (3.0).")
	p.add_argument("--score-filter-kernel", type=int, default=None,
	               help="Spatial avg-pool kernel applied to features before scoring "
	                    "(1=off, 3/5/7=smooth). Helps when features are too diverse (e.g. backbone L4).")
	p.add_argument("--imgsz", type=int, default=640,
	               help="Inference image size for both build and val (default 640).")
	p.add_argument("--rebuild", action="store_true",
	               help="Force rebuild of cached YOLOAnomaly banks (default: reuse cache)")
	args = p.parse_args()

	names = _parse_categories(args)
	print(f"Categories : {names}")
	print(f"Mode       : {args.mode}")
	print(f"Base model : {args.base_model}")

	build_overrides = dict(
		feature_mode  = "per_level",
		ad_conf       = args.ad_conf,
		active_layers = [0, 1, 2],
		return_heatmap= False,
	)
	if args.max_bank_size is not None:
		build_overrides["max_bank_size"] = args.max_bank_size
	if args.accumulate_thresh is not None:
		build_overrides["accumulate_thresh"] = args.accumulate_thresh
	if args.no_auto_temperature:
		build_overrides["auto_temperature"] = False
		build_overrides["temperature"] = 3.0
	if args.score_filter_kernel is not None:
		build_overrides["score_filter_kernel"] = args.score_filter_kernel

	rows = []
	for name in names:
		try:
			info = _resolve(name)
		except KeyError as e:
			print(f"[{name}] {e}"); continue

		tag = info["name"]  # canonical-cased
		print(f"\n=== {tag}: {len(info['support'])} train, "
		      f"{info['n_val']} val ({info['n_anom']} anom) ===")

		if args.mode in ("yolo", "both"):
			try:
				m = val_plain_yolo(info["yaml"], base_model=args.base_model)
				rows.append({"name": f"{tag}/yolo", "n_val": info["n_val"], **m})
			except Exception as e:  # noqa: BLE001
				print(f"[{tag}/yolo]  FAILED: {e!r}")

		if args.mode in ("yoloa", "both"):
			try:
				bb_heatmap = None
				if args.bb_heatmap_layers is not None:
					assert args.bb_heatmap_channels is not None and \
						len(args.bb_heatmap_channels) == len(args.bb_heatmap_layers), \
						"--bb-heatmap-channels must align with --bb-heatmap-layers"
					bb_heatmap = (list(args.bb_heatmap_layers), list(args.bb_heatmap_channels))
				m = val_yoloa(info["yaml"],
				              base_model=args.base_model,
				              support_imgs=info["support"],
				              build_overrides=build_overrides,
				              support_cap=args.support_cap,
				              support_batch=args.support_batch,
				              bb_heatmap=bb_heatmap,
				              imgsz=args.imgsz,
				              category=info["name"], rebuild=args.rebuild)
				rows.append({"name": f"{tag}/yoloa", "n_val": info["n_val"], **m})
			except Exception as e:  # noqa: BLE001
				print(f"[{tag}/yoloa] FAILED: {e!r}")

	if not rows:
		print("\nNo rows produced.")
		return

	title = (f"Anomaly val  (mode={args.mode}, ad_conf={args.ad_conf}, "
	         f"support_cap={args.support_cap})")
	print_metric_table(rows, title=title, name_width=28)
	saved = save_metric_csv(rows, args.csv)
	print(f"\nSaved → {saved}")


if __name__ == "__main__":
	main()
