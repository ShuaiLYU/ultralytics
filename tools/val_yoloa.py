"""Run model.val() for cached YOLOAnomaly models loaded by config + category.

LOAD-ONLY: expects ``tools/build_yoloa.py --config <yaml>`` to have populated the cache.

Usage:
    python tools/val_yoloa.py --config tools/configs/yoloa_l6_320.yaml --category mvtec-leather
    python tools/val_yoloa.py --config tools/configs/yoloa_l6_320.yaml --all mvtec --mode both
    python tools/val_yoloa.py --config tools/configs/yoloa_l6_320.yaml --category mvtec-cable --ad-conf 0.3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _anomaly_common import (
	DEFAULT_OUT_CSV, add_category_args, cache_dir_for, load_config,
	parse_categories, resolve_category,
)

from ultra_ext.yoloa import (
	extract_metrics, load_yoloa, print_metric_table, save_metric_csv, val_plain_yolo,
)


def main():
	p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
	add_category_args(p)
	p.add_argument("--mode", choices=["yolo", "yoloa", "both"], default="both")
	p.add_argument("--csv", default=str(DEFAULT_OUT_CSV))
	p.add_argument("--ad-conf", type=float, default=None,
	               help="Override config's anomaly_arg.ad_conf at load time.")
	args = p.parse_args()

	cfg = load_config(args.config)
	cache_dir = cache_dir_for(cfg)
	val_kw = dict(cfg["model_arg"])     # forwarded to model.val()
	val_kw.pop("agnostic_nms", None)    # val handles NMS internally; not a val() kwarg
	names = parse_categories(args)

	print(f"Config     : {cfg['_path']}")
	print(f"Cache dir  : {cache_dir}")
	print(f"Mode       : {args.mode}    imgsz={val_kw['imgsz']}")
	print(f"Categories : {names}")

	rows = []
	for name in names:
		try:
			info = resolve_category(name)
		except KeyError as e:
			print(f"[{name}] {e}"); continue

		tag = info["name"]
		print(f"\n=== {tag}: {info['n_val']} val ({info['n_anom']} anom) ===")

		# Plain YOLO baseline
		if args.mode in ("yolo", "both"):
			try:
				m = val_plain_yolo(info["yaml"], base_model=cfg["base_model"])
				rows.append({"name": f"{tag}/yolo", "n_val": info["n_val"], **m})
			except Exception as e:  # noqa: BLE001
				print(f"[{tag}/yolo]  FAILED: {e!r}")

		# YOLOAnomaly: load cache → optional ad_conf override → val
		if args.mode in ("yoloa", "both"):
			try:
				model = load_yoloa(tag, cfg["base_model"], cache_dir=cache_dir)
				if args.ad_conf is not None:
					model.set_anomaly_args(ad_conf=args.ad_conf)
				res = model.val(data=str(info["yaml"]), split="val", batch=1, **val_kw)
				m = extract_metrics(res)
				rows.append({"name": f"{tag}/yoloa", "n_val": info["n_val"], **m})
			except FileNotFoundError as e:
				print(f"[{tag}/yoloa] {e}")
			except Exception as e:  # noqa: BLE001
				print(f"[{tag}/yoloa] FAILED: {e!r}")

	if not rows:
		print("\nNo rows produced.")
		return

	title = f"Anomaly val  (config={cfg['_name']}, mode={args.mode})"
	print_metric_table(rows, title=title, name_width=28)
	saved = save_metric_csv(rows, args.csv)
	print(f"\nSaved → {saved}")


if __name__ == "__main__":
	main()
