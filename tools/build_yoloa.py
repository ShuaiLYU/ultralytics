"""Build memory-bank YOLOAnomaly models from a YAML config and cache them.

Usage:
    python tools/build_yoloa.py --config tools/configs/yoloa_l6_320.yaml --category mvtec-leather
    python tools/build_yoloa.py --config tools/configs/yoloa_l6_320.yaml --all mvtec
    python tools/build_yoloa.py --config tools/configs/yoloa_l6_320.yaml --category mvtec-cable --rebuild

Cache path:  ``runs/temp/yoloa_cache/<yaml_stem>/<base_stem>_<category>.pt``
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _anomaly_common import (
	add_category_args, anomaly_arg_for_set, bb_heatmap_for, cache_dir_for,
	load_config, parse_categories, resolve_category,
)

from ultra_ext.yoloa import get_or_build_yoloa, yoloa_cache_path


def main():
	p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
	add_category_args(p)
	p.add_argument("--rebuild", action="store_true",
	               help="Force rebuild even if the cache file exists.")
	args = p.parse_args()

	cfg = load_config(args.config)
	cache_dir = cache_dir_for(cfg)
	cache_dir.mkdir(parents=True, exist_ok=True)
	build_overrides = anomaly_arg_for_set(cfg)
	bb_heatmap = bb_heatmap_for(cfg)
	imgsz = cfg["model_arg"]["imgsz"]
	build_cfg = cfg["build"]
	names = parse_categories(args)

	print(f"Config     : {cfg['_path']}")
	print(f"Cache dir  : {cache_dir}")
	print(f"Base model : {cfg['base_model']}")
	print(f"imgsz      : {imgsz}    rebuild={args.rebuild}")
	if bb_heatmap:
		print(f"bb_heatmap : layers={bb_heatmap[0]}  channels={bb_heatmap[1]}")
	print(f"Categories : {names}")

	for name in names:
		try:
			info = resolve_category(name)
		except KeyError as e:
			print(f"[{name}] {e}"); continue

		tag = info["name"]
		cache_path = yoloa_cache_path(tag, cfg["base_model"], cache_dir=cache_dir)
		print(f"\n=== {tag}: {len(info['support'])} train images ===")
		print(f"   cache: {cache_path}")

		try:
			_ = get_or_build_yoloa(
				category=tag,
				base_model=cfg["base_model"],
				support_imgs=info["support"],
				rebuild=args.rebuild,
				cache_dir=cache_dir,
				build_overrides=build_overrides,
				support_cap=build_cfg["support_cap"],
				support_batch=build_cfg["support_batch"],
				bb_heatmap=bb_heatmap,
				imgsz=imgsz,
				verbose=True,
			)
			print(f"   ✓ {tag}")
		except Exception as e:  # noqa: BLE001
			print(f"   ✗ {tag} FAILED: {e!r}")


if __name__ == "__main__":
	main()
