"""Shared helpers for ``tools/build_yoloa.py``, ``tools/val_yoloa.py``, ``tools/predict_yoloa.py``.

Three thin scripts split responsibility:
  * ``build_yoloa.py``   — build the memory bank, save to cache (expensive).
  * ``val_yoloa.py``     — load cache, run ``model.val()`` for metrics (cheap).
  * ``predict_yoloa.py`` — load cache, run ``model.predict()`` + save composites (cheap).

Build configuration lives in a YAML under ``tools/configs/``.  Schema:

    base_model: <path>
    model_arg:        # forwarded to model.val() / model.predict()
      imgsz, conf, iou, max_det, single_cls, rect, plots, verbose, agnostic_nms
    anomaly_arg:      # forwarded to set_anomaly_args (mirrors get_arguments() shape)
      mode, feature_mode, active_layers, ad_conf, ad_max_det, accumulate_thresh,
      auto_temperature, em_iters, calibration_interval, calibration_target_score,
      score_filter_kernel, max_bank_size, fused_layers, fused_use_pre_clshead,
      return_heatmap,
      bb_heatmap_layers, bb_heatmap_channels    # special — routed to enable_backbone_heatmap
    build:            # memory bank construction budget
      support_cap, support_batch

The YAML's filename stem is the cache key:
``runs/temp/yoloa_cache/<yaml_stem>/<base_stem>_<category>.pt``
"""
from __future__ import annotations

from pathlib import Path

import yaml

from ultra_ext.yoloa import MVTEC_CATEGORIES, DAGM_CATEGORIES, get_mvtec_yolo_data


# Where per-config caches live.  Each config gets a subdir under this root, and
# all artifacts (cached .pt files, val_results.csv, predict_<cat>/) live inside
# that subdir so a config is fully self-contained.
DEFAULT_CACHE_ROOT = Path("./runs/temp/yoloa_cache")


# ── Dataset registry ───────────────────────────────────────────────────────
DATASETS: dict[str, list[str]] = {
	"mvtec": MVTEC_CATEGORIES,
	"dagm":  DAGM_CATEGORIES,
}


def resolve_category(cat: str) -> dict:
	"""Resolve a category name (case-insensitive, optional ``mvtec-`` / ``dagm-`` prefix)
	to its on-disk paths via :func:`get_mvtec_yolo_data`."""
	if cat.lower().startswith(("mvtec-", "dagm-")):
		cat = cat.split("-", 1)[1]
	all_cats = MVTEC_CATEGORIES + DAGM_CATEGORIES
	canon = {c.lower(): c for c in all_cats}.get(cat.lower())
	if canon is None:
		raise KeyError(f"Unknown category '{cat}'. Valid: {all_cats}")
	data = get_mvtec_yolo_data(canon)
	return {
		"name":       canon,
		"yaml":       Path(data["data_yaml"]),
		"support":    data["train_im_list"],
		"test_im":    data["test_im_list"],
		"test_good":  data["test_good_im_list"],
		"test_anom":  data["test_anomaly_im_list"],
		"n_val":      len(data["test_im_list"]),
		"n_anom":     len(data["test_anomaly_im_list"]),
	}


def parse_categories(args) -> list[str]:
	"""Expand ``args.all`` (dataset key) and ``args.category`` to canonical names."""
	out: list[str] = []
	if args.all:
		for ds in [s.strip().lower() for s in args.all.split(",") if s.strip()]:
			if ds not in DATASETS:
				raise SystemExit(f"Unknown dataset '{ds}'. Valid: {list(DATASETS)}")
			out.extend(DATASETS[ds])
	if args.category:
		for spec in [s.strip() for s in args.category.split(",") if s.strip()]:
			name = spec.split("-", 1)[1] if spec.lower().startswith(("mvtec-", "dagm-")) else spec
			out.append(name)
	if not out:
		raise SystemExit("Pass either --all <dataset> or --category <name>.")
	return out


def add_category_args(p) -> None:
	"""Add ``--config`` / ``--category`` / ``--all`` to an argparse parser."""
	p.add_argument("--config", required=True,
	               help="Path to YAML config (see tools/configs/*.yaml).")
	p.add_argument("--category", default=None,
	               help="Comma-separated <dataset>-<name>, e.g. 'mvtec-leather,dagm-class1'")
	p.add_argument("--all", default=None,
	               help="Dataset key(s), e.g. 'mvtec' or 'dagm' or 'mvtec,dagm'")


# ── Config loading ─────────────────────────────────────────────────────────

# Fall-back defaults for top-level sections.
_MODEL_ARG_DEFAULTS = {
	"imgsz": 640, "conf": 0.001, "iou": 0.001, "max_det": 1000,
	"single_cls": True, "rect": False, "plots": False, "verbose": False,
	"agnostic_nms": True,
}
_ANOMALY_ARG_DEFAULTS: dict = {
	"mode": "anomaly",
	"feature_mode": "fused_heatmap",
	"active_layers": [0, 1, 2],
	"fused_layers": [0],
	"fused_use_pre_clshead": True,
	"return_heatmap": True,
	"ad_conf": 0.5,
	"ad_max_det": 10,
	"accumulate_thresh": 0.3,
	"auto_temperature": True,
	"em_iters": 1,
	"calibration_interval": 1,
	"calibration_target_score": 0.2,
	"score_filter_kernel": 1,
	"max_bank_size": None,
	"bb_heatmap_layers": None,
	"bb_heatmap_channels": None,
}
_BUILD_DEFAULTS = {"support_cap": 2000, "support_batch": 1}

# Keys inside anomaly_arg that are NOT forwarded to model.set_anomaly_args
# (they are handled by other model methods, e.g. enable_backbone_heatmap).
_ANOMALY_ARG_EXCLUDE = {"bb_heatmap_layers", "bb_heatmap_channels"}


def load_config(path: str | Path) -> dict:
	"""Parse YAML, merge defaults, and stash provenance fields ``_name`` / ``_path``.

	Required key: ``base_model``.  Missing ``model_arg`` / ``anomaly_arg`` / ``build``
	sections fall back to defaults.
	"""
	cfg_path = Path(path)
	with cfg_path.open() as f:
		raw = yaml.safe_load(f) or {}
	if "base_model" not in raw:
		raise SystemExit(f"{cfg_path}: missing required key 'base_model'")
	cfg = {
		"base_model": raw["base_model"],
		"model_arg":  {**_MODEL_ARG_DEFAULTS,   **(raw.get("model_arg")   or {})},
		"anomaly_arg":{**_ANOMALY_ARG_DEFAULTS, **(raw.get("anomaly_arg") or {})},
		"build":      {**_BUILD_DEFAULTS,       **(raw.get("build")       or {})},
		"_name":      cfg_path.stem,
		"_path":      str(cfg_path.resolve()),
	}
	return cfg


def cache_dir_for(cfg: dict) -> Path:
	"""Per-config cache subdir: ``runs/temp/yoloa_cache/<config_name>/``.

	This is where the cached ``.pt`` files live AND where val_yoloa.py writes
	``val_results.csv`` and predict_yoloa.py writes ``predict_<cat>/``.
	"""
	return DEFAULT_CACHE_ROOT / cfg["_name"]


def val_csv_for(cfg: dict) -> Path:
	"""Default CSV path for val metrics: ``<cache_dir>/val_results.csv``."""
	return cache_dir_for(cfg) / "val_results.csv"


def predict_dir_for(cfg: dict, category: str) -> Path:
	"""Default predict output dir: ``<cache_dir>/predict_<category>/``."""
	return cache_dir_for(cfg) / f"predict_{category}"


def anomaly_arg_for_set(cfg: dict) -> dict:
	"""Subset of ``cfg['anomaly_arg']`` safe to forward to ``model.set_anomaly_args``.

	Filters out keys handled by other methods (bb_heatmap_*).  Keys whose value is
	``None`` are also dropped so the head's own default applies.
	"""
	out = {}
	for k, v in cfg["anomaly_arg"].items():
		if k in _ANOMALY_ARG_EXCLUDE:
			continue
		if v is None:
			continue
		out[k] = v
	return out


def bb_heatmap_for(cfg: dict):
	"""Return ``(layers, channels)`` or ``None`` if not configured."""
	a = cfg["anomaly_arg"]
	layers = a.get("bb_heatmap_layers")
	channels = a.get("bb_heatmap_channels")
	if not layers:
		return None
	assert channels and len(channels) == len(layers), \
		"bb_heatmap_channels must align with bb_heatmap_layers"
	return (list(layers), list(channels))
