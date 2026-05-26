"""Generate visual analysis charts from a config's val_results.csv.

Reads ``<cache_dir>/val_results.csv`` (produced by ``tools/val_yoloa.py``)
and saves 6 individual PNGs plus a combined dashboard to
``<cache_dir>/analysis/``.

Usage:
    python tools/analyze_val.py --config tools/configs/yoloa_l6_320_yolo26m.yaml
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _anomaly_common import cache_dir_for, load_config, val_csv_for


# ── Read & parse CSV ───────────────────────────────────────────────────────

def _read_csv(path: Path) -> list[dict]:
	"""Return a list of row dicts.  Float columns are coerced; NaN preserved as math.nan."""
	rows: list[dict] = []
	with path.open() as f:
		for row in csv.DictReader(f):
			out = {"name": row["name"]}
			for k in ("image_auroc", "pixel_auroc", "map10", "map25", "map50",
			         "map50_95", "P", "R"):
				v = row.get(k, "")
				try:
					out[k] = float(v)
				except (TypeError, ValueError):
					out[k] = math.nan
			try:
				out["n_val"] = int(row.get("n_val", 0))
			except (TypeError, ValueError):
				out["n_val"] = 0
			out["category"] = row["name"].split("/")[0]
			out["variant"] = row["name"].split("/", 1)[1] if "/" in row["name"] else ""
			rows.append(out)
	return rows


def _grade_color(v: float, good: float = 0.95, ok: float = 0.9) -> str:
	"""Three-tier green/yellow/red colour by metric value (defaults tuned for AUROC)."""
	if math.isnan(v):
		return "#bbbbbb"
	if v >= good:
		return "#2ca02c"
	if v >= ok:
		return "#ff9f1c"
	return "#d62728"


# Per-metric grade thresholds.  AUROC stays at 0.95/0.90; mAP is lower-scale so uses 0.5/0.3.
GRADE_THRESHOLDS: dict[str, tuple[float, float]] = {
	"image_auroc": (0.95, 0.90),
	"pixel_auroc": (0.95, 0.90),
	"map10":       (0.50, 0.30),
	"map25":       (0.50, 0.30),
	"map50":       (0.50, 0.30),
}


def _nan_mean(values) -> float:
	finite = [v for v in values if not math.isnan(v)]
	return sum(finite) / len(finite) if finite else math.nan


# ── Individual charts ──────────────────────────────────────────────────────

def chart_metric_bar(rows, metric_key: str, title: str, ax=None):
	"""Horizontal bar chart sorted by metric descending, with AVG dashed line.

	Colour grade thresholds auto-pick from :data:`GRADE_THRESHOLDS` per metric.
	"""
	good, ok = GRADE_THRESHOLDS.get(metric_key, (0.95, 0.90))
	finite_rows = [(r["category"], r[metric_key]) for r in rows if not math.isnan(r[metric_key])]
	nan_rows = [r["category"] for r in rows if math.isnan(r[metric_key])]
	finite_rows.sort(key=lambda x: x[1], reverse=True)
	cats = [c for c, _ in finite_rows]
	vals = [v for _, v in finite_rows]
	avg = _nan_mean(vals)

	created_fig = ax is None
	if created_fig:
		fig, ax = plt.subplots(figsize=(10, max(4, 0.32 * len(rows) + 1.5)))
	bars = ax.barh(cats, vals, color=[_grade_color(v, good=good, ok=ok) for v in vals],
	               edgecolor="#333")
	for bar, v in zip(bars, vals):
		ax.text(min(v + 0.005, 1.005), bar.get_y() + bar.get_height() / 2,
		        f"{v:.4f}", va="center", fontsize=8)
	if not math.isnan(avg):
		ax.axvline(avg, color="#1f77b4", linestyle="--", linewidth=1.2,
		           label=f"AVG = {avg:.4f}")
		ax.legend(loc="lower right")
	ax.set_xlim(0, 1.05)
	ax.invert_yaxis()
	ax.set_xlabel(metric_key)
	ax.set_title(title + (f"  (excluded NaN: {', '.join(nan_rows)})" if nan_rows else ""))
	ax.grid(axis="x", alpha=0.3)
	if created_fig:
		plt.tight_layout()
		return fig


# Backwards-compat alias (old name)
chart_auroc_bar = chart_metric_bar


def chart_map_grouped(rows, ax=None):
	"""Grouped bar chart of map10 / map25 / map50 per category, sorted by map10 desc."""
	rows_sorted = sorted(rows, key=lambda r: (-r["map10"] if not math.isnan(r["map10"]) else 1e9, r["category"]))
	cats = [r["category"] for r in rows_sorted]
	m10 = [r["map10"] for r in rows_sorted]
	m25 = [r["map25"] for r in rows_sorted]
	m50 = [r["map50"] for r in rows_sorted]
	x = np.arange(len(cats))
	w = 0.27

	created_fig = ax is None
	if created_fig:
		fig, ax = plt.subplots(figsize=(max(10, 0.6 * len(cats) + 3), 5))
	ax.bar(x - w, m10, w, label="map10", color="#2ca02c")
	ax.bar(x,     m25, w, label="map25", color="#ff9f1c")
	ax.bar(x + w, m50, w, label="map50", color="#d62728")
	ax.set_xticks(x)
	ax.set_xticklabels(cats, rotation=35, ha="right")
	ax.set_ylim(0, 1.05)
	ax.set_ylabel("mAP")
	ax.set_title("mAP @ IoU=0.10 / 0.25 / 0.50  (sorted by map10)")
	ax.legend(loc="upper right")
	ax.grid(axis="y", alpha=0.3)
	if created_fig:
		plt.tight_layout()
		return fig


def chart_img_vs_pix_scatter(rows, ax=None):
	"""Scatter: img_auroc x pix_auroc.  Diagonal = parity reference."""
	created_fig = ax is None
	if created_fig:
		fig, ax = plt.subplots(figsize=(7, 7))
	for r in rows:
		x, y = r["image_auroc"], r["pixel_auroc"]
		if math.isnan(x) or math.isnan(y):
			continue
		ax.scatter(x, y, s=80, alpha=0.7, color=_grade_color(min(x, y)),
		           edgecolor="#333", linewidth=0.7)
		ax.annotate(r["category"], (x, y), xytext=(4, 4), textcoords="offset points",
		            fontsize=8)
	ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="img = pix")
	ax.set_xlim(0.4, 1.02)
	ax.set_ylim(0.4, 1.02)
	ax.set_xlabel("image_auroc")
	ax.set_ylabel("pixel_auroc")
	ax.set_title("image_auroc vs pixel_auroc  (per category)")
	ax.legend(loc="lower right")
	ax.grid(alpha=0.3)
	if created_fig:
		plt.tight_layout()
		return fig


def chart_pr_scatter(rows, ax=None):
	"""Scatter: P x R per category, point size = n_val."""
	created_fig = ax is None
	if created_fig:
		fig, ax = plt.subplots(figsize=(7, 7))
	for r in rows:
		p, recall, n = r["P"], r["R"], r["n_val"]
		if math.isnan(p) or math.isnan(recall):
			continue
		ax.scatter(recall, p, s=max(20, n * 1.5), alpha=0.55,
		           color=_grade_color(min(p, recall), good=0.7, ok=0.4),
		           edgecolor="#333", linewidth=0.7)
		ax.annotate(r["category"], (recall, p), xytext=(5, 5),
		            textcoords="offset points", fontsize=8)
	# F1 contour lines (P = f1*R / (2*R - f1) — invalid where R <= f1/2).
	rr = np.linspace(0.01, 1.0, 100)
	with np.errstate(divide="ignore", invalid="ignore"):
		for f1 in (0.2, 0.4, 0.6, 0.8):
			denom = 2 * rr - f1
			safe = np.where(denom == 0, 1, denom)
			pp = np.where(denom > 1e-9, f1 * rr / safe, np.nan)
			mask = (pp > 0) & (pp <= 1.05) & np.isfinite(pp)
			ax.plot(rr[mask], pp[mask], color="grey", linestyle=":", alpha=0.4)
			j = int(0.7 * mask.sum()) if mask.any() else 0
			if mask.any():
				ax.text(rr[mask][j], pp[mask][j], f"F1={f1}", color="grey", fontsize=7)
	ax.set_xlim(0, 1.05)
	ax.set_ylim(0, 1.05)
	ax.set_xlabel("Recall")
	ax.set_ylabel("Precision")
	ax.set_title("Precision vs Recall  (size ∝ n_val)")
	ax.grid(alpha=0.3)
	if created_fig:
		plt.tight_layout()
		return fig


def chart_summary_table(rows, ax=None):
	"""Text panel: stats + top/bottom categories."""
	created_fig = ax is None
	if created_fig:
		fig, ax = plt.subplots(figsize=(8, 6))
	ax.axis("off")

	def _stats(vals):
		finite = sorted([v for v in vals if not math.isnan(v)])
		if not finite:
			return None
		n = len(finite)
		return dict(
			mean=sum(finite) / n,
			median=finite[n // 2] if n % 2 else (finite[n // 2 - 1] + finite[n // 2]) / 2,
			std=(sum((v - sum(finite) / n) ** 2 for v in finite) / n) ** 0.5,
			min=finite[0],
			max=finite[-1],
			n=n,
		)

	stats_by_key = {
		"image_auroc": _stats([r["image_auroc"] for r in rows]),
		"pixel_auroc": _stats([r["pixel_auroc"] for r in rows]),
		"map10":       _stats([r["map10"] for r in rows]),
		"map25":       _stats([r["map25"] for r in rows]),
		"map50":       _stats([r["map50"] for r in rows]),
	}
	nans = [r["category"] for r in rows if math.isnan(r["image_auroc"]) or math.isnan(r["pixel_auroc"])]

	top_img = sorted([r for r in rows if not math.isnan(r["image_auroc"])],
	                 key=lambda r: r["image_auroc"], reverse=True)[:5]
	bot_img = sorted([r for r in rows if not math.isnan(r["image_auroc"])],
	                 key=lambda r: r["image_auroc"])[:5]

	lines = []
	lines.append("== Summary ==")
	for key, s in stats_by_key.items():
		if s is None:
			continue
		lines.append(f"{key:<12} n={s['n']:<3}  mean={s['mean']:.4f}  median={s['median']:.4f}  "
		             f"std={s['std']:.4f}  range=[{s['min']:.4f}, {s['max']:.4f}]")
	if nans:
		lines.append("")
		lines.append(f"NaN AUROC: {', '.join(nans)}")
	lines.append("")
	lines.append("Top 5 (image_auroc):")
	for r in top_img:
		lines.append(f"  {r['category']:<14} img={r['image_auroc']:.4f}  pix={r['pixel_auroc']:.4f}  "
		             f"m10={r['map10']:.4f}  m25={r['map25']:.4f}  m50={r['map50']:.4f}")
	lines.append("")
	lines.append("Bottom 5 (image_auroc):")
	for r in bot_img:
		lines.append(f"  {r['category']:<14} img={r['image_auroc']:.4f}  pix={r['pixel_auroc']:.4f}  "
		             f"m10={r['map10']:.4f}  m25={r['map25']:.4f}  m50={r['map50']:.4f}")

	ax.text(0.01, 0.99, "\n".join(lines), family="monospace", fontsize=9,
	        va="top", transform=ax.transAxes)
	if created_fig:
		plt.tight_layout()
		return fig


# ── Main ───────────────────────────────────────────────────────────────────

def main():
	p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
	p.add_argument("--config", required=True, help="Path to YAML config.")
	p.add_argument("--csv", default=None, help="Override CSV path (default: <cache_dir>/val_results.csv).")
	p.add_argument("--out-dir", default=None, help="Override output dir (default: <cache_dir>/analysis/).")
	args = p.parse_args()

	cfg = load_config(args.config)
	csv_path = Path(args.csv) if args.csv else val_csv_for(cfg)
	if not csv_path.exists():
		raise SystemExit(f"CSV not found: {csv_path}\nRun tools/val_yoloa.py first.")
	out_dir = Path(args.out_dir) if args.out_dir else (cache_dir_for(cfg) / "analysis")
	out_dir.mkdir(parents=True, exist_ok=True)

	rows = _read_csv(csv_path)
	# Filter to YOLOAnomaly rows only (drop the optional plain-YOLO rows).
	rows = [r for r in rows if r["variant"] in ("yoloa", "")]
	print(f"Config: {cfg['_name']}    rows: {len(rows)}    csv: {csv_path}")
	print(f"Output: {out_dir}")

	# ── Individual panels ──────────────────────────────────────────────────
	fig = chart_metric_bar(rows, "image_auroc", "Image AUROC")
	fig.savefig(out_dir / "01_img_auroc_bar.png", dpi=130, bbox_inches="tight")
	plt.close(fig)

	fig = chart_metric_bar(rows, "pixel_auroc", "Pixel AUROC")
	fig.savefig(out_dir / "02_pix_auroc_bar.png", dpi=130, bbox_inches="tight")
	plt.close(fig)

	fig = chart_metric_bar(rows, "map10", "mAP @ IoU=0.10")
	fig.savefig(out_dir / "03_map10_bar.png", dpi=130, bbox_inches="tight")
	plt.close(fig)

	fig = chart_metric_bar(rows, "map25", "mAP @ IoU=0.25")
	fig.savefig(out_dir / "04_map25_bar.png", dpi=130, bbox_inches="tight")
	plt.close(fig)

	fig = chart_metric_bar(rows, "map50", "mAP @ IoU=0.50")
	fig.savefig(out_dir / "05_map50_bar.png", dpi=130, bbox_inches="tight")
	plt.close(fig)

	fig = chart_map_grouped(rows)
	fig.savefig(out_dir / "06_map_grouped_bar.png", dpi=130, bbox_inches="tight")
	plt.close(fig)

	fig = chart_img_vs_pix_scatter(rows)
	fig.savefig(out_dir / "07_img_vs_pix_scatter.png", dpi=130, bbox_inches="tight")
	plt.close(fig)

	fig = chart_pr_scatter(rows)
	fig.savefig(out_dir / "08_pr_scatter.png", dpi=130, bbox_inches="tight")
	plt.close(fig)

	fig = chart_summary_table(rows)
	fig.savefig(out_dir / "09_summary_table.png", dpi=130, bbox_inches="tight")
	plt.close(fig)

	# ── Dashboard composite (4 rows × 2 cols, 9 panels filled) ────────────
	fig = plt.figure(figsize=(22, 26))
	gs = fig.add_gridspec(4, 2, hspace=0.45, wspace=0.22)
	chart_metric_bar(rows, "image_auroc", "Image AUROC",     ax=fig.add_subplot(gs[0, 0]))
	chart_metric_bar(rows, "pixel_auroc", "Pixel AUROC",     ax=fig.add_subplot(gs[0, 1]))
	chart_metric_bar(rows, "map10",       "mAP @ IoU=0.10",  ax=fig.add_subplot(gs[1, 0]))
	chart_metric_bar(rows, "map25",       "mAP @ IoU=0.25",  ax=fig.add_subplot(gs[1, 1]))
	chart_metric_bar(rows, "map50",       "mAP @ IoU=0.50",  ax=fig.add_subplot(gs[2, 0]))
	chart_map_grouped(rows,                                  ax=fig.add_subplot(gs[2, 1]))
	chart_img_vs_pix_scatter(rows,                           ax=fig.add_subplot(gs[3, 0]))
	chart_summary_table(rows,                                ax=fig.add_subplot(gs[3, 1]))
	fig.suptitle(f"YOLOAnomaly val analysis — config: {cfg['_name']}", fontsize=14, y=0.997)
	fig.savefig(out_dir / "dashboard.png", dpi=130, bbox_inches="tight")
	plt.close(fig)

	pngs = sorted(out_dir.glob("*.png"))
	print(f"\nSaved {len(pngs)} PNGs to {out_dir}/")
	for p in sorted(out_dir.glob("*.png")):
		print(f"  {p.name}")


if __name__ == "__main__":
	main()
