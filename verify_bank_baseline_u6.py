"""Verify bank baseline on ultra6 — 5 cats, bank-only heatmap.

Matches run_yoloa.py's exact path: full training set, coreset 10k, heat_edge=True.
Run: python verify_bank_baseline_u6.py
"""
import sys; sys.path.insert(0, ".")
import logging; logging.getLogger("ultralytics").setLevel(logging.WARNING)
from pathlib import Path
import json, time, torch
from ultralytics.yoloa import YOLOA
from ultralytics.models.yolo.anomaly_v2.val import resolve_mvtec_root, run_mvtec_ood_eval

CKPT = "/home/louis/ultra_louis_work/expman/data/pulled/yoloa_clean/26m_yoloav2_softhint_maskonly_aug3_mixup_ood_aug2x_ep15_lr2x_v1/weights/best.pt"

DEVICE = "cuda:0"
IMGSZ = 640
CATS = ["bottle", "cable", "screw", "zipper", "toothbrush"]

ROOT = resolve_mvtec_root(None)
print(f"ultra6 bank baseline | device={DEVICE} | cats={CATS}", flush=True)
print(f"ckpt: {CKPT}", flush=True)
print(f"mvtec_root: {ROOT}", flush=True)

results = {}
for cat in CATS:
    gd = ROOT / cat / "train" / "good" if ROOT else None
    if not gd.is_dir():
        gd = ROOT / cat / "train" if ROOT else None
    if not gd or not gd.is_dir():
        print(f"  {cat:15s} SKIP — no train/good dir", flush=True)
        continue
    t0 = time.time()
    m = YOLOA(CKPT)
    # Bank-only: full training set, edge_weight on (matches fit YAML)
    m.fit(gd, name=cat, imgsz=IMGSZ, max_images=0, refit=True,
          fit_decoder=False, fit_disc=False)
    m.model.set_prior_mode("heatmap")
    # heat_edge must be set BEFORE eval (run_mvtec_ood_eval checks model attr)
    m.model.heatmap_edge_weight = True
    rows = run_mvtec_ood_eval(m.model, ROOT, categories=[cat],
                              modes=("heatmap",), imgsz=IMGSZ,
                              batch=4, device=DEVICE,
                              heatmap_edge_weight=True)
    d = [x for x in rows if x["category"] == cat and x.get("mode") == "heatmap"][0]
    r = {
        "mAP10": round(d["mAP10"], 4),
        "mAP25": round(d["mAP25"], 4),
        "mAP50": round(d["mAP50"], 4),
        "im_auroc": round(d["image_auroc"], 4),
        "px_auroc": round(d["pixel_auroc"], 4),
        "time_s": round(time.time() - t0, 1),
    }
    results[cat] = r
    print(f"  {cat:15s} mAP10={r['mAP10']:.4f} mAP25={r['mAP25']:.4f} mAP50={r['mAP50']:.4f}  "
          f"im_auroc={r['im_auroc']:.4f}  ({r['time_s']}s)", flush=True)

print("\n--- Summary ---", flush=True)
avg = {k: round(sum(r[k] for r in results.values()) / len(results), 4)
       for k in ["mAP10", "mAP25", "mAP50", "im_auroc"]}
print(f"AVG: mAP10={avg['mAP10']:.4f} mAP25={avg['mAP25']:.4f} mAP50={avg['mAP50']:.4f}  im_auroc={avg['im_auroc']:.4f}")

# Compare with hard-coded baselines
HARD = {
    "bottle":     {"mAP10": 0.8109, "mAP25": 0.6202, "mAP50": 0.1761},
    "cable":      {"mAP10": 0.5161, "mAP25": 0.3592, "mAP50": 0.0886},
    "screw":      {"mAP10": 0.3962, "mAP25": 0.0945, "mAP50": 0.0109},
    "zipper":     {"mAP10": 0.9493, "mAP25": 0.9218, "mAP50": 0.5346},
    "toothbrush": {"mAP10": 0.5393, "mAP25": 0.2961, "mAP50": 0.1173},
}
print("\n--- vs hard-coded baseline ---", flush=True)
for cat in CATS:
    h = HARD[cat]; r = results[cat]
    d10 = r["mAP10"] - h["mAP10"]
    d25 = r["mAP25"] - h["mAP25"]
    print(f"  {cat:15s} ΔmAP10={d10:+.4f}  ΔmAP25={d25:+.4f}", flush=True)
