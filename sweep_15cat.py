"""15-category MVTec sweep: bank baseline vs InvAD decoder."""
import sys; sys.path.insert(0, ".")
import logging; logging.getLogger("ultralytics").setLevel(logging.WARNING)
from pathlib import Path
import math, csv, numpy as np
from ultralytics.yoloa import YOLOA
from ultralytics.models.yolo.anomaly_v2.val import run_mvtec_ood_eval, MVTEC_CATEGORIES

CKPT = "/Users/louis/workspace/ultra_louis_work/expman/data/pulled/yoloa_clean/26m_yoloav2_softhint_maskonly_aug3_mixup_ood_aug2x_ep15_lr2x_v1/weights/best.pt"
ROOT = Path("/Users/louis/workspace/ultra_louis_work/buffer/AnomalyData/MVTEC/MVTec-YOLO")
CATS = MVTEC_CATEGORIES  # 15 categories
DEVICE = "cpu"
IMGSZ = 640
N_TRAIN = 50
STEPS = 500
DECODER_KW = {"num_blocks": 4, "decoder_ch": 128, "steps": STEPS}
OUT = Path("runs/temp/invad_mvtec_15cat.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

rows = []
for ci, cat in enumerate(CATS, 1):
    print(f"\n[{ci}/{len(CATS)}] {cat}", flush=True)
    m = YOLOA(CKPT)
    gd = ROOT / cat / "train/good"
    if not gd.is_dir():
        print(f"  SKIP: no train dir", flush=True)
        continue

    # -- Bank baseline --
    print(f"  fitting bank...", flush=True, end=" ")
    m.fit(str(gd), name=cat, imgsz=IMGSZ, max_images=N_TRAIN, bb_max_bank_size=500)
    m.model.set_prior_mode("heatmap")
    b_rows = run_mvtec_ood_eval(m.model, ROOT, categories=[cat],
        modes=("heatmap",), imgsz=IMGSZ, batch=4, device=DEVICE)
    b = [x for x in b_rows if x["category"] == cat][0]

    # -- InvAD decoder --
    print(f"fitting decoder...", flush=True, end=" ")
    m.fit(str(gd), name=cat, imgsz=IMGSZ, max_images=N_TRAIN, bb_max_bank_size=500,
          refit=True, fit_decoder=DECODER_KW)
    dec = m.model._feat_inv_decoder
    print(f"gamma={dec._gamma:.3f}", flush=True, end=" ")
    m.model.set_prior_mode("heatmap_reconstruct")
    d_rows = run_mvtec_ood_eval(m.model, ROOT, categories=[cat],
        modes=("heatmap_reconstruct",), imgsz=IMGSZ, batch=4, device=DEVICE)
    d = [x for x in d_rows if x["category"] == cat][0]

    row = {
        "category": cat,
        "bank_im_auroc": b["image_auroc"], "bank_px_auroc": b["pixel_auroc"], "bank_mAP10": b["mAP10"],
        "invad_im_auroc": d["image_auroc"], "invad_px_auroc": d["pixel_auroc"], "invad_mAP10": d["mAP10"],
        "invad_gamma": dec._gamma,
    }
    rows.append(row)
    print(f"\n  bank:  im={b['image_auroc']:.4f} px={b['pixel_auroc']:.4f} mAP10={b['mAP10']:.4f}", flush=True)
    print(f"  invad: im={d['image_auroc']:.4f} px={d['pixel_auroc']:.4f} mAP10={d['mAP10']:.4f}", flush=True)

# Averages
avg = {"category": "AVERAGE"}
for k in ["bank_im_auroc", "bank_px_auroc", "bank_mAP10",
          "invad_im_auroc", "invad_px_auroc", "invad_mAP10", "invad_gamma"]:
    vals = [r[k] for r in rows if not math.isnan(r.get(k, math.nan))]
    avg[k] = float(np.mean(vals)) if vals else math.nan
rows.append(avg)

# Print summary
print(f"\n{'='*80}")
print(f"{'Category':15s} {'bank_im':>8s} {'invad_im':>8s} {'bank_px':>8s} {'invad_px':>8s} {'bank_mAP10':>10s} {'invad_mAP10':>10s} {'gamma':>6s}")
print(f"{'-'*80}")
for r in rows:
    print(f"{r['category']:15s} {r['bank_im_auroc']:8.4f} {r['invad_im_auroc']:8.4f} "
          f"{r['bank_px_auroc']:8.4f} {r['invad_px_auroc']:8.4f} "
          f"{r['bank_mAP10']:10.4f} {r['invad_mAP10']:10.4f} {r.get('invad_gamma',0):6.3f}")

# CSV
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["category", "bank_im_auroc", "bank_px_auroc", "bank_mAP10",
        "invad_im_auroc", "invad_px_auroc", "invad_mAP10", "invad_gamma"])
    w.writeheader()
    w.writerows(rows)
print(f"\nCSV -> {OUT}", flush=True)
print("DONE", flush=True)
