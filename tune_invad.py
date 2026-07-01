"""InvAD hyperparameter grid search across 5 categories. Saves results incrementally.

Check progress: cat runs/temp/invad_tune.csv  OR  tail -f runs/temp/invad_tune.log
"""
import sys; sys.path.insert(0, ".")
import logging; logging.getLogger("ultralytics").setLevel(logging.WARNING)
from pathlib import Path
import csv, json, math, time, numpy as np
from ultralytics.yoloa import YOLOA
from ultralytics.models.yolo.anomaly_v2.val import run_mvtec_ood_eval

CKPT = "/Users/louis/workspace/ultra_louis_work/expman/data/pulled/yoloa_clean/26m_yoloav2_softhint_maskonly_aug3_mixup_ood_aug2x_ep15_lr2x_v1/weights/best.pt"
ROOT = Path("/Users/louis/workspace/ultra_louis_work/buffer/AnomalyData/MVTEC/MVTec-YOLO")
DEVICE = "mps"
IMGSZ = 640

CATS = ["bottle", "cable", "screw", "zipper", "toothbrush"]
N_TRAIN = 100  # decoder training only; bank baselines use ALL images (like run_yoloa.py)

# Grid search space
CONFIGS = []
for s in [500, 1000, 2000]:
    CONFIGS.append((f"steps{s}_ch256_b4", {"num_blocks": 4, "decoder_ch": 256, "steps": s, "lr": 1e-3}))
for ch in [128, 512]:
    CONFIGS.append((f"ch{ch}_b4_s500", {"num_blocks": 4, "decoder_ch": ch, "steps": 500, "lr": 1e-3}))
for nb in [2, 6, 8]:
    CONFIGS.append((f"b{nb}_ch256_s500", {"num_blocks": nb, "decoder_ch": 256, "steps": 500, "lr": 1e-3}))
for lr in [2e-3, 5e-4, 2e-4]:
    CONFIGS.append((f"lr{lr}_b4_ch256_s500", {"num_blocks": 4, "decoder_ch": 256, "steps": 500, "lr": lr}))
CONFIGS.append(("s2000_ch128_b6_lr5e4", {"num_blocks": 6, "decoder_ch": 128, "steps": 2000, "lr": 5e-4}))
CONFIGS.append(("s2000_ch256_b6_lr5e4", {"num_blocks": 6, "decoder_ch": 256, "steps": 2000, "lr": 5e-4}))

OUT_CSV = Path("runs/temp/invad_tune.csv")
OUT_PROG = Path("runs/temp/invad_tune_progress.json")
LOG = Path("runs/temp/invad_tune.log")
BANK_JSON = Path("runs/temp/invad_tune_bank.json")
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

MAX_HOURS = 10
start_time = time.time()

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

# Clear log on fresh start
if not OUT_PROG.exists():
    LOG.write_text("")

log(f"=== InvAD tune start: {len(CATS)} cats x {len(CONFIGS)} configs ===")
log(f"CKPT: {Path(CKPT).parent.parent.name}/{Path(CKPT).parent.name}")
log(f"Cats: {CATS}")
log(f"Configs: {[c[0] for c in CONFIGS]}")

# Load existing progress
done = set()
if OUT_PROG.exists():
    try:
        done = set(json.loads(OUT_PROG.read_text()).get("done", []))
        log(f"Resuming: {len(done)} already done")
    except Exception:
        pass

# Write CSV header if new
fieldnames = ["category", "config",
              "bank_mAP10", "invad_mAP10", "delta_mAP10",
              "bank_mAP50", "invad_mAP50", "delta_mAP50",
              "invad_im_auroc", "invad_px_auroc", "gamma"]
if not OUT_CSV.exists():
    with open(OUT_CSV, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

# --- Bank baselines via run_mvtec_ood_eval (matches run_yoloa.py exactly) ---
bank_cache = {}
if BANK_JSON.exists():
    bank_cache = json.loads(BANK_JSON.read_text())
    log(f"Loaded bank baselines from cache: {list(bank_cache.keys())}")

if not all(c in bank_cache for c in CATS):
    log("Building bank baselines (ALL training images, like run_yoloa.py)...")
    m0 = YOLOA(CKPT)
    b_rows = run_mvtec_ood_eval(m0.model, ROOT, categories=[c for c in CATS if c not in bank_cache],
                                modes=("heatmap",), imgsz=IMGSZ, batch=4, device=DEVICE)
    for b in b_rows:
        if b["category"] in CATS and b.get("mode") == "heatmap":
            bank_cache[b["category"]] = {"mAP10": b["mAP10"], "mAP50": b["mAP50"]}
    BANK_JSON.write_text(json.dumps(bank_cache))
    del m0

log("Bank baselines (ALL training images):")
for cat in CATS:
    if cat in bank_cache:
        log(f"  {cat}: mAP10={bank_cache[cat]['mAP10']:.4f}  mAP50={bank_cache[cat]['mAP50']:.4f}")

rows = []

# --- Grid search ---
for cfg_name, kw in CONFIGS:
    if cfg_name in done:
        log(f"[SKIP] {cfg_name} (already done)")
        continue

    elapsed = (time.time() - start_time) / 3600
    if elapsed > MAX_HOURS:
        log(f"Budget exceeded ({elapsed:.1f}h > {MAX_HOURS}h), stopping.")
        break

    log(f"\n--- {cfg_name}: {kw} ---")
    cat_results = {}

    for cat in CATS:
        if cat not in bank_cache:
            continue

        key = f"{cfg_name}_{cat}"
        if key in done:
            log(f"  [SKIP] {key}")
            continue

        elapsed = (time.time() - start_time) / 3600
        if elapsed > MAX_HOURS:
            break

        log(f"  {cat}...")
        try:
            m = YOLOA(CKPT)
            m.fit(str(ROOT / cat / "train/good"), name=cat, imgsz=IMGSZ,
                  max_images=N_TRAIN, refit=True, fit_decoder=kw)
            dec = m.model._feat_inv_decoder
            gamma = dec._gamma if dec else float("nan")
            m.model.set_prior_mode("heatmap_reconstruct")
            d_rows = run_mvtec_ood_eval(m.model, ROOT, categories=[cat],
                                        modes=("heatmap_reconstruct",), imgsz=IMGSZ,
                                        batch=4, device=DEVICE)
            d = [x for x in d_rows if x["category"] == cat][0]
            row = {
                "category": cat, "config": cfg_name,
                "bank_mAP10": round(bank_cache[cat]["mAP10"], 4),
                "invad_mAP10": round(d["mAP10"], 4),
                "delta_mAP10": round(d["mAP10"] - bank_cache[cat]["mAP10"], 4),
                "bank_mAP50": round(bank_cache[cat]["mAP50"], 4),
                "invad_mAP50": round(d["mAP50"], 4),
                "delta_mAP50": round(d["mAP50"] - bank_cache[cat]["mAP50"], 4),
                "invad_im_auroc": round(d["image_auroc"], 4),
                "invad_px_auroc": round(d["pixel_auroc"], 4),
                "gamma": round(gamma, 4),
            }
            cat_results[cat] = row
            log(f"    invad mAP10={row['invad_mAP10']:.4f} (Δ{row['delta_mAP10']:+.4f})  "
                f"mAP50={row['invad_mAP50']:.4f} (Δ{row['delta_mAP50']:+.4f})  "
                f"gamma={row['gamma']:.4f}")
        except Exception as e:
            log(f"  FAILED: {e}")
            import traceback; traceback.print_exc()
            continue

        # Save incrementally after each category
        rows.append(row)
        with open(OUT_CSV, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow(row)

        done.add(key)
        OUT_PROG.write_text(json.dumps({"done": list(done), "elapsed_h": round(elapsed, 2)}))

        if elapsed > MAX_HOURS:
            break

    if cfg_name not in done and all(f"{cfg_name}_{c}" in done for c in CATS if c in bank_cache):
        done.add(cfg_name)

    if cat_results:
        avgs = {k: round(float(np.mean([r[k] for r in cat_results.values()])), 4)
                for k in ["invad_mAP10", "invad_mAP50", "delta_mAP10", "delta_mAP50",
                          "bank_mAP10", "bank_mAP50"]}
        log(f"  AVERAGE({cfg_name}): bank_mAP10={avgs['bank_mAP10']}  invad_mAP10={avgs['invad_mAP10']} "
            f"(Δ{avgs['delta_mAP10']:+.4f})  bank_mAP50={avgs['bank_mAP50']}  "
            f"invad_mAP50={avgs['invad_mAP50']} (Δ{avgs['delta_mAP50']:+.4f})")

# --- Final summary ---
log(f"\n{'='*80}")
log(f"FINAL SUMMARY ({len(rows)} rows, {len(CATS)} cats, elapsed={(time.time()-start_time)/3600:.1f}h)")
log(f"{'='*80}")

configs_seen = sorted(set(r["config"] for r in rows))
log(f"\n{'Config':30s} {'avg_bank_mAP10':>15s} {'avg_invad_mAP10':>16s} {'avg_Δ':>9s} "
    f"{'avg_bank_mAP50':>15s} {'avg_invad_mAP50':>16s} {'avg_Δ':>9s}")
log("-"*115)
for cfg in configs_seen:
    rd = [r for r in rows if r["config"] == cfg]
    if len(rd) < 2:
        continue
    a = {k: round(float(np.mean([r[k] for r in rd])), 4) for k in
         ["bank_mAP10", "invad_mAP10", "delta_mAP10", "bank_mAP50", "invad_mAP50", "delta_mAP50"]}
    log(f"{cfg:30s} {a['bank_mAP10']:15.4f} {a['invad_mAP10']:16.4f} {a['delta_mAP10']:+9.4f} "
        f"{a['bank_mAP50']:15.4f} {a['invad_mAP50']:16.4f} {a['delta_mAP50']:+9.4f}")

log(f"\nCSV -> {OUT_CSV}")
log("DONE")
