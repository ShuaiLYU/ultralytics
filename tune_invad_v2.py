"""InvAD v2 grid search — expanded hyperparameter space, multi-GPU.

Usage on ultra6 (4 GPUs, one worker per GPU):
  CUDA_VISIBLE_DEVICES=4 nohup python tune_invad_v2.py --gpu 0 --total-gpus 4 > runs/temp/invad_v2_gpu0.log 2>&1 &
  CUDA_VISIBLE_DEVICES=5 nohup python tune_invad_v2.py --gpu 1 --total-gpus 4 > runs/temp/invad_v2_gpu1.log 2>&1 &
  CUDA_VISIBLE_DEVICES=6 nohup python tune_invad_v2.py --gpu 2 --total-gpus 4 > runs/temp/invad_v2_gpu2.log 2>&1 &
  CUDA_VISIBLE_DEVICES=7 nohup python tune_invad_v2.py --gpu 3 --total-gpus 4 > runs/temp/invad_v2_gpu3.log 2>&1 &
"""
import sys; sys.path.insert(0, ".")
import logging; logging.getLogger("ultralytics").setLevel(logging.WARNING)
from pathlib import Path
import argparse, csv, json, math, time, numpy as np, traceback, os
from ultralytics.yoloa import YOLOA
from ultralytics.models.yolo.anomaly_v2.val import run_mvtec_ood_eval

# -- CLI ------------------------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--gpu", type=int, default=0, help="GPU index (0-based, within CUDA_VISIBLE_DEVICES)")
ap.add_argument("--total-gpus", type=int, default=4, help="Total GPU workers")
args = ap.parse_args()

GPU_RANK = args.gpu
GPU_COUNT = args.total_gpus
DEVICE = "cuda:0"  # CUDA_VISIBLE_DEVICES controls which physical GPU

# -- Paths & constants -----------------------------------------------------
CKPT = "/home/louis/ultra_louis_work/expman/data/pulled/yoloa_clean/26m_yoloav2_softhint_maskonly_aug3_mixup_ood_aug2x_ep15_lr2x_v1/weights/best.pt"
ROOT = Path("/data/shared-datasets/louis_data/MVTec-YOLO/MVTec-YOLO")
IMGSZ = 640
CATS = ["bottle", "cable", "screw", "zipper", "toothbrush"]

# Verified on ultra6 (2026-07-02, run_yoloa.py --mode val --prior heatmap)
BANK_BASELINE = {
    "bottle":     {"mAP10": 0.8087, "mAP25": 0.6408, "mAP50": 0.1929},
    "cable":      {"mAP10": 0.3599, "mAP25": 0.2543, "mAP50": 0.0591},
    "screw":      {"mAP10": 0.3273, "mAP25": 0.0830, "mAP50": 0.0143},
    "zipper":     {"mAP10": 0.9555, "mAP25": 0.9196, "mAP50": 0.5203},
    "toothbrush": {"mAP10": 0.5251, "mAP25": 0.2581, "mAP50": 0.1012},
}

# -- Config builder --------------------------------------------------------
def make_configs():
    """Generate expanded grid, return list of (name, kw) tuples."""
    cfgs = []

    # Core anchor (previous best): decoder_ch=256, num_blocks=4, lr=5e-4, steps=500
    base = {"num_blocks": 4, "decoder_ch": 256, "lr": 5e-4, "steps": 500, "style_ch": 64,
            "loss_mode": "mse", "bb_layers": [6]}

    # Dim 1: loss × style_ch × bb_layers  (main hypothesis test)
    for loss in ["mse", "cosine", "mse+cosine"]:
        for sty in [16, 32, 64, 128, 256]:
            for lyr in [[6], [4, 6], [6, 10]]:
                d = dict(base, loss_mode=loss, style_ch=sty, bb_layers=list(lyr))
                lyrs = "".join(str(x) for x in lyr)
                name = f"L{loss[:3]}_sty{sty}_ly{lyrs}"
                cfgs.append((name, d))

    # Dim 2: num_blocks × decoder_ch  (architecture depth, best settings from above unknown,
    #         so use sensible centre: mse+cosine, sty=64, ly=6)
    arch_base = dict(base, loss_mode="mse+cosine", style_ch=64, bb_layers=[6])
    for nb in [2, 4, 6, 8]:
        for dch in [64, 128, 256, 512]:
            if nb == 4 and dch == 256:
                continue  # already in dim 1
            d = dict(arch_base, num_blocks=nb, decoder_ch=dch)
            name = f"b{nb}_ch{dch}"
            cfgs.append((name, d))

    # Dim 3: steps × lr (training schedule, best arch from above unknown)
    train_base = dict(base, loss_mode="mse+cosine", style_ch=64, bb_layers=[6])
    for st in [512, 1024, 2048]:
        for lr in [2e-4, 5e-4, 1e-3, 2e-3]:
            if st == 500 and lr == 5e-4:
                continue  # in dim 1
            d = dict(train_base, steps=st, lr=lr)
            name = f"s{st}_lr{lr}"
            cfgs.append((name, d))

    return cfgs


CONFIGS = make_configs()
# Partition configs: each GPU gets configs where index % GPU_COUNT == GPU_RANK
my_configs = [(name, kw) for i, (name, kw) in enumerate(CONFIGS) if i % GPU_COUNT == GPU_RANK]

print(f"[gpu{GPU_RANK}/{GPU_COUNT}] total configs: {len(CONFIGS)}, my share: {len(my_configs)}", flush=True)

# -- Output paths ----------------------------------------------------------
SUFFIX = f"_gpu{GPU_RANK}"
OUT_CSV = Path(f"runs/temp/invad_v2_tune{SUFFIX}.csv")
OUT_PROG = Path(f"runs/temp/invad_v2_progress{SUFFIX}.json")
LOG = Path(f"runs/temp/invad_v2_tune{SUFFIX}.log")
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] gpu{GPU_RANK} {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

# Clear log on fresh start
if not OUT_PROG.exists():
    LOG.write_text("")

log(f"InvAD v2 tune | {len(my_configs)} configs × {len(CATS)} cats | device={DEVICE}")
log(f"Checkpoints: {Path(CKPT).parent.parent.name}/{Path(CKPT).parent.name}")

# -- Load progress ---------------------------------------------------------
done = set()
if OUT_PROG.exists():
    try:
        d = json.loads(OUT_PROG.read_text())
        done = set(d.get("done", []))
        log(f"Resuming: {len(done)} done")
    except Exception:
        pass

fieldnames = ["category", "config", "loss_mode", "style_ch", "bb_layers", "num_blocks",
              "decoder_ch", "steps", "lr",
              "bank_mAP10", "invad_mAP10", "delta_mAP10",
              "bank_mAP25", "invad_mAP25", "delta_mAP25",
              "bank_mAP50", "invad_mAP50", "delta_mAP50",
              "invad_im_auroc", "invad_px_auroc", "gamma"]
if not OUT_CSV.exists():
    with open(OUT_CSV, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

rows = []
for cfg_name, kw in my_configs:
    if cfg_name in done:
        continue

    log(f"\n--- {cfg_name}: {kw} ---")
    cat_results = {}

    for cat in CATS:
        key = f"{cfg_name}_{cat}"
        if key in done:
            log(f"  [SKIP] {cat}")
            continue

        log(f"  {cat}...")
        try:
            m = YOLOA(CKPT)
            gd = str(ROOT / cat / "train/good")

            # Build fit kwargs: split bb_layers (handled by m.fit) from decoder kwargs
            bb_layers = kw.pop("bb_layers", [6])
            decoder_kw = dict(kw)  # everything else goes to the decoder

            m.fit(gd, name=cat, imgsz=IMGSZ, max_images=100, refit=True,
                  bb_layers=list(bb_layers), fit_decoder=decoder_kw)

            # Put bb_layers back for logging
            kw["bb_layers"] = bb_layers

            dec = m.model._feat_inv_decoder
            gamma = dec._gamma if dec else float("nan")
            m.model.set_prior_mode("heatmap_reconstruct")
            d_rows = run_mvtec_ood_eval(m.model, ROOT, categories=[cat],
                                        modes=("heatmap_reconstruct",), imgsz=IMGSZ,
                                        batch=4, device=DEVICE)
            d = [x for x in d_rows if x["category"] == cat and x.get("mode") == "heatmap_reconstruct"][0]
            b = BANK_BASELINE[cat]
            row = {
                "category": cat, "config": cfg_name,
                "loss_mode": kw.get("loss_mode", "mse"),
                "style_ch": kw.get("style_ch", 64),
                "bb_layers": ",".join(str(x) for x in kw.get("bb_layers", [6])),
                "num_blocks": kw.get("num_blocks", 4),
                "decoder_ch": kw.get("decoder_ch", 256),
                "steps": kw.get("steps", 500),
                "lr": kw.get("lr", 5e-4),
                "bank_mAP10": b["mAP10"], "invad_mAP10": round(d["mAP10"], 4),
                "delta_mAP10": round(d["mAP10"] - b["mAP10"], 4),
                "bank_mAP25": b["mAP25"], "invad_mAP25": round(d["mAP25"], 4),
                "delta_mAP25": round(d["mAP25"] - b["mAP25"], 4),
                "bank_mAP50": b["mAP50"], "invad_mAP50": round(d["mAP50"], 4),
                "delta_mAP50": round(d["mAP50"] - b["mAP50"], 4),
                "invad_im_auroc": round(d["image_auroc"], 4),
                "invad_px_auroc": round(d["pixel_auroc"], 4),
                "gamma": round(gamma, 4),
            }
            cat_results[cat] = row
            log(f"    mAP10={row['invad_mAP10']:.4f} (Δ{row['delta_mAP10']:+.4f})  "
                f"mAP25={row['invad_mAP25']:.4f} (Δ{row['delta_mAP25']:+.4f})  "
                f"γ={row['gamma']:.4f}")
        except Exception as e:
            log(f"  FAILED: {e}")
            traceback.print_exc()
            continue

        rows.append(row)
        with open(OUT_CSV, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow(row)
        done.add(key)
        OUT_PROG.write_text(json.dumps({"done": list(done), "gpu": GPU_RANK}))

    if cat_results:
        avgs = {k: round(float(np.mean([r[k] for r in cat_results.values()])), 4)
                for k in ["invad_mAP10", "delta_mAP10", "invad_mAP25", "delta_mAP25"]}
        bb = BANK_BASELINE
        avg_b = round(float(np.mean([bb[c]["mAP10"] for c in CATS])), 4)
        log(f"  AVG: mAP10={avgs['invad_mAP10']} (Δ{avgs['delta_mAP10']:+.4f}) vs bank={avg_b}")

log(f"\nDONE — {len(rows)} rows saved to {OUT_CSV}")
