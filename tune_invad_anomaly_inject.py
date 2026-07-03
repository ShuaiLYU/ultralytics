"""InvAD — synthetic anomaly injection grid search on best decoder config.

Builds on the best SSM config (ssm_r_dch512_nb2_sty32_bat_resinte):
decoder_ch=512, num_blocks=2, style_ch=32, residual_mode=inter_block,
norm_type=batch, mse+cosine, lr=0.001, steps=2000.

Sweeps anomaly_ratio × anomaly_lambda × anomaly_margin.
Eval with minmax calibration (best from Step 2).

Usage on ultra6 (1 GPU):
  CUDA_VISIBLE_DEVICES=6 nohup /home/louis/miniconda3/envs/ultra/bin/python \
    tune_invad_anomaly_inject.py --gpu 0 --total-gpus 1 \
    > runs/temp/invad_anj_tune.log 2>&1 &

Multi-GPU:
  for gpu in 6 7; do
    CUDA_VISIBLE_DEVICES=$gpu nohup ... --gpu $((gpu-6)) --total-gpus 2 \
      > runs/temp/invad_anj_tune_gpu$((gpu-6)).log 2>&1 &
  done
"""
import sys; sys.path.insert(0, ".")
import logging; logging.getLogger("ultralytics").setLevel(logging.WARNING)
from pathlib import Path
import argparse, csv, json, time, traceback, torch
from ultralytics.yoloa import YOLOA
from ultralytics.models.yolo.anomaly_v2.val import run_mvtec_ood_eval

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--total-gpus", type=int, default=1)
args = ap.parse_args()

GPU_INDEX = args.gpu
GPU_COUNT = args.total_gpus
DEVICE = "cuda:0"

CKPT = "/home/louis/ultra_louis_work/expman/data/pulled/yoloa_clean/26m_yoloav2_softhint_maskonly_aug3_mixup_ood_aug2x_ep15_lr2x_v1/weights/best.pt"
ROOT = Path("/data/shared-datasets/louis_data/MVTec-YOLO/MVTec-YOLO")
IMGSZ = 640
BB_LAYERS = [6]
CATS = ["bottle", "cable", "screw", "zipper", "toothbrush"]

BANK_BASELINE = {
    "bottle":     {"mAP10": 0.8092, "mAP25": 0.6410, "mAP50": 0.1930, "im_auroc": 0.9543, "px_auroc": 0.9166},
    "cable":      {"mAP10": 0.3601, "mAP25": 0.2546, "mAP50": 0.0588, "im_auroc": 0.9185, "px_auroc": 0.9232},
    "screw":      {"mAP10": 0.3289, "mAP25": 0.0836, "mAP50": 0.0144, "im_auroc": 0.8623, "px_auroc": 0.9372},
    "zipper":     {"mAP10": 0.9556, "mAP25": 0.9196, "mAP50": 0.5203, "im_auroc": 0.9842, "px_auroc": 0.9192},
    "toothbrush": {"mAP10": 0.5266, "mAP25": 0.2585, "mAP50": 0.1012, "im_auroc": 1.0000, "px_auroc": 0.9482},
}

# Best SSM config from Step 1 — fixed base, only anomaly params sweep
BASE_KW = {
    "arch": "ssm", "batch": 32, "decoder_ch": 512, "num_blocks": 2,
    "style_ch": 32, "residual_mode": "inter_block", "norm_type": "batch",
    "loss_mode": "mse+cosine", "lr": 0.001, "steps": 2000,
}

# Baseline (no injection) for comparison
BASELINE_NAME = "anj_baseline"

# Anomaly injection sweep
def make_configs():
    cfgs = []
    # Baseline: no injection
    cfgs.append((BASELINE_NAME, dict(BASE_KW)))

    # Sweep anomaly_ratio × lambda × margin
    for ar in [0.1, 0.3, 0.5]:
        for al in [0.1, 0.5, 1.0]:
            for am in [0.1, 0.3, 0.5]:
                d = dict(BASE_KW, anomaly_ratio=ar, anomaly_lambda=al, anomaly_margin=am)
                cfgs.append((f"anj_r{ar}_l{al}_m{am}", d))

    # Mode-specific variations (fixed ratio=0.3, lambda=0.5, margin=0.3)
    mode_variants = [
        (["gaussian_noise"], "anj_gau"),
        (["cutpaste"], "anj_cut"),
        (["channel_dropout"], "anj_chd"),
        (["spatial_shift"], "anj_shf"),
        (["gaussian_noise", "cutpaste", "channel_dropout"], "anj_all4"),
    ]
    for modes, tag in mode_variants:
        d = dict(BASE_KW, anomaly_ratio=0.3, anomaly_lambda=0.5, anomaly_margin=0.3,
                 anomaly_modes=modes)
        cfgs.append((tag, d))

    return cfgs


CONFIGS = make_configs()
my_configs = [(n, kw) for i, (n, kw) in enumerate(CONFIGS) if i % GPU_COUNT == GPU_INDEX]

print(f"[gpu{GPU_INDEX}/{GPU_COUNT}] {len(my_configs)} configs × {len(CATS)} cats", flush=True)

SUFFIX = f"_gpu{GPU_INDEX}"
OUT_CSV = Path(f"runs/temp/invad_anj_tune{SUFFIX}.csv")
OUT_PROG = Path(f"runs/temp/invad_anj_progress{SUFFIX}.json")
LOG = Path(f"runs/temp/invad_anj_tune{SUFFIX}.log")
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] g{GPU_INDEX} {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

if not OUT_PROG.exists():
    LOG.write_text("")

log(f"Anomaly injection tune | {len(my_configs)} configs × {len(CATS)} cats | device={DEVICE}")
log(f"Base: decoder_ch=512 nb=2 style_ch=32 inter_block batch mse+cosine lr=0.001 steps=2000")

done = set()
if OUT_PROG.exists():
    try:
        d = json.loads(OUT_PROG.read_text())
        done = set(d.get("done", []))
        log(f"Resuming: {len(done)} done")
    except Exception:
        pass

fieldnames = ["category", "config", "anomaly_ratio", "anomaly_lambda", "anomaly_margin",
              "anomaly_modes", "anomaly_noise_std",
              "invad_px_auroc", "invad_im_auroc",
              "invad_mAP10", "invad_mAP25", "invad_mAP50",
              "bank_mAP10", "bank_px_auroc",
              "delta_mAP10", "delta_px_auroc"]

if not OUT_CSV.exists():
    with open(OUT_CSV, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

for cfg_name, kw in my_configs:
    cfg_cats_done = sum(1 for cat in CATS if f"{cfg_name}_{cat}" in done)
    if cfg_cats_done == len(CATS):
        log(f"\n--- {cfg_name} [SKIP: all done] ---")
        continue

    log(f"\n--- {cfg_name} ({cfg_cats_done}/{len(CATS)} done) {kw} ---")

    for cat in CATS:
        key = f"{cfg_name}_{cat}"
        if key in done:
            log(f"  [SKIP] {cat}")
            continue

        log(f"  {cat}...")
        try:
            m = YOLOA(CKPT)
            m.model.to(DEVICE)
            gd = str(ROOT / cat / "train/good")

            m.fit(gd, name=cat, imgsz=IMGSZ, max_images=0, refit=True,
                  bb_layers=BB_LAYERS, fit_decoder=kw)

            dec = m.model._feat_inv_decoder
            # Set minmax calibration (best from Step 2)
            dec._gamma = 1.0
            dec._cal_mode = "minmax"

            m.model.set_prior_mode("heatmap_reconstruct")
            d_rows = run_mvtec_ood_eval(m.model, ROOT, categories=[cat],
                                        modes=("heatmap_reconstruct",), imgsz=IMGSZ,
                                        batch=4, device=DEVICE)
            d = [x for x in d_rows if x["category"] == cat and x.get("mode") == "heatmap_reconstruct"][0]
            b = BANK_BASELINE[cat]
            row = {
                "category": cat, "config": cfg_name,
                "anomaly_ratio": kw.get("anomaly_ratio", 0),
                "anomaly_lambda": kw.get("anomaly_lambda", 0),
                "anomaly_margin": kw.get("anomaly_margin", 0),
                "anomaly_modes": str(kw.get("anomaly_modes", "")),
                "anomaly_noise_std": kw.get("anomaly_noise_std", 0.1),
                "invad_px_auroc": round(d["pixel_auroc"], 4),
                "invad_im_auroc": round(d["image_auroc"], 4),
                "invad_mAP10": round(d["mAP10"], 4),
                "invad_mAP25": round(d["mAP25"], 4),
                "invad_mAP50": round(d["mAP50"], 4),
                "bank_mAP10": b["mAP10"],
                "bank_px_auroc": b["px_auroc"],
                "delta_mAP10": round(d["mAP10"] - b["mAP10"], 4),
                "delta_px_auroc": round(d["pixel_auroc"] - b["px_auroc"], 4),
            }
            log(f"    px={row['invad_px_auroc']:.4f}  mAP10={row['invad_mAP10']:.4f}"
                f" (Δ{row['delta_mAP10']:+.4f})")

            with open(OUT_CSV, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writerow(row)
            done.add(key)
            OUT_PROG.write_text(json.dumps({"done": list(done)}))

        except Exception as e:
            log(f"  FAILED: {e}")
            traceback.print_exc()
            continue

log(f"\nDONE — {OUT_CSV}")
