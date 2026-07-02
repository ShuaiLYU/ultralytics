"""InvAD v3 grid search — expanded architectures (SSM variants + UNet + Diffusion).

Key changes from v2:
  - max_images=0 (all training images, matches bank baseline)
  - batch=32 (was 8)
  - New archs: UNetFeatureDecoder, DiffusionFeatureDecoder
  - SSM now has residual_mode + norm_type
  - Multi-worker per GPU: --workers-per-gpu N --worker 0..N-1

Usage on ultra6 (6 GPUs × 3 workers = 18 slots):
  for gpu in {2..7}; do
    for w in {0..2}; do
      CUDA_VISIBLE_DEVICES=$gpu nohup /home/louis/miniconda3/envs/ultra/bin/python \
        tune_invad_v3.py --gpu $((gpu-2)) --total-gpus 6 --workers-per-gpu 3 --worker $w \
        > runs/temp/invad_v3_gpu$((gpu-2))_w$w.log 2>&1 &
    done
  done
"""
import sys; sys.path.insert(0, ".")
import logging; logging.getLogger("ultralytics").setLevel(logging.WARNING)
from pathlib import Path
import argparse, csv, json, math, random, time, numpy as np, traceback, os, torch
from ultralytics.yoloa import YOLOA
from ultralytics.models.yolo.anomaly_v2.val import run_mvtec_ood_eval

# -- CLI ------------------------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--gpu", type=int, default=0, help="Physical GPU index (within --total-gpus)")
ap.add_argument("--total-gpus", type=int, default=4, help="Number of physical GPUs in pool")
ap.add_argument("--workers-per-gpu", type=int, default=1, help="Workers per physical GPU")
ap.add_argument("--worker", type=int, default=0, help="Worker index within this physical GPU (0..workers_per_gpu-1)")
args = ap.parse_args()

GPU_INDEX = args.gpu          # physical GPU index
GPU_COUNT = args.total_gpus   # physical GPU count
WP_GPU   = args.workers_per_gpu
W_INDEX  = args.worker

# Effective rank for config partitioning across all worker slots
EFFECTIVE_RANK  = GPU_INDEX * WP_GPU + W_INDEX
EFFECTIVE_TOTAL = GPU_COUNT * WP_GPU

DEVICE = "cuda:0"

# -- Paths & constants -----------------------------------------------------
CKPT = "/home/louis/ultra_louis_work/expman/data/pulled/yoloa_clean/26m_yoloav2_softhint_maskonly_aug3_mixup_ood_aug2x_ep15_lr2x_v1/weights/best.pt"
ROOT = Path("/data/shared-datasets/louis_data/MVTec-YOLO/MVTec-YOLO")
IMGSZ = 640
CATS = ["bottle", "cable", "screw", "zipper", "toothbrush"]

# Verified on ultra6 (2026-07-02) with run_yoloa.py --mode val --prior heatmap
# using yoloa_fit_default.yaml: K=5, temp=5.0, calibration_target=0.4
BANK_BASELINE = {
    "bottle":     {"mAP10": 0.8092, "mAP25": 0.6410, "mAP50": 0.1930, "im_auroc": 0.9543, "px_auroc": 0.9166},
    "cable":      {"mAP10": 0.3601, "mAP25": 0.2546, "mAP50": 0.0588, "im_auroc": 0.9185, "px_auroc": 0.9232},
    "screw":      {"mAP10": 0.3289, "mAP25": 0.0836, "mAP50": 0.0144, "im_auroc": 0.8623, "px_auroc": 0.9372},
    "zipper":     {"mAP10": 0.9556, "mAP25": 0.9196, "mAP50": 0.5203, "im_auroc": 0.9842, "px_auroc": 0.9192},
    "toothbrush": {"mAP10": 0.5266, "mAP25": 0.2585, "mAP50": 0.1012, "im_auroc": 1.0000, "px_auroc": 0.9482},
}

BB_LAYERS = [6]

# -- Config builder --------------------------------------------------------
def make_configs():
    cfgs = []

    # ==================================================================
    # Block 1: SSM anchor — best settings from v2, sweep loss/lr/steps
    # ==================================================================
    ssm_base = {"arch": "ssm", "batch": 32, "decoder_ch": 256, "num_blocks": 4,
                "style_ch": 64, "residual_mode": "block", "norm_type": "instance"}
    for loss in ["mse", "cosine", "mse+cosine"]:
        for lr in [5e-4, 1e-3, 2e-3]:
            for st in [1000, 2000, 4000]:
                d = dict(ssm_base, loss_mode=loss, lr=lr, steps=st)
                loss_tag = loss[:3] if loss != "mse+cosine" else "msc"
                name = f"ssm_L{loss_tag}_lr{lr}_s{st}"
                cfgs.append((name, d))

    # ==================================================================
    # Block 2: SSM residual_mode × norm_type
    # ==================================================================
    ssm_res = dict(ssm_base, loss_mode="mse+cosine", lr=1e-3, steps=2000)
    for rm in ["block", "none", "inter_block", "dense"]:
        for nt in ["instance", "group", "batch"]:
            d = dict(ssm_res, residual_mode=rm, norm_type=nt)
            name = f"ssm_res{rm[:4]}_n{nt[:3]}"
            cfgs.append((name, d))

    # ==================================================================
    # Block 3: SSM architecture random (20 combos)
    # ==================================================================
    rng = random.Random(42)
    ssm_arch_base = dict(ssm_base, loss_mode="mse+cosine", lr=1e-3, steps=2000,
                         residual_mode="block")
    seen_random = set()
    while len(cfgs) < 39 + 20:  # 39 = first 3 blocks, 20 = random block size
        dch = rng.choice([128, 256, 512])
        nb = rng.choice([2, 4, 6, 8])
        sty = rng.choice([32, 64, 128, 256])
        nt = rng.choice(["instance", "group", "batch"])
        rm = rng.choice(["block", "none", "inter_block", "dense"])
        name = f"ssm_r_dch{dch}_nb{nb}_sty{sty}_{nt[:3]}_res{rm[:4]}"
        if name in seen_random:
            continue
        seen_random.add(name)
        d = dict(ssm_arch_base, decoder_ch=dch, num_blocks=nb, style_ch=sty,
                 norm_type=nt, residual_mode=rm)
        cfgs.append((name, d))

    # ==================================================================
    # Block 4: UNet decoder
    # ==================================================================
    unet_base = {"arch": "unet", "batch": 32, "loss_mode": "mse+cosine"}
    for base_ch in [64, 128]:
        for num_lv in [2, 3]:
            for st in [1000, 2000]:
                for lr in [5e-4, 1e-3]:
                    d = dict(unet_base, base_ch=base_ch, num_levels=num_lv,
                            steps=st, lr=lr)
                    name = f"unet_ch{base_ch}_lv{num_lv}_s{st}_lr{lr}"
                    cfgs.append((name, d))

    # ==================================================================
    # Block 5: Diffusion decoder
    # ==================================================================
    diff_base = {"arch": "diffusion", "batch": 16, "loss_mode": "mse+cosine",
                 "num_infer_steps": 5}
    for u_ch in [64, 128]:
        for n_diff in [5, 10]:
            for st in [1000, 2000]:
                for lr in [5e-4, 1e-3]:
                    d = dict(diff_base, unet_ch=u_ch, num_diff_steps=n_diff,
                            steps=st, lr=lr)
                    name = f"diff_u{u_ch}_t{n_diff}_s{st}_lr{lr}"
                    cfgs.append((name, d))

    return cfgs


CONFIGS = make_configs()
my_configs = [(name, kw) for i, (name, kw) in enumerate(CONFIGS) if i % EFFECTIVE_TOTAL == EFFECTIVE_RANK]

print(f"[gpu{GPU_INDEX}/{GPU_COUNT} w{W_INDEX}/{WP_GPU-1}] "
      f"effective rank {EFFECTIVE_RANK}/{EFFECTIVE_TOTAL} "
      f"| {len(my_configs)} configs × {len(CATS)} cats", flush=True)

# -- Output paths ----------------------------------------------------------
SUFFIX = f"_gpu{GPU_INDEX}_w{W_INDEX}"
OUT_CSV = Path(f"runs/temp/invad_v3_tune{SUFFIX}.csv")
OUT_PROG = Path(f"runs/temp/invad_v3_progress{SUFFIX}.json")
LOG = Path(f"runs/temp/invad_v3_tune{SUFFIX}.log")
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
DECODER_DIR = OUT_CSV.parent / "decoders"
DECODER_DIR.mkdir(parents=True, exist_ok=True)

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] g{GPU_INDEX}w{W_INDEX} {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

if not OUT_PROG.exists():
    LOG.write_text("")

log(f"InvAD v3 tune | {len(my_configs)} configs × {len(CATS)} cats | device={DEVICE}")
log(f"max_images=0 (all data), batch=32, bb_layers={BB_LAYERS}")

# -- Load progress ---------------------------------------------------------
done = set()
if OUT_PROG.exists():
    try:
        d = json.loads(OUT_PROG.read_text())
        done = set(d.get("done", []))
        log(f"Resuming: {len(done)} done")
    except Exception:
        pass

fieldnames = ["category", "config", "arch", "loss_mode", "lr", "steps",
              "decoder_ch", "style_ch", "num_blocks", "residual_mode", "norm_type",
              "base_ch", "num_levels", "unet_ch", "num_diff_steps", "num_infer_steps",
              "bank_mAP10", "invad_mAP10", "delta_mAP10",
              "bank_mAP25", "invad_mAP25", "delta_mAP25",
              "bank_mAP50", "invad_mAP50", "delta_mAP50",
              "bank_im_auroc", "invad_im_auroc", "delta_im_auroc",
              "bank_px_auroc", "invad_px_auroc", "delta_px_auroc",
              "gamma"]
if not OUT_CSV.exists():
    with open(OUT_CSV, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

rows = []
for cfg_name, kw in my_configs:
    if cfg_name in done:
        continue

    arch = kw.get("arch", "ssm")
    log(f"\n--- {cfg_name} [{arch}] {kw} ---")
    cat_results = {}

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
            gamma = dec._gamma if dec else float("nan")

            # Save decoder weights for Step 2 calibration sweep
            torch.save(dec.state_dict(), DECODER_DIR / f"{cfg_name}_{cat}.pt")

            m.model.set_prior_mode("heatmap_reconstruct")
            d_rows = run_mvtec_ood_eval(m.model, ROOT, categories=[cat],
                                        modes=("heatmap_reconstruct",), imgsz=IMGSZ,
                                        batch=4, device=DEVICE)
            d = [x for x in d_rows if x["category"] == cat and x.get("mode") == "heatmap_reconstruct"][0]
            b = BANK_BASELINE[cat]
            row = {
                "category": cat, "config": cfg_name, "arch": arch,
                "loss_mode": kw.get("loss_mode", ""),
                "lr": kw.get("lr", 0),
                "steps": kw.get("steps", 0),
                "decoder_ch": kw.get("decoder_ch", 0) if arch == "ssm" else "",
                "style_ch": kw.get("style_ch", 0) if arch == "ssm" else "",
                "num_blocks": kw.get("num_blocks", 0) if arch == "ssm" else "",
                "residual_mode": kw.get("residual_mode", "") if arch == "ssm" else "",
                "norm_type": kw.get("norm_type", "") if arch != "diffusion" else "",
                "base_ch": kw.get("base_ch", 0) if arch == "unet" else "",
                "num_levels": kw.get("num_levels", 0) if arch == "unet" else "",
                "unet_ch": kw.get("unet_ch", 0) if arch == "diffusion" else "",
                "num_diff_steps": kw.get("num_diff_steps", 0) if arch == "diffusion" else "",
                "num_infer_steps": kw.get("num_infer_steps", 0) if arch == "diffusion" else "",
                "bank_mAP10": b["mAP10"], "invad_mAP10": round(d["mAP10"], 4),
                "delta_mAP10": round(d["mAP10"] - b["mAP10"], 4),
                "bank_mAP25": b["mAP25"], "invad_mAP25": round(d["mAP25"], 4),
                "delta_mAP25": round(d["mAP25"] - b["mAP25"], 4),
                "bank_mAP50": b["mAP50"], "invad_mAP50": round(d["mAP50"], 4),
                "delta_mAP50": round(d["mAP50"] - b["mAP50"], 4),
                "bank_im_auroc": b["im_auroc"], "invad_im_auroc": round(d["image_auroc"], 4),
                "delta_im_auroc": round(d["image_auroc"] - b["im_auroc"], 4),
                "bank_px_auroc": b["px_auroc"], "invad_px_auroc": round(d["pixel_auroc"], 4),
                "delta_px_auroc": round(d["pixel_auroc"] - b["px_auroc"], 4),
                "gamma": round(gamma, 4),
            }
            cat_results[cat] = row
            log(f"    px_auroc={row['invad_px_auroc']:.4f} (Δ{row['delta_px_auroc']:+.4f})  "
                f"im_auroc={row['invad_im_auroc']:.4f} (Δ{row['delta_im_auroc']:+.4f})  "
                f"mAP10={row['invad_mAP10']:.4f} (Δ{row['delta_mAP10']:+.4f})")
        except Exception as e:
            log(f"  FAILED: {e}")
            traceback.print_exc()
            continue

        rows.append(row)
        with open(OUT_CSV, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow(row)
        done.add(key)
        OUT_PROG.write_text(json.dumps({"done": list(done), "gpu": GPU_INDEX, "worker": W_INDEX}))

    if cat_results:
        avgs = {k: round(float(np.mean([r[k] for r in cat_results.values()])), 4)
                for k in ["invad_px_auroc", "delta_px_auroc",
                          "invad_im_auroc", "delta_im_auroc",
                          "invad_mAP10", "delta_mAP10"]}
        avg_b_px = round(float(np.mean([BANK_BASELINE[c]["px_auroc"] for c in CATS])), 4)
        avg_b_m = round(float(np.mean([BANK_BASELINE[c]["mAP10"] for c in CATS])), 4)
        log(f"  AVG: px_auroc={avgs['invad_px_auroc']} (Δ{avgs['delta_px_auroc']:+.4f}) vs bank_px={avg_b_px}  "
            f"mAP10={avgs['invad_mAP10']} (Δ{avgs['delta_mAP10']:+.4f}) vs bank_mAP10={avg_b_m}")

log(f"\nDONE — {len(rows)} rows saved to {OUT_CSV}")
