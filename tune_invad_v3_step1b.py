"""InvAD v3 Step 1b — re-eval top-N SSM configs with calibration sweep.

Loads saved decoder weights from Step 1, skips retraining (steps=1),
evaluates mAP10 + px_auroc with minmax/g0.3/zscore calibration.

Usage on ultra6:
  CUDA_VISIBLE_DEVICES=6 nohup /home/louis/miniconda3/envs/ultra/bin/python \
    tune_invad_v3_step1b.py --gpu 0 --total-gpus 1 \
    > runs/temp/invad_v3_step1b.log 2>&1 &

For multi-GPU:
  for gpu in 6 7; do
    CUDA_VISIBLE_DEVICES=$gpu nohup ... --gpu $((gpu-6)) --total-gpus 2 \
      > runs/temp/invad_v3_step1b_gpu$((gpu-6)).log 2>&1 &
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
args_in = ap.parse_args()

GPU_INDEX = args_in.gpu
GPU_COUNT = args_in.total_gpus
DEVICE = "cuda:0"

CKPT = "/home/louis/ultra_louis_work/expman/data/pulled/yoloa_clean/26m_yoloav2_softhint_maskonly_aug3_mixup_ood_aug2x_ep15_lr2x_v1/weights/best.pt"
ROOT = Path("/data/shared-datasets/louis_data/MVTec-YOLO/MVTec-YOLO")
DECODER_DIR = Path("runs/temp/decoders")
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

# Top-10 SSM by px_auroc from Step 1
TARGET_CONFIGS = [
    "ssm_r_dch512_nb2_sty32_bat_resinte",
    "ssm_r_dch256_nb6_sty32_bat_resdens",
    "ssm_r_dch512_nb6_sty64_bat_resbloc",
    "ssm_Lmse_lr0.0005_s4000",
    "ssm_resnone_nins",
    "ssm_Lcos_lr0.002_s4000",
    "ssm_resdens_nins",
    "ssm_Lcos_lr0.002_s2000",
    "ssm_resnone_nbat",
    "ssm_Lmsc_lr0.001_s1000",
]

def parse_config_kw(cfg_name: str) -> dict:
    """Parse Step 1 config name back to fit_decoder kwargs."""
    parts = cfg_name.split("_")
    kw = {"arch": "ssm", "batch": 32, "steps": 1}  # steps=1: skip retraining

    if cfg_name.startswith("ssm_L"):
        loss_part = parts[1]
        loss_map = {"Lmse": "mse", "Lcos": "cosine", "Lmsc": "mse+cosine"}
        kw["loss_mode"] = loss_map.get(loss_part, "mse+cosine")
        for p in parts:
            if p.startswith("lr"):
                kw["lr"] = float(p[2:])
            elif p.startswith("s") and not p.startswith("sty") and p[1:].isdigit():
                pass  # skip original steps — we override to 1
        kw.setdefault("decoder_ch", 256)
        kw.setdefault("num_blocks", 4)
        kw.setdefault("style_ch", 64)
        kw.setdefault("residual_mode", "block")
        kw.setdefault("norm_type", "instance")
    elif cfg_name.startswith("ssm_res"):
        rm_part = parts[1]
        rm_map = {"resbloc": "block", "resnone": "none", "resinte": "inter_block", "resdens": "dense"}
        nt_map = {"nins": "instance", "ngro": "group", "nbat": "batch"}
        kw["loss_mode"] = "mse+cosine"
        kw["lr"] = 0.001
        kw["decoder_ch"] = 256
        kw["num_blocks"] = 4
        kw["style_ch"] = 64
        kw["residual_mode"] = rm_map.get(rm_part, "block")
        kw["norm_type"] = nt_map.get(parts[2], "instance")
    elif cfg_name.startswith("ssm_r_"):
        kw["loss_mode"] = "mse+cosine"
        kw["lr"] = 0.001
        for p in parts:
            if p.startswith("dch"):
                kw["decoder_ch"] = int(p[3:])
            elif p.startswith("nb"):
                kw["num_blocks"] = int(p[2:])
            elif p.startswith("sty"):
                kw["style_ch"] = int(p[3:])
        nt_map = {"ins": "instance", "gro": "group", "bat": "batch"}
        rm_map = {"resbloc": "block", "resnone": "none", "resinte": "inter_block", "resdens": "dense"}
        for p in parts:
            if p in nt_map:
                kw["norm_type"] = nt_map[p]
            elif p in rm_map:
                kw["residual_mode"] = rm_map[p]
        kw.setdefault("norm_type", "instance")
        kw.setdefault("residual_mode", "block")
    else:
        raise ValueError(f"Unknown config pattern: {cfg_name}")

    return kw


# Partition configs across GPUs
my_configs = [c for i, c in enumerate(TARGET_CONFIGS) if i % GPU_COUNT == GPU_INDEX]
print(f"[gpu{GPU_INDEX}/{GPU_COUNT}] {len(my_configs)} configs × {len(CATS)} cats", flush=True)

# Calibration sweep: (label, gamma, cal_mode)
CAL_SWEEP = [
    ("minmax", 1.0, "minmax"),
    ("g0.3",  0.3, "none"),
    ("zscore", 1.0, "zscore"),
]

OUT_CSV = Path(f"runs/temp/invad_v3_step1b_gpu{GPU_INDEX}.csv")
LOG = Path(f"runs/temp/invad_v3_step1b_gpu{GPU_INDEX}.log")
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] g{GPU_INDEX} {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

log(f"Step 1b: {len(my_configs)} configs × {len(CATS)} cats × {len(CAL_SWEEP)} modes")
log(f"Configs: {my_configs}")

fieldnames = ["category", "config", "cal_mode",
              "invad_px_auroc", "invad_im_auroc",
              "invad_mAP10", "invad_mAP25", "invad_mAP50",
              "bank_mAP10", "bank_px_auroc",
              "delta_mAP10", "delta_px_auroc"]

with open(OUT_CSV, "w", newline="") as f:
    csv.DictWriter(f, fieldnames=fieldnames).writeheader()

for cfg_name in my_configs:
    kw = parse_config_kw(cfg_name)
    arch_kw = {k: v for k, v in kw.items() if k != "steps"}
    log(f"\n--- {cfg_name} | {arch_kw} ---")

    for cat in CATS:
        weight_path = DECODER_DIR / f"{cfg_name}_{cat}.pt"
        if not weight_path.exists():
            log(f"  SKIP {cat}: no weights")
            continue

        # Build bank once per (config, cat) — decoder arch from kwargs, steps=1 skips training
        try:
            m = YOLOA(CKPT)
            m.model.to(DEVICE)
            gd = str(ROOT / cat / "train/good")
            m.fit(gd, name=cat, imgsz=IMGSZ, max_images=0, refit=True,
                  bb_layers=BB_LAYERS, fit_decoder=kw)
        except Exception as e:
            log(f"  FAILED bank build {cat}: {e}")
            continue

        dec = m.model._feat_inv_decoder
        state = torch.load(weight_path, map_location=DEVICE)
        if any(k.startswith("module.") for k in state.keys()):
            state = {k[7:]: v for k, v in state.items()}
        dec.load_state_dict(state)

        for cal_name, gamma, cal_mode in CAL_SWEEP:
            log(f"  {cat} [{cal_name}]")
            try:
                dec._gamma = gamma
                dec._cal_mode = cal_mode

                m.model.set_prior_mode("heatmap_reconstruct")
                d_rows = run_mvtec_ood_eval(m.model, ROOT, categories=[cat],
                                            modes=("heatmap_reconstruct",), imgsz=IMGSZ,
                                            batch=4, device=DEVICE)
                d = [x for x in d_rows if x["category"] == cat and x.get("mode") == "heatmap_reconstruct"][0]
                b = BANK_BASELINE[cat]

                row = {
                    "category": cat,
                    "config": cfg_name,
                    "cal_mode": cal_name,
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
                with open(OUT_CSV, "a", newline="") as f:
                    csv.DictWriter(f, fieldnames=fieldnames).writerow(row)

                log(f"    px={row['invad_px_auroc']:.4f}  mAP10={row['invad_mAP10']:.4f}"
                    f" (Δ{row['delta_mAP10']:+.4f})")

            except Exception as e:
                log(f"    FAILED: {e}")
                traceback.print_exc()
                continue

log(f"\nDONE — {OUT_CSV}")
