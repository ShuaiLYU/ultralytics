"""InvAD v3 Step 2 — calibration sweep on trained decoders.

Loads saved decoder weights from Step 1, sweeps calibration modes
(none/minmax/zscore/gamma), evaluates mAP10 + px_auroc + im_auroc.

Usage (single GPU):
  CUDA_VISIBLE_DEVICES=6 nohup /home/louis/miniconda3/envs/ultra/bin/python \
    tune_invad_v3_step2.py --config ssm_r_dch512_nb2_sty32_bat_resinte \
    > runs/temp/invad_v3_step2.log 2>&1 &
"""
import sys; sys.path.insert(0, ".")
import logging; logging.getLogger("ultralytics").setLevel(logging.WARNING)
from pathlib import Path
import argparse, csv, json, time, numpy as np, traceback, torch
from ultralytics.yoloa import YOLOA
from ultralytics.models.yolo.anomaly_v2.val import run_mvtec_ood_eval

ap = argparse.ArgumentParser()
ap.add_argument("--config", type=str, required=True, help="Step 1 config name to evaluate")
ap.add_argument("--device", type=str, default="cuda:0")
args = ap.parse_args()

CKPT = "/home/louis/ultra_louis_work/expman/data/pulled/yoloa_clean/26m_yoloav2_softhint_maskonly_aug3_mixup_ood_aug2x_ep15_lr2x_v1/weights/best.pt"
ROOT = Path("/data/shared-datasets/louis_data/MVTec-YOLO/MVTec-YOLO")
DECODER_DIR = Path("runs/temp/decoders")
IMGSZ = 640
BB_LAYERS = [6]
CATS = ["bottle", "cable", "screw", "zipper", "toothbrush"]
DEVICE = args.device

BANK_BASELINE = {
    "bottle":     {"mAP10": 0.8092, "mAP25": 0.6410, "mAP50": 0.1930, "im_auroc": 0.9543, "px_auroc": 0.9166},
    "cable":      {"mAP10": 0.3601, "mAP25": 0.2546, "mAP50": 0.0588, "im_auroc": 0.9185, "px_auroc": 0.9232},
    "screw":      {"mAP10": 0.3289, "mAP25": 0.0836, "mAP50": 0.0144, "im_auroc": 0.8623, "px_auroc": 0.9372},
    "zipper":     {"mAP10": 0.9556, "mAP25": 0.9196, "mAP50": 0.5203, "im_auroc": 0.9842, "px_auroc": 0.9192},
    "toothbrush": {"mAP10": 0.5266, "mAP25": 0.2585, "mAP50": 0.1012, "im_auroc": 1.0000, "px_auroc": 0.9482},
}

# Architecture kwargs for the target config — must match Step 1 exactly.
# Parsed from the config name.
def parse_config_kw(cfg_name: str) -> dict:
    """Parse Step 1 config name back to fit_decoder kwargs."""
    parts = cfg_name.split("_")
    arch = "ssm"  # all step2 targets are SSM
    kw = {"arch": "ssm", "batch": 32}

    if cfg_name.startswith("ssm_L"):  # anchor block
        # ssm_L{mse|cos|msc}_lr{lr}_s{steps}
        loss_part = parts[1]  # Lmse, Lcos, Lmsc
        loss_map = {"Lmse": "mse", "Lcos": "cosine", "Lmsc": "mse+cosine"}
        kw["loss_mode"] = loss_map.get(loss_part, "mse+cosine")
        for p in parts:
            if p.startswith("lr"):
                kw["lr"] = float(p[2:])
            elif p.startswith("s") and not p.startswith("sty") and p[1:].isdigit():
                kw["steps"] = int(p[1:])
        kw.setdefault("decoder_ch", 256)
        kw.setdefault("num_blocks", 4)
        kw.setdefault("style_ch", 64)
        kw.setdefault("residual_mode", "block")
        kw.setdefault("norm_type", "instance")
    elif cfg_name.startswith("ssm_res"):  # residual × norm
        # ssm_res{rm}_n{nt}
        rm_part = parts[1]
        rm_map = {"resbloc": "block", "resnone": "none", "resinte": "inter_block", "resdens": "dense"}
        nt_map = {"nins": "instance", "ngro": "group", "nbat": "batch"}
        kw["loss_mode"] = "mse+cosine"
        kw["lr"] = 0.001
        kw["steps"] = 2000
        kw["decoder_ch"] = 256
        kw["num_blocks"] = 4
        kw["style_ch"] = 64
        kw["residual_mode"] = rm_map.get(rm_part, "block")
        kw["norm_type"] = nt_map.get(parts[2], "instance")
    elif cfg_name.startswith("ssm_r_"):  # random block
        # ssm_r_dch{decoder_ch}_nb{num_blocks}_sty{style_ch}_{norm_type}_res{residual_mode}
        kw["loss_mode"] = "mse+cosine"
        kw["lr"] = 0.001
        kw["steps"] = 2000
        for p in parts:
            if p.startswith("dch"):
                kw["decoder_ch"] = int(p[3:])
            elif p.startswith("nb"):
                kw["num_blocks"] = int(p[2:])
            elif p.startswith("sty"):
                kw["style_ch"] = int(p[3:])
        # Last two meaningful parts: norm_type and residual_mode
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


CONFIG = args.config
KW = parse_config_kw(CONFIG)

# Calibration sweep
# (_cal_mode, _gamma) — gamma applied before _apply_calibration
CAL_SWEEP = [
    ("none",   1.0, "none"),
    ("minmax", 1.0, "minmax"),
    ("zscore", 1.0, "zscore"),
    ("g0.2",   0.2, "none"),
    ("g0.3",   0.3, "none"),
    ("g0.5",   0.5, "none"),
    ("g0.7",   0.7, "none"),
    ("g0.9",   0.9, "none"),
]

OUT_CSV = Path(f"runs/temp/invad_v3_step2_{CONFIG}.csv")
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

LOG_PATH = Path(f"runs/temp/invad_v3_step2_{CONFIG}.log")

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")

log(f"Step 2 calibration sweep: {CONFIG}")
log(f"  Arch kwargs: {KW}")
log(f"  Device: {DEVICE}")
log(f"  Modes: {[m[0] for m in CAL_SWEEP]}")

fieldnames = ["category", "config", "cal_mode", "cal_gamma",
              "invad_px_auroc", "invad_im_auroc",
              "invad_mAP10", "invad_mAP25", "invad_mAP50",
              "bank_mAP10", "bank_px_auroc",
              "delta_mAP10", "delta_px_auroc"]

with open(OUT_CSV, "w", newline="") as f:
    csv.DictWriter(f, fieldnames=fieldnames).writeheader()

rows = []
for cat in CATS:
    weight_path = DECODER_DIR / f"{CONFIG}_{cat}.pt"
    if not weight_path.exists():
        log(f"  SKIP {cat}: no weights at {weight_path}")
        continue

    for cal_name, gamma, cal_mode in CAL_SWEEP:
        log(f"  {cat} [{cal_name}] gamma={gamma} mode={cal_mode}...")
        try:
            m = YOLOA(CKPT)
            m.model.to(DEVICE)
            gd = str(ROOT / cat / "train/good")

            # Fit builds memory bank + creates decoder arch
            m.fit(gd, name=cat, imgsz=IMGSZ, max_images=0, refit=True,
                  bb_layers=BB_LAYERS, fit_decoder=KW)

            dec = m.model._feat_inv_decoder
            state = torch.load(weight_path, map_location=DEVICE)

            # Handle wrapped keys (some checkpoints save as DataParallel)
            if any(k.startswith("module.") for k in state.keys()):
                state = {k[7:]: v for k, v in state.items()}
            dec.load_state_dict(state)

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
                "config": CONFIG,
                "cal_mode": cal_name,
                "cal_gamma": gamma,
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
            rows.append(row)
            with open(OUT_CSV, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writerow(row)

            log(f"    px={row['invad_px_auroc']:.4f}  mAP10={row['invad_mAP10']:.4f}"
                f" (Δ{row['delta_mAP10']:+.4f})")

        except Exception as e:
            log(f"  FAILED: {e}")
            traceback.print_exc()
            continue

# Summary
log(f"\n--- Results ---")
by_cal = {}
for r in rows:
    cal = r["cal_mode"]
    if cal not in by_cal:
        by_cal[cal] = []
    by_cal[cal].append(r)

for cal_name in [m[0] for m in CAL_SWEEP]:
    cal_rows = by_cal.get(cal_name, [])
    if len(cal_rows) == 5:
        avg_px = sum(float(r["invad_px_auroc"]) for r in cal_rows) / 5
        avg_im = sum(float(r["invad_im_auroc"]) for r in cal_rows) / 5
        avg_m10 = sum(float(r["invad_mAP10"]) for r in cal_rows) / 5
        avg_dm10 = sum(float(r["delta_mAP10"]) for r in cal_rows) / 5
        log(f"  {cal_name:8s}  px={avg_px:.4f}  im={avg_im:.4f}  mAP10={avg_m10:.4f} (Δ{avg_dm10:+.4f})")
    elif cal_rows:
        log(f"  {cal_name:8s}  {len(cal_rows)}/5 done")

log(f"\nDONE — {len(rows)} rows → {OUT_CSV}")
