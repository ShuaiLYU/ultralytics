"""Compare nonlinear calibration methods for InvAD heatmap."""
import sys
sys.path.insert(0, ".")
import logging
logging.getLogger("ultralytics").setLevel(logging.WARNING)

from pathlib import Path
import cv2, torch, numpy as np, types
from ultralytics.yoloa import YOLOA
from ultralytics.models.yolo.anomaly_v2.val import run_mvtec_ood_eval

CKPT = "/Users/louis/workspace/ultra_louis_work/expman/data/pulled/yoloa_clean/26m_yoloav2_softhint_maskonly_aug3_mixup_ood_aug2x_ep15_lr2x_v1/weights/best.pt"
ROOT = Path("/Users/louis/workspace/ultra_louis_work/buffer/AnomalyData/MVTEC/MVTec-YOLO")
CAT = "bottle"
DEVICE = "cpu"

m = YOLOA(CKPT)
model = m.model

# Fit
m.fit(str(ROOT / CAT / "train/good"), name=CAT, imgsz=640, max_images=50,
      bb_max_bank_size=500, fit_decoder={"num_blocks": 4, "decoder_ch": 128, "steps": 300})
dec = model._feat_inv_decoder

# Collect paired heatmaps on normal images
gd = ROOT / CAT / "train/good"
norm_paths = sorted(gd.glob("*.png"))[:50]
invad_vals, bank_vals = [], []
model.eval()
for p in norm_paths:
    img = cv2.imread(str(p)); img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640))
    t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    t = t.unsqueeze(0)
    model.set_prior_mode("heatmap_reconstruct")
    with torch.no_grad(): _ = model(t)
    bb = dict(model._bb_feats)
    # Use raw cosine distance (bypass bias calibration)
    recon = dec.forward(bb)
    maps = []
    for li, ef in bb.items():
        if li not in recon: continue
        en = torch.nn.functional.normalize(recon[li], p=2, dim=1)
        dn = torch.nn.functional.normalize(ef, p=2, dim=1)
        dist = ((1.0 - (en * dn).sum(dim=1, keepdim=True)) * 0.5).clamp(0, 1)
        maps.append(dist)
    imap = torch.stack(maps).mean(dim=0).detach().cpu().numpy().ravel()
    model.set_prior_mode("heatmap")
    with torch.no_grad(): _ = model(t)
    bmap = model._last_heatmap.detach().cpu().numpy().ravel()
    invad_vals.append(imap)
    bank_vals.append(bmap)

inv_all = np.concatenate(invad_vals)
bank_all = np.concatenate(bank_vals)
# Match sizes (bank heatmap may be at different resolution)
n = min(len(inv_all), len(bank_all))
rng = np.random.RandomState(0)
idx_inv = rng.choice(len(inv_all), n, replace=False)
idx_bank = rng.choice(len(bank_all), n, replace=False)
inv_all = inv_all[idx_inv]
bank_all = bank_all[idx_bank]
print(f"bank mean={bank_all.mean():.4f} median={np.median(bank_all):.4f} p95={np.percentile(bank_all, 95):.4f}")
print(f"invad mean={inv_all.mean():.4f} median={np.median(inv_all):.4f} p95={np.percentile(inv_all, 95):.4f}")

# ---- Calibration params ----
# Quantile map
sq = np.quantile(inv_all, np.linspace(0, 1, 257))
tq = np.quantile(bank_all, np.linspace(0, 1, 257))

# Best gamma (power law)
bg, bl = 0.3, float("inf")
for g in [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8]:
    loss = np.mean((np.power(inv_all.clip(1e-6), g) - bank_all) ** 2)
    if loss < bl: bl, bg = loss, g

# Sigmoid
a = float(np.median(inv_all))
b = max(float(np.percentile(inv_all, 90) - np.percentile(inv_all, 10)) / 4, 1e-4)
so = float(np.median(bank_all))

# Linear
ls = float(np.mean(bank_all) / max(np.mean(inv_all), 1e-6))

print(f"linear_scale={ls:.2f}  gamma={bg:.2f}  sigmoid(a={a:.4f},b={b:.4f},so={so:.4f})")

# ---- Calibration transforms ----
def calib_quantile(h):
    hn = h.detach().cpu().numpy()
    return torch.from_numpy(np.interp(hn, sq, tq)).to(h.device).float().clamp(0, 1)

def calib_linear(h):
    return (h * ls).clamp(0, 1)

def calib_power(h):
    return h.clamp(0, 1).pow(bg)

def calib_sigmoid(h):
    return (torch.sigmoid((h - a) / b) * so * 2).clamp(0, 1)

# ---- Test each method ----
orig_fn = model._prior_from_heatmap
results = {}

# Uncalibrated baseline first
rows = run_mvtec_ood_eval(model, ROOT, categories=[CAT],
    modes=("heatmap_reconstruct",), imgsz=640, batch=4, device=DEVICE)
r = [x for x in rows if x["category"] == CAT][0]
results["uncalibrated"] = (r["image_auroc"], r["pixel_auroc"], r["mAP10"])

for name, calib_fn in [("linear", calib_linear), ("power_law", calib_power),
                        ("sigmoid", calib_sigmoid), ("quantile", calib_quantile)]:
    def _make_patch(fn):
        def patched(self, mask_size, **kw):
            h = orig_fn(mask_size, **kw)
            return fn(h) if h is not None else None
        return types.MethodType(patched, model)
    model._prior_from_heatmap = _make_patch(calib_fn)
    rows = run_mvtec_ood_eval(model, ROOT, categories=[CAT],
        modes=("heatmap_reconstruct",), imgsz=640, batch=4, device=DEVICE)
    r = [x for x in rows if x["category"] == CAT][0]
    results[name] = (r["image_auroc"], r["pixel_auroc"], r["mAP10"])

model._prior_from_heatmap = orig_fn

print()
print(f'{"Method":20s} {"im_auroc":>10s} {"px_auroc":>10s} {"mAP10":>10s}')
print("-" * 52)
for name, (ia, pa, m10) in results.items():
    print(f"{name:20s} {ia:10.4f} {pa:10.4f} {m10:10.4f}")
