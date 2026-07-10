# Golden-output capture for the memory-bank format unification.
# Run BEFORE refactoring (at eb469f304) to snapshot flat + spatial behavior.
# Post-refactor, scripts_dev/parity_check.py compares against these files.
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from ultralytics.nn.tasks import YOLOAnomalyV2Model  # noqa: E402

MVTEC = Path("/Users/louis/workspace/ultra_louis_work/buffer/AnomalyData/MVTEC/MVTec-YOLO/bottle")
OUT = REPO / "scripts_dev" / "golden"
OUT.mkdir(parents=True, exist_ok=True)

TRAIN_N = 20
IMGSZ = 320
def _first_pngs(d: Path, n: int) -> list[Path]:
    return sorted(d.glob("*.png"))[:n]


TEST_IMGS = _first_pngs(MVTEC / "test/good", 2) + _first_pngs(MVTEC / "test/broken_large", 1) + _first_pngs(
    MVTEC / "test/broken_small", 1
)


def build_model() -> YOLOAnomalyV2Model:
    torch.manual_seed(0)
    m = YOLOAnomalyV2Model(cfg=str(REPO / "ultralytics/cfg/models/v2/yolo26-anomaly.yaml"), ch=3, nc=1, verbose=False)
    m.eval()
    return m


def train_paths() -> list[str]:
    good = MVTEC / "train/good"
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    return [str(p) for p in sorted(good.iterdir()) if p.suffix.lower() in exts][:TRAIN_N]


def capture_test_feats(model) -> dict[int, torch.Tensor]:
    """Run the 4 test images through the model, return the tapped backbone feature dict."""
    import cv2
    import numpy as np

    imgs = []
    for p in TEST_IMGS:
        img = cv2.imread(str(p))
        assert img is not None, p
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(cv2.resize(img, (IMGSZ, IMGSZ), interpolation=cv2.INTER_LINEAR))
    batch = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).float() / 255.0
    model._bb_feats = {}
    with torch.no_grad():
        model(batch)
    return {k: v.clone() for k, v in model._bb_feats.items()}


def legacy_cache_entry(mb) -> dict:
    """Replicate the eb469f304 yoloa.py cache-write format verbatim."""
    if mb.spatial:
        entry = {"memory_bank": {
            "bank_stacked": mb._spatial_bank_stacked.detach().cpu(),
            "bank_sizes": mb._spatial_bank_sizes.cpu(),
            "comp": mb._spatial_comp_stacked.cpu(),
            "thresh": mb._spatial_thresh_stacked.cpu(),
            "feature_dim": mb.feature_dim,
            "temperature": float(mb.temperature),
            "H": mb._spatial_H,
            "W": mb._spatial_W,
        }, "feature_dim": mb.feature_dim, "temperature": float(mb.temperature)}
        entry["_threshold_arr"] = mb._spatial_thresh_stacked.cpu()
        entry["_compactness_arr"] = mb._spatial_comp_stacked.cpu()
        entry["_calibrated"] = True
    else:
        entry = {
            "memory_bank": mb.memory_bank.detach().cpu(),
            "feature_dim": mb.feature_dim,
            "temperature": float(mb.temperature),
        }
        if getattr(mb, "_calibrated", False):
            entry["_threshold"] = mb._threshold
            entry["_compactness"] = mb._compactness
            entry["_calibrated"] = True
    return entry


def run_variant(tag: str, spatial: bool, max_bank_size, state_dict, test_feats_ref: list):
    model = build_model()
    if state_dict is not None:
        model.load_state_dict(state_dict)
    mb = model.memory_bank
    mb.spatial = spatial
    torch.manual_seed(42)  # freeze uses unseeded randperm for holdout splits
    n = model.load_support_set(train_paths(), imgsz=IMGSZ, device="cpu", batch=4,
                               max_bank_size=max_bank_size, verbose=False)
    feats = capture_test_feats(model)
    if not test_feats_ref:
        test_feats_ref.append(feats)  # identical across variants (same weights) — save once
    with torch.no_grad():
        hmap = mb(feats)
    rec = {
        "n": n,
        "hmap": hmap,
        "temperature": float(mb.temperature),
        "cache_entry": legacy_cache_entry(mb),
    }
    if spatial:
        rec.update(H=mb._spatial_H, W=mb._spatial_W,
                   bank_stacked=mb._spatial_bank_stacked.clone(),
                   bank_sizes=mb._spatial_bank_sizes.clone(),
                   thresh=mb._spatial_thresh_stacked.clone(),
                   comp=mb._spatial_comp_stacked.clone())
    else:
        rec.update(memory_bank=mb.memory_bank.clone(),
                   threshold=mb._threshold, compactness=mb._compactness)
    torch.save(rec, OUT / f"{tag}.pt")
    torch.save(mb, OUT / f"{tag}_module.pt")  # pickled module for __setstate__ migration test
    print(f"[{tag}] n={n} hmap={tuple(hmap.shape)} min={hmap.min():.4f} max={hmap.max():.4f} "
          f"mean={hmap.mean():.4f}")
    return model


def main():
    base = build_model()
    torch.save(base.state_dict(), OUT / "model_state.pt")
    sd = torch.load(OUT / "model_state.pt", weights_only=False)

    feats_ref: list = []
    run_variant("flat", spatial=False, max_bank_size=2000, state_dict=sd, test_feats_ref=feats_ref)
    run_variant("spatial_nocap", spatial=True, max_bank_size=2000, state_dict=sd, test_feats_ref=feats_ref)
    run_variant("spatial_cap8", spatial=True, max_bank_size=8, state_dict=sd, test_feats_ref=feats_ref)
    torch.save(feats_ref[0], OUT / "test_feats.pt")
    print("golden capture complete ->", OUT)


if __name__ == "__main__":
    main()
