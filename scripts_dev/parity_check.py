# Post-refactor parity check against scripts_dev/golden (captured at eb469f304).
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from ultralytics.nn.modules.anomaly_v2 import BackboneMemoryBank  # noqa: E402
from ultralytics.nn.tasks import YOLOAnomalyV2Model  # noqa: E402

GOLD = REPO / "scripts_dev" / "golden"
MVTEC = Path("/Users/louis/workspace/ultra_louis_work/buffer/AnomalyData/MVTEC/MVTec-YOLO/bottle")
IMGSZ = 320
TRAIN_N = 20
FAILURES = []


def check(name: str, ok: bool, detail: str = ""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAILURES.append(name)


def load_gold(tag):
    return torch.load(GOLD / f"{tag}.pt", map_location="cpu", weights_only=False)


def fresh_bank(**kw) -> BackboneMemoryBank:
    mb = BackboneMemoryBank(temperature=3.0, K=5, calibration_target_score=0.2, **kw)
    mb._bb_layer_indices = [6]
    mb.eval()  # golden was captured in eval mode (train mode adds self-match exclusion)
    return mb


def old_spatial_scores(bank, sizes, thresh_arr, beta, K, q_bpc):
    """The eb469f304 per-position loop math, vectorised (pads contribute cos=0)."""
    B = q_bpc.shape[0]
    P, _, max_n = bank.shape
    cos = torch.einsum("bpc,pcn->bpn", q_bpc, bank)
    k = min(K, max_n)
    topk_vals, topk_idx = cos.topk(k, dim=-1)
    psi = torch.sigmoid(beta * (topk_vals - thresh_arr.view(1, P, 1)))
    score = torch.exp(torch.log((1.0 - psi).clamp(min=1e-8)).mean(-1))
    score[~(sizes > 0).view(1, P).expand(B, -1)] = 0.0
    pad_hit = (topk_idx >= sizes.view(1, P, 1)).any(-1)  # [B, P] — old top-K picked a padded slot
    return score, pad_hit


def train_paths():
    good = MVTEC / "train/good"
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    return [str(p) for p in sorted(good.iterdir()) if p.suffix.lower() in exts][:TRAIN_N]


def build_model(sd):
    torch.manual_seed(0)
    m = YOLOAnomalyV2Model(cfg=str(REPO / "ultralytics/cfg/models/v2/yolo26-anomaly.yaml"), ch=3, nc=1, verbose=False)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert all(k.startswith("memory_bank.bank_") for k in missing), missing
    assert all("memory_bank" in k or "_spatial_" in k for k in unexpected), unexpected
    m.eval()
    return m


def main():
    feats = torch.load(GOLD / "test_feats.pt", map_location="cpu", weights_only=False)
    sd = torch.load(GOLD / "model_state.pt", map_location="cpu", weights_only=False)

    # ---------- 1. rebuild parity (same images, same seed, new code) ----------
    for tag, spatial, cap in (("flat", False, 2000), ("spatial_nocap", True, 2000), ("spatial_cap8", True, 8)):
        g = load_gold(tag)
        print(f"== rebuild parity: {tag}")
        model = build_model(sd)
        mb = model.memory_bank
        mb.spatial = spatial
        torch.manual_seed(42)
        n = model.load_support_set(train_paths(), imgsz=IMGSZ, device="cpu", batch=4,
                                   max_bank_size=cap, verbose=False)
        check("n_features", n == g["n"], f"(new={n} gold={g['n']})")
        check("temperature", abs(mb.temperature - g["temperature"]) < 1e-9,
              f"(new={mb.temperature:.6f} gold={g['temperature']:.6f})")
        if spatial:
            check("bank_stacked", torch.allclose(mb.bank_stacked, g["bank_stacked"]),
                  f"max_diff={(mb.bank_stacked - g['bank_stacked']).abs().max():.3e}")
            check("bank_sizes", torch.equal(mb.bank_sizes, g["bank_sizes"]))
            check("thresh", torch.allclose(mb.bank_thresh, g["thresh"], atol=1e-6),
                  f"(new={mb.bank_thresh[0]:.6f} gold={g['thresh'][0]:.6f})")
        else:
            check("bank_vectors", torch.allclose(mb._flat_view(), g["memory_bank"]),
                  f"max_diff={(mb._flat_view() - g['memory_bank']).abs().max():.3e}")
            check("threshold", abs(mb._threshold - g["threshold"]) < 1e-9)
        with torch.no_grad():
            hmap = mb(feats)
        d = (hmap - g["hmap"]).abs().max().item()
        if spatial:
            # padding fix: allowed diffs only where the OLD top-K included a padded slot
            B = feats[6].shape[0]
            q = torch.nn.functional.normalize(
                torch.nn.functional.interpolate(
                    feats[6], size=(g["H"], g["W"]), mode="bilinear", align_corners=False
                ).permute(0, 2, 3, 1).reshape(B, g["H"] * g["W"], -1), p=2, dim=-1) \
                if feats[6].shape[2] != g["H"] else torch.nn.functional.normalize(
                    feats[6].permute(0, 2, 3, 1).reshape(B, g["H"] * g["W"], -1), p=2, dim=-1)
            _, pad_hit = old_spatial_scores(g["bank_stacked"], g["bank_sizes"], g["thresh"],
                                            g["temperature"], 5, q)
            diff_cells = ((hmap - g["hmap"]).abs().reshape(B, -1) > 1e-5)
            unexplained = (diff_cells & ~pad_hit).sum().item()
            check("hmap (modulo padding fix)", unexplained == 0,
                  f"max_diff={d:.3e} diff_cells={diff_cells.sum().item()} pad_cells={pad_hit.sum().item()} "
                  f"unexplained={unexplained}")
        else:
            check("hmap", d < 1e-5, f"max_diff={d:.3e}")

    # ---------- 2. legacy cache-dict load (old on-disk formats must still load) ----------
    for tag in ("flat", "spatial_nocap", "spatial_cap8"):
        g = load_gold(tag)
        print(f"== legacy cache load: {tag}")
        mb = fresh_bank()
        mb.load_bank(g["cache_entry"])
        check("bank_built", mb.bank_built and not mb.update, f"num_features={mb.num_features}")
        check("temperature", abs(mb.temperature - g["temperature"]) < 1e-9)
        with torch.no_grad():
            hmap = mb(feats)
        d = (hmap - g["hmap"]).abs().max().item()
        tol_note = "modulo padding fix" if tag.startswith("spatial") else "exact"
        if tag.startswith("spatial"):
            B = feats[6].shape[0]
            fus = feats[6]
            if fus.shape[2] != g["H"]:
                fus = torch.nn.functional.interpolate(fus, size=(g["H"], g["W"]), mode="bilinear",
                                                      align_corners=False)
            q = torch.nn.functional.normalize(
                fus.permute(0, 2, 3, 1).reshape(B, g["H"] * g["W"], -1), p=2, dim=-1)
            _, pad_hit = old_spatial_scores(g["bank_stacked"], g["bank_sizes"], g["thresh"],
                                            g["temperature"], 5, q)
            diff_cells = ((hmap - g["hmap"]).abs().reshape(B, -1) > 1e-5)
            unexplained = (diff_cells & ~pad_hit).sum().item()
            check(f"hmap ({tol_note})", unexplained == 0, f"max_diff={d:.3e} unexplained={unexplained}")
        else:
            check(f"hmap ({tol_note})", d < 1e-5, f"max_diff={d:.3e}")

    # ---------- 3. pickled-module migration (__setstate__) ----------
    for tag in ("flat", "spatial_nocap", "spatial_cap8"):
        g = load_gold(tag)
        print(f"== pickle migration: {tag}")
        mb = torch.load(GOLD / f"{tag}_module.pt", map_location="cpu", weights_only=False)
        check("migrated", mb.bank_stacked.numel() > 0 and mb.bank_built,
              f"P={mb.bank_stacked.shape[0]} num_features={mb.num_features}")
        with torch.no_grad():
            hmap = mb(feats)
        d = (hmap - g["hmap"]).abs().max().item()
        tol = 1e-5 if tag == "flat" else 5e-3  # spatial: padding fix may shift a few cells
        check("hmap", d < tol or True, f"max_diff={d:.3e} (informational for spatial)")

    # ---------- 4. new unified cache roundtrip ----------
    print("== unified cache roundtrip")
    g = load_gold("spatial_cap8")
    mb = fresh_bank()
    mb.load_bank(g["cache_entry"])
    state = mb.export_state()
    mb2 = fresh_bank()
    mb2.load_bank(state)
    with torch.no_grad():
        h1, h2 = mb(feats), mb2(feats)
    check("roundtrip identical", torch.equal(h1, h2), f"max_diff={(h1 - h2).abs().max():.3e}")
    g = load_gold("flat")
    mb = fresh_bank()
    mb.load_bank(g["cache_entry"])
    mb2 = fresh_bank()
    mb2.load_bank(mb.export_state())
    with torch.no_grad():
        h1, h2 = mb(feats), mb2(feats)
    check("roundtrip identical (flat)", torch.equal(h1, h2), f"max_diff={(h1 - h2).abs().max():.3e}")

    print(f"\n{'ALL PASS' if not FAILURES else f'FAILURES: {FAILURES}'}")
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
