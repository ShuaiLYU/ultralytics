#!/usr/bin/env python3
"""Sanity gate: beta=0 YOLOAnomalyV2SegModel == vanilla SegmentationModel forward.

Builds the v2 seg model, runs eval-mode forward with NO mask (passthrough), runs again
with a synthetic prior but beta still 0, and asserts the outputs are identical.
Then sets beta=1.0 and checks that the output differs (proving the bias path is live).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
REPO = FILE.parents[1]
sys.path.insert(0, str(REPO))

from ultralytics.nn.tasks import YOLOAnomalyV2SegModel

CFG = "ultralytics/cfg/models/v2/yolo26-anomaly-v2-seg.yaml"


def to_tensor(out):
    """Flatten any nested output structure into a single float tensor for comparison."""
    if isinstance(out, dict):
        vals = []
        for v in out.values():
            vals.append(to_tensor(v))
        return torch.cat(vals) if vals else torch.zeros(0)
    if isinstance(out, (list, tuple)):
        vals = [to_tensor(v) for v in out]
        return torch.cat(vals) if vals else torch.zeros(0)
    if isinstance(out, torch.Tensor):
        return out.float().ravel()
    return torch.tensor([float(out)], dtype=torch.float32)


def main():
    torch.manual_seed(42)

    m = YOLOAnomalyV2SegModel(CFG, ch=3, nc=1, verbose=False, p_drop=0.0)
    m.eval()

    dummy = torch.randn(2, 3, 640, 640)

    # --- No mask (passthrough) ---
    m.disable_mask_once()
    with torch.no_grad():
        out_off = m(dummy)
    t_off = to_tensor(out_off)

    # --- With prior, beta=0 ---
    prior = torch.rand(2, 1, 160, 160)
    m.set_external_mask_once(prior)
    with torch.no_grad():
        out_beta0 = m(dummy)
    t_beta0 = to_tensor(out_beta0)

    if not torch.allclose(t_off, t_beta0, atol=1e-6):
        print("FAIL: beta=0 with prior != no-mask (max diff {:.6e})".format(
            (t_off - t_beta0).abs().max().item()))
        sys.exit(1)

    # --- With prior, beta=1.0 ---
    m.heatmap_bias_fusion.beta.data.fill_(1.0)
    m.set_external_mask_once(prior)
    with torch.no_grad():
        out_beta1 = m(dummy)
    t_beta1 = to_tensor(out_beta1)

    if torch.allclose(t_off, t_beta1, atol=1e-6):
        print("FAIL: beta=1.0 with prior == no-mask (bias path is dead)")
        sys.exit(1)

    print("PASS softhint-seg sanity.")
    sys.exit(0)


if __name__ == "__main__":
    main()
