# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Build a pretrained donor for yolo26-k6.yaml that starts out identical to the stock checkpoint.

yolo26-k6.yaml widens the two early downsampling convs from k=3,s=2 to k=6,s=2,p=2. Those kernels have no
same-shape counterpart in the stock checkpoint, so a plain transfer leaves them randomly initialized --
12 tensors and ~166k parameters, in the two earliest feature layers, while every downstream layer stays
pretrained and expects the features those layers used to produce. Any comparison against the baseline
would then confound the wider kernel with the loss of pretrained early weights, and a negative result
would be uninterpretable.

A 6x6 stride-2 conv contains the 3x3 stride-2 conv exactly: with p=2 the 6x6 window spans input offsets
-2..3, and the 3x3 window with p=1 spans -1..1, which is offsets 1..3 of that window. Placing the
pretrained kernel at [1:4, 1:4] and zeroing the rest reproduces the original layer bit for bit, borders
included, so training starts from the baseline and the extra taps start unused.

Usage:
    python scripts/anomaly_bench/make_k6_donor.py --src yolo26n.pt --cfg yolo26n-k6.yaml \
        --out yolo26n-k6.pt

Verify the result is exactly equivalent before spending GPU hours on it:
    python scripts/anomaly_bench/make_k6_donor.py --src yolo26n.pt --cfg yolo26n-k6.yaml \
        --out yolo26n-k6.pt --check
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Running this as a file puts sys.path[0] at scripts/anomaly_bench/, so `import ultralytics` would resolve
# to whatever the editable install points at rather than the checkout this script lives in.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ultralytics import YOLO  # noqa: E402
from ultralytics.utils import LOGGER  # noqa: E402
from ultralytics.utils.downloads import attempt_download_asset  # noqa: E402


def build(src: str, cfg: str, out: str) -> tuple[Path, int, int]:
    """Write a donor checkpoint for `cfg` whose widened kernels embed `src`'s narrower ones.

    Args:
        src (str): Stock checkpoint to take weights from.
        cfg (str): Model yaml with the widened downsampling convs.
        out (str): Destination checkpoint path.

    Returns:
        (tuple[Path, int, int]): Output path, tensors transferred, kernels embedded.
    """
    ckpt = torch.load(attempt_download_asset(src), map_location="cpu", weights_only=False)
    donor = (ckpt.get("ema") or ckpt["model"]).float().state_dict()
    model = YOLO(cfg).model.float()
    sd = model.state_dict()

    kept, embedded = {}, 0
    for k, v in sd.items():
        d = donor.get(k)
        if d is None:
            continue
        if d.shape == v.shape:
            kept[k] = d
        elif d.ndim == 4 and v.ndim == 4 and d.shape[:2] == v.shape[:2] and d.shape[2:] == (3, 3) and v.shape[2:] == (6, 6):
            w = torch.zeros_like(v)
            w[:, :, 1:4, 1:4] = d  # a 3x3 p=1 window is offsets 1..3 of a 6x6 p=2 window
            kept[k] = w
            embedded += 1
            LOGGER.info(f"embedded {k}: {tuple(d.shape)} -> {tuple(v.shape)}")

    missing = [k for k in sd if k not in kept]
    if missing:
        raise SystemExit(f"{len(missing)} tensors would stay random, refusing to write: {missing[:6]}")
    model.load_state_dict(kept)

    # Mirror the stock checkpoint so the trainer's pretrained path consumes it unchanged
    ckpt.pop("ema", None)
    ckpt.pop("updates", None)
    ckpt["model"] = model.half()
    torch.save(ckpt, out)
    return Path(out), len(kept), embedded


def check(src: str, out: str, imgsz: int = 320) -> float:
    """Return the largest absolute difference between the donor's and the source's forward pass."""
    a = YOLO(src).model.float().eval()
    b = YOLO(out).model.float().eval()
    x = torch.rand(1, 3, imgsz, imgsz)
    with torch.inference_mode():
        ya, yb = a(x), b(x)
    ya = ya[0] if isinstance(ya, (list, tuple)) else ya
    yb = yb[0] if isinstance(yb, (list, tuple)) else yb
    return (ya - yb).abs().max().item()


def main() -> None:
    """Parse arguments, build the donor, and optionally verify it against the source checkpoint."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", default="yolo26n.pt", help="stock checkpoint to take weights from")
    p.add_argument("--cfg", default="yolo26n-k6.yaml", help="model yaml with widened downsampling convs")
    p.add_argument("--out", default="yolo26n-k6.pt", help="destination checkpoint")
    p.add_argument("--check", action="store_true", help="verify the donor matches the source forward pass")
    # Launcher bookkeeping, unused here. The two launchers disagree: nohuppython demands --project/--name
    # and hard-errors on a bare name= token, while expman-cli launch demands exactly that token. Accept
    # both so this can be launched either way.
    p.add_argument("--project", help="ignored; required by the nohuppython launcher")
    p.add_argument("--name", help="ignored; required by the nohuppython launcher")
    p.add_argument("token", nargs="?", help="ignored; expman-cli launch requires a name= token in the args")
    a = p.parse_args()

    path, kept, embedded = build(a.src, a.cfg, a.out)
    LOGGER.info(f"wrote {path} — {kept} tensors, {embedded} kernels embedded")
    if a.check:
        d = check(a.src, str(path))
        LOGGER.info(f"max abs diff vs {a.src}: {d:.3e}")
        assert d == 0.0, f"donor is not exactly equivalent to {a.src} (diff {d:.3e})"
        LOGGER.info("donor is bit-identical to the source checkpoint")


if __name__ == "__main__":
    main()
