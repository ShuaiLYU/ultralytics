# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Prove that SPD-Conv, as `Focus(k=3)`, is the same operator as `Conv(k=6, s=2)`.

The SPD-Conv design proposal treats space-to-depth followed by a 3x3 stride-1 conv as a distinct
primitive whose value is that it "discards no information", unlike a strided conv. That framing does not
survive contact with the arithmetic. Space-to-depth splits a 2x2 neighbourhood into four sub-lattices, so
a following 3x3 kernel reaches 3 half-resolution positions x 2 sub-pixels = 6 original pixels per axis:
a 6x6 window at stride 2, with 4C x 9 = 36 independent taps per input channel. A `Conv(k=6, s=2)` has the
same 6x6 window, the same stride, and the same 36 taps -- and the same parameter count. The two are
related by a pure reindexing of the weights, so they are not merely similar, they are the same linear
operator family. Whatever a space-to-depth block can learn, a 6x6 strided conv can learn identically,
using one conv kernel instead of four non-contiguous slices, a concat, and a conv.

That is why this benchmark widens the early downsampling convs to k=6 instead of adding a Focus layer,
and why `Focus` in this repository is legacy plumbing no shipped model uses.

Claim 3 is the initialization that makes the swap free: a 6x6 stride-2 conv contains a 3x3 stride-2 conv
exactly, so pretrained weights transfer with no layer left randomly initialized.

Run:
    python scripts/anomaly_bench/spd_equivalence.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# Running this as a file puts sys.path[0] at scripts/anomaly_bench/, so `import ultralytics` would resolve
# to whatever the editable install points at rather than the checkout this script lives in.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ultralytics.nn.modules import Conv  # noqa: E402

C, C2, H = 5, 7, 16
PAR = {0: (0, 0), 1: (1, 0), 2: (0, 1), 3: (1, 1)}  # Focus concat order -> (row parity, col parity)


def space_to_depth(x: torch.Tensor) -> torch.Tensor:
    """Return `Focus.forward`'s slicing, i.e. everything it does before the conv."""
    return torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1)


def as_focus_kernel(w6: torch.Tensor) -> torch.Tensor:
    """Reindex a 6x6 stride-2 kernel into the equivalent Focus 3x3 kernel over 4C channels."""
    wn = torch.zeros(w6.shape[0], 4 * w6.shape[1], 3, 3, dtype=w6.dtype)
    for p, (rp, cp) in PAR.items():
        for dr in range(3):
            for dc in range(3):
                wn[:, p * w6.shape[1] : (p + 1) * w6.shape[1], dr, dc] = w6[:, :, 2 * dr + rp, 2 * dc + cp]
    return wn


def main() -> None:
    """Check the three claims and assert each holds to floating point precision."""
    x = torch.randn(1, C, H, H, dtype=torch.float64)

    # 1. A stride-2 3x3 conv embeds exactly into Focus(k=3): place it at [1:4, 1:4] of the 6x6 window.
    w3 = torch.randn(C2, C, 3, 3, dtype=torch.float64)
    w6 = torch.zeros(C2, C, 6, 6, dtype=torch.float64)
    w6[:, :, 1:4, 1:4] = w3
    d1 = (F.conv2d(x, w3, stride=2, padding=1) - F.conv2d(space_to_depth(x), as_focus_kernel(w6), padding=1)).abs().max()

    # 2. Focus(k=3) and Conv(k=6, s=2) are the same operator for arbitrary weights.
    w6r = torch.randn(C2, C, 6, 6, dtype=torch.float64)
    d2 = (
        (F.conv2d(x, w6r, stride=2, padding=2) - F.conv2d(space_to_depth(x), as_focus_kernel(w6r), padding=1))
        .abs()
        .max()
    )

    # 3. The same embedding makes Ultralytics' Conv(k=6, s=2, p=2) a drop-in for Conv(k=3, s=2).
    #    p=2 is mandatory: autopad gives p=3 for k=6, which changes the output size.
    a, b = Conv(16, 32, 3, 2).eval().double(), Conv(16, 32, 6, 2, 2).eval().double()
    with torch.no_grad():
        b.conv.weight.zero_()
        b.conv.weight[:, :, 1:4, 1:4] = a.conv.weight
        b.bn.load_state_dict(a.bn.state_dict())
    xi = torch.randn(1, 16, 64, 64, dtype=torch.float64)
    with torch.inference_mode():
        d3 = (a(xi) - b(xi)).abs().max()
    autopad_shape = tuple(Conv(16, 32, 6, 2)(xi.float()).shape)

    print(f"1  stride-2 3x3 embeds into Focus(k=3)        max|diff| = {d1:.3e}")
    print(f"2  Conv(k=6,s=2) == Focus(k=3)                max|diff| = {d2:.3e}")
    print(f"3  Conv(k=6,s=2,p=2) drop-in for Conv(k=3,s=2) max|diff| = {d3:.3e}")
    print(f"   taps/params per channel: k=3 {C * C2 * 9}, Focus(k=3) {4 * C * C2 * 9}, k=6 {C * C2 * 36}")
    print(f"   k=6 with autopad gives {autopad_shape} instead of (1, 32, 32, 32) -- p=2 is required")
    for name, d in (("1", d1), ("2", d2), ("3", d3)):
        assert d < 1e-12, f"claim {name} failed: max|diff| = {d:.3e}"
    print("all three hold")


if __name__ == "__main__":
    main()
