# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""How many anchors can a GT box actually recruit?

`TaskAlignedAssigner` only considers anchors whose centre falls inside the GT box, then keeps the
top-`topk` of those. So `topk` is only the binding constraint when the in-box pool is larger than
`topk`; for tiny defects the pool is the constraint and no amount of topk tuning can widen it.
This measures the pool directly from a COCO GT json, at the strides YOLO26 detects on.

The clamp is reproduced exactly, including its non-monotone legacy branch: a side below `stride[0]`
jumps to `stride[1]`, while a side in `[stride[0], stride[1])` is left alone and therefore recruits
*fewer* anchors than a smaller one. `--min-side` replaces it with the plain floor that `tal_min_side`
installs, so the dose of that knob can be chosen from the pool it produces instead of guessed.

Usage:
    python scripts/anomaly_bench/anchor_pool.py <gt_val.json> [--imgsz 640 960] [--topk 10]
    python scripts/anomaly_bench/anchor_pool.py <gt_val.json> --imgsz 640 --min-side 8 16 24 32
"""

import argparse
import json
from collections import Counter
from pathlib import Path

STRIDES = (8, 16, 32)  # P3, P4, P5


def pool_sizes(boxes: list[tuple[float, float]], min_side: float | None = None, strides=STRIDES) -> Counter:
    """Count anchors per GT: centres on a `stride` grid that land inside a `w x h` box.

    Mirrors `TaskAlignedAssigner.select_candidates_in_gts`, including its clamp on the GT sides.
    """
    c = Counter()
    for w, h in boxes:
        if min_side:
            w, h = max(w, min_side), max(h, min_side)
        else:  # legacy clamp: below strides[0] jumps to strides[1], the band between them is untouched
            w = strides[1] if w < strides[0] else w
            h = strides[1] if h < strides[0] else h
        c[sum(max(0, int(w // s)) * max(0, int(h // s)) for s in strides)] += 1
    return c


def report(tag: str, boxes: list[tuple[float, float]], topk: int, min_side: float | None) -> None:
    """Print the pool distribution for one (imgsz, min_side) cell."""
    c = pool_sizes(boxes, min_side)
    n = sum(c.values())
    starved = sum(v for k, v in c.items() if k < topk)
    med = sorted(k for k, v in c.items() for _ in range(v))[n // 2]
    print(
        f"  {tag}  median pool={med:5d}  "
        f"pool<topk({topk}): {starved:5d} ({100 * starved / n:5.1f}%)  "
        f"pool<=1: {c[0] + c[1]:5d} ({100 * (c[0] + c[1]) / n:5.1f}%)  "
        f"pool==0: {c[0]:5d} ({100 * c[0] / n:5.1f}%)"
    )


def main() -> None:
    """Report the in-box anchor-pool distribution for each requested image size."""
    ap = argparse.ArgumentParser()
    ap.add_argument("gt", type=Path)
    ap.add_argument("--imgsz", type=int, nargs="+", default=[640, 960])
    ap.add_argument("--topk", type=int, default=10, help="o2m tal_topk to compare the pool against")
    ap.add_argument("--min-side", type=float, nargs="+", help="tal_min_side doses to sweep; omit for the legacy clamp")
    args = ap.parse_args()

    gt = json.loads(args.gt.read_text())
    dims = {i["id"]: max(i["width"], i["height"]) for i in gt["images"]}
    raw = [(a["bbox"][2], a["bbox"][3], dims[a["image_id"]]) for a in gt["annotations"]]
    print(f"{args.gt.parent.name}: {len(raw)} GT boxes")

    for imgsz in args.imgsz:
        boxes = [(w * imgsz / d, h * imgsz / d) for w, h, d in raw]
        band = sum(STRIDES[0] <= min(w, h) < STRIDES[1] for w, h in boxes)
        print(f"  imgsz={imgsz:4d}  short side in [{STRIDES[0]}, {STRIDES[1]}) px: {100 * band / len(boxes):5.1f}%")
        report(f"imgsz={imgsz:4d}  min_side=legacy", boxes, args.topk, None)
        for ms in args.min_side or []:
            report(f"imgsz={imgsz:4d}  min_side={ms:6.1f}", boxes, args.topk, ms)


if __name__ == "__main__":
    main()
