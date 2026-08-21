# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""How many anchors can a GT box actually recruit?

`TaskAlignedAssigner` only considers anchors whose centre falls inside the GT box, then keeps the
top-`topk` of those. So `topk` is only the binding constraint when the in-box pool is larger than
`topk`; for tiny defects the pool is the constraint and no amount of topk tuning can widen it.
This measures the pool directly from a COCO GT json, at the strides YOLO26 detects on.

Usage:
    python runs/tests/anchor_pool.py <gt_val.json> [--imgsz 640 960] [--topk 10]
"""

import argparse
import json
from collections import Counter
from pathlib import Path

STRIDES = (8, 16, 32)  # P3, P4, P5


def pool_sizes(boxes: list[tuple[float, float]], strides=STRIDES) -> Counter:
    """Count anchors per GT: centres on a `stride` grid that land inside a `w x h` box.

    Mirrors `TaskAlignedAssigner.select_candidates_in_gts`, including the clamp that inflates any
    side below the smallest stride up to that stride.
    """
    c = Counter()
    for w, h in boxes:
        w, h = max(w, strides[0]), max(h, strides[0])  # tal.py inflates sub-stride sides
        c[sum(max(0, int(w // s)) * max(0, int(h // s)) for s in strides)] += 1
    return c


def main() -> None:
    """Report the in-box anchor-pool distribution for each requested image size."""
    ap = argparse.ArgumentParser()
    ap.add_argument("gt", type=Path)
    ap.add_argument("--imgsz", type=int, nargs="+", default=[640, 960])
    ap.add_argument("--topk", type=int, default=10, help="o2m tal_topk to compare the pool against")
    args = ap.parse_args()

    gt = json.loads(args.gt.read_text())
    dims = {i["id"]: max(i["width"], i["height"]) for i in gt["images"]}
    raw = [(a["bbox"][2], a["bbox"][3], dims[a["image_id"]]) for a in gt["annotations"]]
    print(f"{args.gt.parent.name}: {len(raw)} GT boxes")

    for imgsz in args.imgsz:
        boxes = [(w * imgsz / d, h * imgsz / d) for w, h, d in raw]
        c = pool_sizes(boxes)
        n = sum(c.values())
        starved = sum(v for k, v in c.items() if k < args.topk)
        med = sorted(k for k, v in c.items() for _ in range(v))[n // 2]
        print(
            f"  imgsz={imgsz:4d}  median pool={med:5d}  "
            f"pool<topk({args.topk}): {starved:5d} ({100 * starved / n:5.1f}%)  "
            f"pool<=1: {c[0] + c[1]:5d} ({100 * (c[0] + c[1]) / n:5.1f}%)"
        )


if __name__ == "__main__":
    main()
