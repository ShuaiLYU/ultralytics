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

`--csv` reads a `warm_pool.csv` instead and splits the same question by aspect ratio, which separates
the two failure modes: a small square defect is short of candidates and is cured by widening the box
(`tal_min_side`, confirmed on 3cad), while a sliver has plenty and is not short of anything countable.

Usage:
    python scripts/anomaly_bench/anchor_pool.py <gt_val.json> [--imgsz 640 960] [--topk 10]
    python scripts/anomaly_bench/anchor_pool.py <gt_val.json> --imgsz 640 --min-side 8 16 24 32
    python scripts/anomaly_bench/anchor_pool.py --csv runs/.../warm_pool.csv
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import median

STRIDES = (8, 16, 32)  # P3, P4, P5
AR_BUCKETS = ((1, 2), (2, 4), (4, 8), (8, 16), (16, float("inf")))


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


def report(tag: str, boxes: list[tuple[float, float]], topk: int, min_side: float | None, strides=STRIDES) -> None:
    """Print the pool distribution for one (imgsz, min_side) cell."""
    c = pool_sizes(boxes, min_side, strides)
    n = sum(c.values())
    starved = sum(v for k, v in c.items() if k < topk)
    med = sorted(k for k, v in c.items() for _ in range(v))[n // 2]
    print(
        f"  {tag}  median pool={med:5d}  "
        f"pool<topk({topk}): {starved:5d} ({100 * starved / n:5.1f}%)  "
        f"pool<=1: {c[0] + c[1]:5d} ({100 * (c[0] + c[1]) / n:5.1f}%)  "
        f"pool==0: {c[0]:5d} ({100 * c[0] / n:5.1f}%)"
    )


def report_ar(rows: list[dict], topk: int) -> None:
    """Split the measured GTs by aspect ratio: is this group short of candidates, or short of levels?

    A small square defect is short of candidates and `tal_min_side` cures it. A sliver is not: by
    AR>=8 the pool is already past `topk` and by AR>=16 nothing is starved at all, so there is no
    count to fix.

    What the level columns show is that there is no composition to fix either. A GT's per-level pool
    is its area over that level's stride squared, so every GT -- tiny, huge, square, sliver -- hands
    P3 the same ~76% of its candidates. Widening the box scales all three levels together and leaves
    the share untouched, which is why S1/S2 could only flood P3. Only a prior that replaces the pool
    with a level-aware rule (`rfla`, `ar_rfla`) can move a sliver to a coarse level.

    A short side under one cell does NOT zero that level: anchor centres sit at `(i + 0.5) * stride`,
    so a 11.5 px side still catches a P4 row whenever it straddles one. Hence `P3-only` stays in the
    single digits even for the most extreme slivers.

    Level shares are averaged per GT, not summed over the group: a sum is dominated by the largest
    boxes and returns the same 76/19/5 for every group, measuring the grid rather than the GTs.
    """
    print(f"{len(rows)} GTs, aspect ratio = long/short, sizes as the assigner saw them (post-augmentation)")
    print(
        f"  {'AR':>9} {'n':>6} {'share':>6} {'short':>7} {'long':>8} {'pool':>7} "
        f"{'P3/P4/P5 per GT':>17} {'P3-only':>8} {'starved':>8}"
    )
    for lo, hi in AR_BUCKETS:
        g = [r for r in rows if lo <= r["ar"] < hi]
        if not g:
            continue
        share = [sum(r[k] / max(r["pool"], 1) for r in g) / len(g) for k in ("p3", "p4", "p5")]
        print(
            f"  {lo:>4g}-{hi:<4g} {len(g):6d} {100 * len(g) / len(rows):5.1f}% "
            f"{median(min(r['w'], r['h']) for r in g):7.1f} {median(max(r['w'], r['h']) for r in g):8.1f} "
            f"{median(r['pool'] for r in g):7.1f} "
            f"{100 * share[0]:8.0f}/{100 * share[1]:3.0f}/{100 * share[2]:3.0f}% "
            f"{100 * sum(r['p4'] + r['p5'] == 0 for r in g) / len(g):7.1f}% "
            f"{100 * sum(r['pool'] < topk for r in g) / len(g):7.1f}%"
        )


def main() -> None:
    """Report the in-box anchor-pool distribution for each requested image size."""
    ap = argparse.ArgumentParser()
    ap.add_argument("gt", type=Path, nargs="?")
    ap.add_argument("--csv", type=Path, help="warm_pool.csv to split by aspect ratio instead of a COCO json")
    ap.add_argument("--imgsz", type=int, nargs="+", default=[640, 960])
    ap.add_argument("--topk", type=int, default=10, help="o2m tal_topk to compare the pool against")
    ap.add_argument("--min-side", type=float, nargs="+", help="tal_min_side doses to sweep; omit for the legacy clamp")
    ap.add_argument("--strides", type=int, nargs="+", default=list(STRIDES), help="detection strides, e.g. 4 8 16 32")
    args = ap.parse_args()

    strides = tuple(sorted(args.strides))
    if args.csv:
        rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(args.csv.read_text().splitlines())]
        for r in rows:
            r["ar"] = max(r["w"], r["h"]) / max(min(r["w"], r["h"]), 1e-9)
        print(f"{args.csv.parent.name}: ", end="")
        report_ar(rows, args.topk)
        return

    gt = json.loads(args.gt.read_text())
    dims = {i["id"]: max(i["width"], i["height"]) for i in gt["images"]}
    raw = [(a["bbox"][2], a["bbox"][3], dims[a["image_id"]]) for a in gt["annotations"]]
    print(f"{args.gt.parent.name}: {len(raw)} GT boxes, strides={strides}")

    for imgsz in args.imgsz:
        boxes = [(w * imgsz / d, h * imgsz / d) for w, h, d in raw]
        band = sum(strides[0] <= min(w, h) < strides[1] for w, h in boxes)
        print(f"  imgsz={imgsz:4d}  short side in [{strides[0]}, {strides[1]}) px: {100 * band / len(boxes):5.1f}%")
        report(f"imgsz={imgsz:4d}  min_side=legacy", boxes, args.topk, None, strides)
        for ms in args.min_side or []:
            report(f"imgsz={imgsz:4d}  min_side={ms:6.1f}", boxes, args.topk, ms, strides)


if __name__ == "__main__":
    main()
