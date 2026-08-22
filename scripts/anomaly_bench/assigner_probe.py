# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Why does an assignment knob diverge on one dataset but not another?

Wave A saw `tal_prior=rfla` and `tal_metric=nwd` peak at epoch 6-22 on `3cad` and collapse to
mAP 0, while the identical configs trained cleanly on `tianchifabirc` and `dspcbsd`. The
hypothesis is that the assigner's total soft label collapses: `v8DetectionLoss.loss` divides the
classification BCE by `max(target_scores.sum(), 1)`, so once the sum falls under 1 the clamp
takes over and `loss[1]` stops being a mean and becomes a raw sum over bs*anchors*nc terms --
two orders of magnitude larger, and the gradient shock is unrecoverable.

This wraps `TaskAlignedAssigner._forward` to log, per step and per head, exactly the quantities
that hypothesis is about. Nothing in `ultralytics/` is modified, so a probe run is a normal run
plus a CSV.

Read the output like this:
    - `tgt_sum` sliding under 1.0 while `cls_loss` jumps  -> hypothesis confirmed
    - `tgt_sum` healthy while mAP still dies              -> hypothesis dead, look elsewhere
    - `fg` (positive count) going to 0                    -> the prior, not the soft label

Usage:
    python scripts/anomaly_bench/assigner_probe.py --data <data.yaml> --epochs 15 --device 0 \
        --name probe_rfla --tal_prior rfla
"""

import argparse
import csv
import sys
from pathlib import Path

# Running this as a file puts sys.path[0] at scripts/anomaly_bench/, so `import ultralytics` would resolve
# to whatever the editable install points at rather than the checkout this script lives in.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ultralytics import YOLO  # noqa: E402
from ultralytics.utils import LOGGER  # noqa: E402
from ultralytics.utils.tal import TaskAlignedAssigner  # noqa: E402

KNOBS = ("tal_min_side", "tal_prior", "tal_rf_scale", "tal_metric", "tal_nwd_gamma")


def install_probe(out: Path, every: int) -> list:
    """Wrap `TaskAlignedAssigner._forward` so every `every`-th call appends one row per head."""
    rows: list[dict] = []
    original = TaskAlignedAssigner._forward
    state = {"calls": 0}

    def probed(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt, anc_strides=None):
        result = original(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt, anc_strides)
        _, _, target_scores, fg_mask, _ = result
        # topk distinguishes the heads without threading extra state: E2ELoss builds o2m at 10, o2o at 7
        head = "o2m" if self.topk == 10 else "o2o"
        if head == "o2m":
            state["calls"] += 1
        if state["calls"] % every == 0:
            n_gt = int(mask_gt.sum())
            tgt_sum = float(target_scores.sum())
            rows.append(
                dict(
                    call=state["calls"],
                    head=head,
                    n_gt=n_gt,
                    fg=int(fg_mask.sum()),
                    # the quantity the max(..., 1) clamp acts on -- under 1.0 the cls loss stops being a mean
                    tgt_sum=round(tgt_sum, 6),
                    clamped=int(tgt_sum < 1.0),
                    tgt_max=round(float(target_scores.max()), 6),
                    fg_per_gt=round(int(fg_mask.sum()) / max(n_gt, 1), 3),
                )
            )
        return result

    TaskAlignedAssigner._forward = probed
    return rows


def main() -> None:
    """Train briefly with a probed assigner and write the per-step assignment health to CSV."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="yolo26n.pt")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--device", default="0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--project", default="runs/yolo26-defect-bench")
    ap.add_argument("--name", required=True)
    ap.add_argument("--every", type=int, default=10, help="log every Nth optimizer step")
    for k in KNOBS:
        ap.add_argument(f"--{k}")
    args = ap.parse_args()

    out = Path(args.project) / args.name
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "assigner_probe.csv"
    rows = install_probe(csv_path, args.every)

    knobs = {k: v for k in KNOBS if (v := getattr(args, k)) is not None}
    knobs = {k: (float(v) if k in {"tal_min_side", "tal_rf_scale", "tal_nwd_gamma"} else v) for k, v in knobs.items()}
    LOGGER.info(f"assigner_probe: knobs={knobs or 'defaults'}  csv={csv_path}")

    try:
        YOLO(args.model).train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            seed=args.seed,
            coco_eval=True,
            project=args.project,
            name=args.name,
            exist_ok=True,
            **knobs,
        )
    finally:  # a divergence that ends in an exception is exactly the case whose log we need
        if rows:
            with open(csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
            o2m = [r for r in rows if r["head"] == "o2m"]
            clamped = sum(r["clamped"] for r in o2m)
            LOGGER.info(
                f"assigner_probe: wrote {len(rows)} rows to {csv_path}\n"
                f"  o2m steps logged: {len(o2m)}   steps with target_scores.sum() < 1: {clamped} "
                f"({100 * clamped / max(len(o2m), 1):.1f}%)\n"
                f"  tgt_sum  min={min(r['tgt_sum'] for r in o2m):.4f}  max={max(r['tgt_sum'] for r in o2m):.4f}\n"
                f"  fg_per_gt min={min(r['fg_per_gt'] for r in o2m):.2f}  "
                f"max={max(r['fg_per_gt'] for r in o2m):.2f}"
            )


if __name__ == "__main__":
    main()
