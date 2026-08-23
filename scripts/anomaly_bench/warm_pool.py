# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Can the pretrained box head see targets the inside-GT pool throws away?

Stage 1 of `TaskAlignedAssigner` (centre inside GT) is model-blind, stage 2 (`align_metric`) is not:
`intersect_dicts` transfers the box head from the donor whenever its shape matches, so at step 0 the
IoU term is already a competent COCO localiser while the cls head is still at its `bias_init` value.
Every knob in the A-wave registry moves stage 1 only, because the 3cad stability law kills any arm
that takes the ranking away from the model.

That leaves one combination untried: let the warm head vote on *membership* instead of order. This
measures whether such a vote would have anything to say -- per GT, how many anchors sit OUTSIDE the
inside-GT pool while their predicted box already matches the GT. Similarity is `bbox_nwd`, not IoU,
because IoU vanishes for the small defects the question is about; here its flatness is the point.

If `out@tau` is ~0 the union gate is a no-op and the idea dies for free. If it is large for the
starved GTs (`pool < topk`) and small for healthy ones, the gate adds candidates exactly where the
pool is the binding constraint.

`zeros` is the tie-bug exposure from the same pass: in-pool anchors whose `align_metric` is 0 and so
lose the topk tie to out-of-pool anchors that `mask_in_gts` then deletes.

Measures the o2m head at epoch 0 and aborts after `--steps`, so the weights are the donor's. Nothing
in `ultralytics/` is modified. `--epochs` stays at the arm's real value for the reason spelled out in
`assigner_probe.py`; the run is aborted long before the schedule matters.

Usage:
    python scripts/anomaly_bench/warm_pool.py --data <data.yaml> --name warm_3cad --device 0
    python scripts/anomaly_bench/warm_pool.py --data <data.yaml> --name warm_3cad --taus 0.3 0.5 0.7
"""

import argparse
import csv
import sys
from pathlib import Path

import torch

# Running this as a file puts sys.path[0] at scripts/anomaly_bench/, so `import ultralytics` would resolve
# to whatever the editable install points at rather than the checkout this script lives in.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ultralytics import YOLO  # noqa: E402
from ultralytics.utils import LOGGER  # noqa: E402
from ultralytics.utils.metrics import bbox_nwd  # noqa: E402
from ultralytics.utils.tal import TaskAlignedAssigner  # noqa: E402


class _Done(Exception):
    """Raised inside the probe to abort training once `--steps` batches are measured."""


def install_probe(out: Path, steps: int, taus: list[float]) -> list:
    """Wrap `TaskAlignedAssigner._forward` to write one row per GT for the first `steps` o2m calls."""
    fields = ["step", "w", "h", "pool", "p3", "p4", "p5", "zeros", "nwd_max_out"]
    fields += [f"out@{t:g}" for t in taus] + [f"in@{t:g}" for t in taus]
    rows: list[dict] = []
    original = TaskAlignedAssigner._forward
    state = {"calls": 0}
    handle = open(out, "w", newline="")
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()

    def probed(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt, anc_strides=None):
        result = original(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt, anc_strides)
        if self.topk != 10:  # o2o head; the pool question is about o2m, which E2ELoss builds at topk=10
            return result
        state["calls"] += 1
        if state["calls"] > steps:
            raise _Done
        with torch.no_grad():
            valid = mask_gt.squeeze(-1).bool()  # (b, n)
            inside = self.select_candidates_in_gts(anc_points, gt_bboxes, mask_gt)  # (b, n, a)
            align, _ = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, inside * mask_gt)
            # the gate a union prior would use: the warm head's own opinion, scored on the GT's scale
            nwd = bbox_nwd(pd_bboxes.unsqueeze(1), gt_bboxes.unsqueeze(2)).squeeze(-1)  # (b, n, a)
            pool = inside & valid[..., None]
            outside = ~inside & valid[..., None]
            lvl = anc_strides.squeeze(-1)  # (a,) per-anchor stride identifies the level
            per_level = [(pool & (lvl == s)).sum(-1) for s in sorted({float(s) for s in lvl.tolist()})]
            per_level += [torch.zeros_like(per_level[0])] * (3 - len(per_level))
            stats = dict(
                pool=pool.sum(-1),
                p3=per_level[0],
                p4=per_level[1],
                p5=per_level[2],
                zeros=((align <= 0) & pool).sum(-1),
                nwd_max_out=nwd.masked_fill(~outside, 0).amax(-1),
            )
            for t in taus:
                stats[f"out@{t:g}"] = ((nwd >= t) & outside).sum(-1)
                stats[f"in@{t:g}"] = ((nwd >= t) & pool).sum(-1)
            wh = (gt_bboxes[..., 2:] - gt_bboxes[..., :2]).clamp(min=0)
            stats = {k: v.float().cpu() for k, v in stats.items()}
            wh, valid = wh.cpu(), valid.cpu()

            for b, n in valid.nonzero().tolist():
                row = dict(step=state["calls"], w=round(float(wh[b, n, 0]), 2), h=round(float(wh[b, n, 1]), 2))
                row |= {k: round(float(v[b, n]), 4) for k, v in stats.items()}
                rows.append(row)
                writer.writerow(row)
            handle.flush()
        return result

    TaskAlignedAssigner._forward = probed
    return rows


def summarize(rows: list[dict], taus: list[float], topk: int) -> str:
    """Aggregate the per-GT rows, split by whether the inside-GT pool is the binding constraint."""
    if not rows:
        return "no rows"
    out = [f"{len(rows)} GTs measured, split at pool < topk({topk})"]
    for tag, sel in (("starved (pool<topk)", lambda r: r["pool"] < topk), ("healthy", lambda r: r["pool"] >= topk)):
        g = [r for r in rows if sel(r)]
        if not g:
            continue
        mean = lambda k: sum(r[k] for r in g) / len(g)  # noqa: E731
        out.append(
            f"  {tag:22s} n={len(g):6d} ({100 * len(g) / len(rows):5.1f}%)  "
            f"short={sum(min(r['w'], r['h']) for r in g) / len(g):6.1f}px  "
            f"pool={mean('pool'):6.2f} (P3/P4/P5 {mean('p3'):.2f}/{mean('p4'):.2f}/{mean('p5'):.2f})  "
            f"zeros={mean('zeros'):5.2f}  best_nwd_outside={mean('nwd_max_out'):.4f}"
        )
        for t in taus:
            out.append(
                f"    tau={t:<4g} outside pool & warm: {mean(f'out@{t:g}'):8.2f}/GT   "
                f"inside pool & warm: {mean(f'in@{t:g}'):6.2f}/GT   "
                f"GTs gaining >=1: {100 * sum(r[f'out@{t:g}'] >= 1 for r in g) / len(g):5.1f}%"
            )
    return "\n".join(out)


def main() -> None:
    """Run a few epoch-0 batches with a measuring assigner and report the warm-vs-geometric pool gap."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="yolo26n.pt")
    ap.add_argument("--epochs", type=int, default=100, help="keep at the arm's value; the run aborts at --steps")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--device", default="0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=10, help="o2m calls to measure before aborting")
    ap.add_argument("--taus", type=float, nargs="+", default=[0.3, 0.5, 0.7], help="NWD gate thresholds to sweep")
    ap.add_argument("--topk", type=int, default=10, help="o2m tal_topk, the pool<topk split point")
    ap.add_argument("--project", default="runs/yolo26-defect-bench")
    ap.add_argument("--name", required=True)
    args = ap.parse_args()

    out = Path(args.project) / args.name
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "warm_pool.csv"
    rows = install_probe(csv_path, args.steps, args.taus)

    try:
        YOLO(args.model).train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            seed=args.seed,
            project=args.project,
            name=args.name,
            exist_ok=True,
        )
    except _Done:
        LOGGER.info(f"warm_pool: aborted after {args.steps} o2m calls, weights still the donor's")
    finally:
        LOGGER.info(f"warm_pool: {len(rows)} rows in {csv_path}\n{summarize(rows, args.taus, args.topk)}")


if __name__ == "__main__":
    main()
