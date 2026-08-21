# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Post-hoc NMS probe: does class-wise NMS recover the precision that `o2o_topk2 > 1` lost?

The one-to-one head of an end2end YOLO26 has no NMS at inference, so it must emit exactly one box
per object. `o2o_topk2 = k` assigns k positives per GT, which should train it to emit k boxes per
object. If duplicates are the mechanism, running NMS over the saved predictions.json recovers the
lost AP; if the head simply learned worse features, NMS changes nothing.

Usage:
    python runs/tests/nms_probe.py <run_dir> [<run_dir> ...] --iou 0.7
"""

import argparse
import json
from pathlib import Path

import numpy as np
from faster_coco_eval import COCO, COCOeval_faster


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    """Greedy NMS on xywh boxes, returning kept indices ordered by descending score."""
    x1, y1 = boxes[:, 0], boxes[:, 1]
    x2, y2 = x1 + boxes[:, 2], y1 + boxes[:, 3]
    areas = boxes[:, 2] * boxes[:, 3]
    order, keep = scores.argsort()[::-1], []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1, yy1 = np.maximum(x1[i], x1[order[1:]]), np.maximum(y1[i], y1[order[1:]])
        xx2, yy2 = np.minimum(x2[i], x2[order[1:]]), np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        order = order[1:][inter / (areas[i] + areas[order[1:]] - inter) <= iou_thr]
    return keep


def apply_nms(preds: list[dict], iou_thr: float) -> list[dict]:
    """Apply per-image per-class NMS to a COCO detection list."""
    groups: dict[tuple, list[dict]] = {}
    for p in preds:
        groups.setdefault((p["image_id"], p["category_id"]), []).append(p)
    out = []
    for g in groups.values():
        boxes = np.array([d["bbox"] for d in g], dtype=np.float64)
        scores = np.array([d["score"] for d in g], dtype=np.float64)
        out += [g[i] for i in nms(boxes, scores, iou_thr)]
    return out


def evaluate(gt_path: Path, preds: list[dict]) -> dict:
    """Run COCO eval and return the AP columns this benchmark tracks."""
    gt = COCO(str(gt_path))
    ev = COCOeval_faster(gt, gt.loadRes(preds), "bbox", print_function=lambda *a: None)
    ev.params.imgIds = sorted(gt.imgs)
    ev.evaluate(), ev.accumulate(), ev.summarize()
    s = ev.stats_as_dict
    return {k: round(s[v], 4) for k, v in
            {"mAP50-95": "AP_all", "mAP50": "AP_50", "AP_S": "AP_small", "AP_M": "AP_medium", "AP_L": "AP_large"}.items()}


def main() -> None:
    """Report AP before and after post-hoc NMS for each run directory."""
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+", type=Path)
    ap.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold (matches default.yaml iou)")
    args = ap.parse_args()

    for d in args.run_dirs:
        preds = json.loads((d / "predictions.json").read_text())
        kept = apply_nms(preds, args.iou)
        print(f"\n{d.name}  ({len(preds)} dets -> {len(kept)} after NMS, {100 * (1 - len(kept) / len(preds)):.1f}% removed)")
        for tag, p in (("raw   ", preds), ("+NMS  ", kept)):
            print(f"  {tag} " + "  ".join(f"{k}={v:.4f}" for k, v in evaluate(d / "gt_val.json", p).items()))


if __name__ == "__main__":
    main()
