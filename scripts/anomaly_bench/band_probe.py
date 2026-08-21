# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Validate pinning the scale bands to a fixed letterbox frame.

Scaling every GT box AND every prediction in an image by the same factor leaves IoU unchanged, so AP_all
must be bit-identical while only the small/medium/large split moves. If that holds, rescaling into a pinned
reference frame is a safe way to redefine the bands.
"""
import json, sys
from pathlib import Path
from faster_coco_eval import COCO, COCOeval_faster

def ev(gt, preds):
    c = COCO(gt); e = COCOeval_faster(c, c.loadRes(preds), "bbox", print_function=lambda *a: None)
    e.params.imgIds = sorted(c.imgs); e.evaluate(); e.accumulate(); e.summarize()
    s = e.stats_as_dict
    return {k: round(s[v], 6) for k, v in
            {"AP": "AP_all", "AP50": "AP_50", "S": "AP_small", "M": "AP_medium", "L": "AP_large"}.items()}

d = Path(sys.argv[1]); ref = int(sys.argv[2]) if len(sys.argv) > 2 else 640
gt = json.loads((d / "gt_val.json").read_text()); preds = json.loads((d / "predictions.json").read_text())
print(f"{d.name}  original frame: {ev(d/'gt_val.json', preds)}")

r = {i["id"]: ref / max(i["width"], i["height"]) for i in gt["images"]}
g2 = {"images": [{**i, "width": round(i["width"]*r[i["id"]]), "height": round(i["height"]*r[i["id"]])} for i in gt["images"]],
      "categories": gt["categories"],
      "annotations": [{**a, "bbox": [v*r[a["image_id"]] for v in a["bbox"]], "area": a["area"]*r[a["image_id"]]**2}
                      for a in gt["annotations"]]}
p2 = [{**p, "bbox": [v*r[p["image_id"]] for v in p["bbox"]]} for p in preds]
tmp = d.parent / "_band_gt.json"; tmp.write_text(json.dumps(g2))
print(f"{d.name}  letterbox-{ref}:  {ev(tmp, p2)}")
tmp.unlink()

# WRONG variant: rescale GT area only, leaving predictions in the original frame.
g3 = {**gt, "annotations": [{**a, "area": a["area"]*r[a["image_id"]]**2} for a in gt["annotations"]]}
tmp.write_text(json.dumps(g3))
print(f"{d.name}  GT-area-only:    {ev(tmp, preds)}   <- unmatched detections are still banded by their own area")
tmp.unlink()
