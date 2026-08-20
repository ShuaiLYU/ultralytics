# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Evaluate finished defect-benchmark runs on any split, reporting COCO-style AP_small/medium/large.

Training only ever validates the `val` split, and `model.val()` returns a DetMetrics object whose
`results_dict` omits the `(B-coco)` keys that `coco_eval` adds. Calling the validator directly returns the
full stats dict instead, which is the only way to get AP_small/medium/large on a non-COCO dataset.

The dataset yaml and image size are read back from each run's own `args.yaml`, so a reported number cannot
drift away from the run that produced it.

Examples:
    Evaluate every run of a project on both splits
    >>> python eval_bench.py --project runs/yolo26-defect-bench --splits val test --device 0

    Evaluate specific runs
    >>> python eval_bench.py --runs runs/yolo26-defect-bench/dspcbsd_baseline_n_s0 --splits test
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER, YAML
from ultralytics.utils.checks import check_file

ND = 4  # decimal places, uniform across the project
KEYS = ("mAP50", "mAP50-95", "mAP_small", "mAP_medium", "mAP_large")


def evaluate(run: Path, split: str, device: str, batch: int, out: Path) -> tuple[dict, list[dict]] | None:
    """Validate one run's best.pt on one split and return its overall and per-class metrics."""
    weights, cfg = run / "weights" / "best.pt", run / "args.yaml"
    for p in (weights, cfg):
        if not p.is_file():
            LOGGER.warning(f"{run.name}: {p.name} missing, skipping")
            return None
    args = YAML.load(cfg)
    data = args["data"]  # a bare yaml name resolves against the package cfg dirs, an absolute path as-is
    if split != "val" and not YAML.load(check_file(data)).get(split):
        LOGGER.warning(f"{run.name}: {data} declares no '{split}' split, skipping")
        return None

    v = DetectionValidator(
        args=dict(
            model=str(weights),
            data=data,
            split=split,
            imgsz=args["imgsz"],
            batch=batch or args["batch"],
            device=device,
            coco_eval=True,
            plots=False,
            project=str(out),
            name=f"{run.name}__{split}",
            exist_ok=True,
        )
    )
    stats = v()

    overall = {"run": run.name, "split": split, "data": data, "weights": str(weights)}
    overall |= {f"native_{k}": round(float(stats.get(f"metrics/{k}(B)", float("nan"))), ND) for k in KEYS[:2]}
    overall |= {k: round(float(stats.get(f"metrics/{k}(B-coco)", float("nan"))), ND) for k in KEYS}

    m = v.metrics
    per_class = [
        {
            "run": run.name,
            "split": split,
            "class": m.names[c],
            "instances": int(m.nt_per_class[c]),
            **dict(zip(("P", "R", "mAP50", "mAP50-95"), (round(float(x), ND) for x in m.class_result(i)))),
        }
        for i, c in enumerate(m.ap_class_index)
    ]
    return overall, per_class


def write(rows: list[dict], path: Path) -> None:
    """Write rows to CSV and echo them as a markdown table for pasting into run.md."""
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    cols = [c for c in rows[0] if c not in {"data", "weights"}]
    LOGGER.info(f"\n{path}\n")
    LOGGER.info("| " + " | ".join(cols) + " |")
    LOGGER.info("|" + "---|" * len(cols))
    for r in rows:
        LOGGER.info("| " + " | ".join(str(r[c]) for c in cols) + " |")


def main() -> None:
    """Evaluate the requested runs and splits, writing overall and per-class CSVs."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", type=Path, help="evaluate every run directory under this project")
    p.add_argument("--runs", type=Path, nargs="+", help="specific run directories")
    p.add_argument("--splits", nargs="+", default=["val", "test"])
    p.add_argument("--device", default="0")
    p.add_argument("--batch", type=int, default=0, help="0 reuses each run's training batch size")
    p.add_argument("--out", type=Path, help="output directory, defaults to <project>/eval")
    a = p.parse_args()

    runs = sorted(a.runs or [d for d in a.project.iterdir() if (d / "args.yaml").is_file()])
    if not runs:
        raise SystemExit("no run directories found")
    out = a.out or (a.project or runs[0].parent) / "eval"
    out.mkdir(parents=True, exist_ok=True)

    overall, per_class = [], []
    for run in runs:
        for split in a.splits:
            if r := evaluate(run, split, a.device, a.batch, out):
                overall.append(r[0])
                per_class.extend(r[1])

    write(overall, out / "overall.csv")
    write(per_class, out / "per_class.csv")


if __name__ == "__main__":
    main()
