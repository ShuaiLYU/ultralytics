# run.md — YOLO26 baselines on the defect benchmark

Logbook for branch `yolo26-defect-bench`. Assignment: [TASK.md](TASK.md).
All metrics rounded to 4 decimal places. Every entry states the `--snap` commit, weight, and dataset yaml.

## Decisions taken (deviations from TASK.md)

| TASK.md says                         | We do                                          | Why                                                                                                                                                 |
| ------------------------------------ | ---------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `yolo26s.pt`                         | **`yolo26n.pt`**                               | Many loss ablations follow; matches the structure/loss design doc's baseline. `n` has wider seed variance, so the noise floor will be wider too.    |
| noise floor on 3 datasets (proposed) | **`dspcbsd` only**                             | Louis's call. Consequence: no σ on `3cad`/`tianchifabirc`, so on those two a delta can only be judged by direction, not significance.               |
| `project=runs/anomaly_bench`         | **`project=yolo26-defect-bench`**              | `nohupyolo` rejects a `project=` containing slashes. Output still lands in `runs/yolo26-defect-bench/`. Name matches the branch.                    |
| do not modify `ultralytics/`         | **modified, behind `coco_eval` (default off)** | Louis's call. AP_small/medium/large is the primary readout for every planned change and is not otherwise obtainable. Non-pollution is proven below. |

## Code changes on this branch

| commit      | what                                                                                                                                                   |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `e9801c281` | `coco_eval` bool arg + `converter.yolo2coco_gt` helper (inert)                                                                                         |
| `2f9cc1af2` | run COCO-style eval on any dataset under `coco_eval`; relax the `is_coco or is_lvis` gates; hoist the `eval_json` call so it also runs during training |
| `b9e441328` | integer image ids — `faster-coco-eval` casts ids to `int` internally, so non-numeric filenames crashed                                                 |

Adds five per-epoch columns to `results.csv`: `metrics/{mAP50,mAP50-95,mAP_small,mAP_medium,mAP_large}(B-coco)`.

**Invariant: the generic path is purely additive.** Native `metrics/*(B)` columns and `fitness` are not
touched, so `best.pt` selection and the training trajectory are identical to a `coco_eval=False` run.
Without this guard, `coco_evaluate` would overwrite `fitness` with `0.9*AP + 0.1*AP50` (default is pure
mAP50-95) and silently change which checkpoint every run keeps.

---

# Phase 0 — smoke tests

Purpose: confirm the data loads and the `coco_eval` path works. **Not** for numbers — 2-5 epochs on 5-20%
of the data, so every AP below is noise and is recorded only as evidence the pipeline runs end to end.

## 0a. Non-pollution A/B (local, coco8)

Ran on the laptop, CPU, `yolo26n.pt`, `coco8.yaml`, 2 epochs, imgsz 320, batch 4, seed 0 — once with
`coco_eval=True`, once with `False`.

```bash
python -c "
from ultralytics import YOLO
YOLO('yolo26n.pt').train(data='coco8.yaml', epochs=2, imgsz=320, batch=4, device='cpu', seed=0,
    project='runs/tests', name='ce_on', exist_ok=True, coco_eval=True, plots=False, val=True)
"
```

**Result: all 14 shared `results.csv` columns bit-identical** (only `time` differs), including
`metrics/mAP50-95(B)` and `fitness`. 5 new `(B-coco)` columns present, one row per epoch. Re-verified
unchanged after the integer-id commit `b9e441328`.

## 0b. ultra6 smoke, three datasets

Snap commit **`b9e441328749`** · weight `yolo26n.pt` (official name, auto-downloaded) · GPU 6 and 7.

```bash
EXP=/Users/louis/workspace/ultra_louis_work/expman/.venv/bin/expman-cli

$EXP bundle

$EXP launch --snap --args "nohupyolo 0 train data=/data/shared-datasets/louis_data/anomaly_bench/3cad/data.yaml model=yolo26n.pt epochs=2 fraction=0.05 imgsz=640 seed=0 device=6 batch=32 coco_eval=True project=yolo26-defect-bench name=smoke_3cad_n_v2"

$EXP launch --snap --args "nohupyolo 0 train data=/data/shared-datasets/louis_data/anomaly_bench/tianchifabirc/data.yaml model=yolo26n.pt epochs=5 fraction=0.2 imgsz=640 seed=0 device=7 batch=32 coco_eval=True project=yolo26-defect-bench name=smoke_tianchifabirc_n_v2"

$EXP launch --snap --args "nohupyolo 0 train data=/data/shared-datasets/louis_data/anomaly_bench/dspcbsd/data.yaml model=yolo26n.pt epochs=5 fraction=0.2 imgsz=640 seed=0 device=6 batch=32 coco_eval=True project=yolo26-defect-bench name=smoke_dspcbsd_n_v2"
```

Logs: `runs/yolo26-defect-bench/<name>.log` on ultra6, pulled to
`expman/data/pulled/yolo26-defect-bench/<name>/`.

### Data loading

| dataset         | train imgs (fraction) | val imgs | val backgrounds | val instances | corrupt | missing | nc  |
| --------------- | --------------------- | -------- | --------------- | ------------- | ------- | ------- | --- |
| `3cad`          | 1,071 (0.05)          | 2,575    | 1,471           | 1,801         | 0       | 0       | 24  |
| `tianchifabirc` | 2,066 (0.2)           | 1,185    | 0               | 1,973         | 0       | 0       | 20  |
| `dspcbsd`       | 1,315 (0.2)           | 1,633    | 0               | 3,167         | 0       | 0       | 9   |

Matches TASK.md on every count that document states: `3cad` val = 1,471 good + 1,104 defect = 2,575;
`tianchifabirc` and `dspcbsd` carry no normal images; class counts 24 / 20 / 9.

### coco_eval works on all three

| dataset         | GT json                 | COCO AP50 | native mAP50 | COCO AP50-95 | native mAP50-95 | AP_S   | AP_M   | AP_L   |
| --------------- | ----------------------- | --------- | ------------ | ------------ | --------------- | ------ | ------ | ------ |
| `3cad`          | 2,575 imgs / 1,801 anns | 0.0040    | 0.0040       | 0.0030       | 0.0032          | 0.0000 | 0.0000 | 0.0070 |
| `tianchifabirc` | 1,185 / 1,973           | —         | —            | —            | —               | 0.0160 | 0.0220 | 0.0110 |
| `dspcbsd`       | 1,633 / 3,167           | —         | —            | —            | —               | 0.0740 | 0.1050 | 0.1300 |

Two independent evaluators agreeing on `3cad` (`0.0040` vs `0.0040`, `0.0030` vs `0.0032`) is the
correctness check on the generated ground truth: coordinate frame, image ids, and category mapping are right.

**Area bands are not degenerate.** The concern was that COCO's absolute thresholds (small `<32²`,
large `>96²`) would collapse to a single band on `3cad`, where 69% of boxes are under 0.1% of image area
— but 0.1% of a large source image is still hundreds of px². All three bands report a real value rather
than COCO's `-1.000` empty-band sentinel on all three datasets, so the standard bands are kept. No
per-dataset tercile split needed.

### Things worth knowing

- **`3cad` val contains 21 of 24 classes.** Missing: `ink`, `fracture`, `wear_crack` — the three rarest
  in the whole dataset (6, 9, 12 boxes). The 24-class vocabulary is intact (`nc=24` confirmed);
  Ultralytics only prints classes that have val instances.
- **`train/l1_loss`, not `train/dfl_loss`.** Confirms `reg_max=1` on YOLO26 routes box supervision
  through CIoU **plus an L1 term on normalized ltrb**, not CIoU alone. This matters for the planned NWD
  work: NWD's `W₂²` is itself a direct-coordinate L₂ distance, so it overlaps the existing L1 term, not
  just CIoU. Test `dfl=0` before writing any NWD code.
- **First `tianchifabirc` attempt produced zero predictions** (2 epochs, `fraction=0.05`), so
  `len(self.jdict) == 0` and the COCO block was skipped entirely — the path was untested, not passing.
  Re-run at 5 epochs / `fraction=0.2` to get predictions. Recorded because a skipped block looks like a
  pass in the log.
- Superseded runs, kept for traceability: `smoke_3cad_n`, `smoke_dspcbsd_n` (commit `2f9cc1af2`, before
  the integer-id fix; `3cad` failed with `invalid literal for int(): '3cad-aluminum-pc__train_good_000126'`),
  `smoke_tianchifabirc_n` (zero predictions).

### Open item — needs Louis

`3cad` is yolo-seg polygons trained as `task=detect`, so Ultralytics derives the boxes. Visual check is
**not done by me**; images pulled and opened for review:

```
expman/data/pulled/yolo26-defect-bench/smoke_3cad_n_v2/train_batch0.jpg
expman/data/pulled/yolo26-defect-bench/smoke_3cad_n_v2/val_batch0_labels.jpg
```

Phase A is gated on this check.
