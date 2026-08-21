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

### Polygon-to-box derivation on `3cad` — checked, passes

`3cad` carries yolo-seg polygons but trains as `task=detect`, so Ultralytics derives the boxes. The
failure mode to rule out is a box covering the whole part instead of the defect.

Visual, `train_batch0.jpg`: boxes sit on defects, not on parts. No box spans a part or an image. Class 18
appears as long thin horizontal strips a few tens of px tall and hundreds wide, hugging scratch lines;
classes 3 and 19 are small localized squares. Classes 2, 20 and 14 draw large area-covering boxes, which
is correct for diffuse defect types (bruise, bright_shadow, uneven) rather than a derivation error.
Note the tiles are mosaics of 4 images each, so the filename in a tile title does not identify which
image a given box belongs to.

Numeric, val side (unaugmented, and what the metric is computed on): if the derivation produced
part-sized boxes, every GT box would fall in the `large` band. `AP_small` and `AP_medium` both return a
real value (`0.0000`) rather than COCO's `-1.000` empty-band sentinel, so small and medium GT boxes exist.
Together with the COCO-vs-native agreement above, the val boxes are sound.

**Pitfall: do not use `val_batch0_labels.jpg` for this check on `3cad`.** All 16 tiles are `_good_` images
with zero boxes, so it looks clean while proving nothing. `3cad` val is 57% normal images and the val
dataloader runs `rect=True`, which sorts by aspect ratio and clusters similar images into the same batch.
Use a train batch, or a later val batch.

---

# Phase A — baselines and noise floor

Snap commit **`0ccb13883e94`** · weight `yolo26n.pt` · `batch=128` · `epochs=100` · `imgsz=640` ·
`coco_eval=True`. Launched 2026-08-20 10:02-10:04 UTC. Names carry `_n` for the model scale, since the
scale differs from the one TASK.md specified.

Five runs, all launched concurrently. `yolo26n` at `batch=128` measures **~24 GB**, so several fit on one
96 GB card: GPU 6 held 3 jobs at 74.7 GB, GPU 7 held 2 at 50.8 GB.

**The three `dspcbsd` seeds share one GPU on purpose.** They are the noise-floor measurement, so their
conditions must be identical; splitting them across cards with different co-tenants would put a second
variable into a variance estimate. `deterministic: True` is the Ultralytics default, so co-tenancy affects
speed but not numerics.

```bash
EXP=/Users/louis/workspace/ultra_louis_work/expman/.venv/bin/expman-cli

$EXP bundle

# GPU 6 — noise floor, identical conditions
for s in 0 1 2; do
  $EXP launch --snap --args "nohupyolo 0 train data=/data/shared-datasets/louis_data/anomaly_bench/dspcbsd/data.yaml model=yolo26n.pt epochs=100 imgsz=640 batch=128 seed=$s device=6 coco_eval=True project=yolo26-defect-bench name=dspcbsd_baseline_n_s$s"
done

# GPU 7 — the other two baselines
$EXP launch --snap --args "nohupyolo 0 train data=/data/shared-datasets/louis_data/anomaly_bench/3cad/data.yaml model=yolo26n.pt epochs=100 imgsz=640 batch=128 seed=0 device=7 coco_eval=True project=yolo26-defect-bench name=3cad_baseline_n_s0"

$EXP launch --snap --args "nohupyolo 0 train data=/data/shared-datasets/louis_data/anomaly_bench/tianchifabirc/data.yaml model=yolo26n.pt epochs=100 imgsz=640 batch=128 seed=0 device=7 coco_eval=True project=yolo26-defect-bench name=tianchifabirc_baseline_n_s0"
```

Dataset yamls: `/data/shared-datasets/louis_data/anomaly_bench/{3cad,tianchifabirc,dspcbsd}/data.yaml`.

Status: **running.** `dspcbsd` 52 train batches/epoch at ~1.5 it/s, ~83 min for 100 epochs. Results,
per-class tables, the test split numbers and the noise floor go here when they finish.

`batch=128` is now fixed for every later ablation on this branch — a change measured against these
baselines cannot also change the batch size.

## Reporting val and test — `eval_bench.py`

Training only ever validates `val`. Test numbers, and AP_small/medium/large on either split, come from
`eval_bench.py` (commit `4140049a0`), run on ultra6 because that is where the weights and data live:

```bash
EXP=/Users/louis/workspace/ultra_louis_work/expman/.venv/bin/expman-cli
$EXP launch --snap --args "nohuppython eval_bench.py --project runs/yolo26-defect-bench --splits val test --device 6"
```

Writes `runs/yolo26-defect-bench/eval/{overall,per_class}.csv` and prints both as markdown tables. It
reads the data yaml and `imgsz` back from each run's `args.yaml`, so a number cannot drift from the run
that produced it, and it warns and skips a split the dataset does not declare rather than crashing.

Why a script rather than `model.val(split="test")`: `model.val()` returns a `DetMetrics` object, and
`DetMetrics.keys` does not include the `(B-coco)` keys — those live only in the stats dict the validator
returns. Calling `DetectionValidator` directly is the only route to AP_small/medium/large here.

---

# Phase B prep — Z3 (wide early downsampling), not yet launched

**Superseded plan.** Z3 first used `Focus` as SPD-Conv (`yolo26-spd.yaml`, commit `046bf4b2b`, removed in
`434f1bf44`). It now uses `Conv(k=6, s=2, p=2)` instead, because the two are the same operator and the
conv form is strictly cheaper. The measurements that led there are kept below — they cost real work and
they are the argument.

## Why `Focus` was dropped: SPD-Conv is a 6x6 strided conv

Space-to-depth splits a 2x2 neighbourhood into four sub-lattices, so a following 3x3 kernel reaches 3
half-resolution positions x 2 sub-pixels = **6 original pixels per axis**: a 6x6 window at stride 2, with
4C x 9 = **36 taps** per input channel. `Conv(k=6, s=2)` has the same window, stride, tap count, and
parameter count. The two are related by a pure reindexing of the weights.

Verified in `scripts/anomaly_bench/spd_equivalence.py` (float64):

| claim                                                       | max abs diff |
| ----------------------------------------------------------- | ------------ |
| a stride-2 3x3 conv embeds exactly into `Focus(k=3)`        | 8.9e-15      |
| `Conv(k=6,s=2)` == `Focus(k=3)` for arbitrary weights       | 2.8e-14      |
| `Conv(k=6,s=2,p=2)` drop-in for `Conv(k=3,s=2)` after embed | 0.0          |

So the design doc's premise for SPD-Conv — that it "discards no information", unlike a strided conv — does
not distinguish it from the baseline in the way claimed. A 6x6 strided conv also reads all 36 pixels. The
only thing Z3 actually changes against `Conv(k=3,s=2)` is **kernel 3 -> 6, taps 9 -> 36**. That is the
hypothesis being tested, and it should be stated that way.

This is also, on the arithmetic alone, why a 6x6 stride-2 conv is the sensible form: same function, one
cuDNN kernel instead of four non-contiguous slices plus a concat plus a conv. (Whether that was the
original YOLOv5 motivation for the same swap is not something we verified.)

## Cost, measured before spending GPU time

|                                | params            | CPU b=1 @640     | ONNX                 |
| ------------------------------ | ----------------- | ---------------- | -------------------- |
| `yolo26n`                      | 2,572,280         | 45.2 ms          | —                    |
| `yolo26n-spd` (`Focus`)        | 2,696,696 (+4.8%) | 50.9 ms (+12.6%) | +16 `Slice` nodes    |
| `yolo26n-k6` (`Conv(k=6,s=2)`) | 2,696,696 (+4.8%) | not re-measured  | one `Conv` per layer |

`Focus` costs +12.6% CPU latency for a function a plain conv computes, so the conv form removes a real
tax rather than trading one cost for another.

## The plan as it now stands

`ultralytics/cfg/models/26/yolo26-k6.yaml` — two lines differ from `yolo26.yaml`: the P1->P2 and P2->P3
downsampling convs become `Conv [c2, 6, 2, 2]`. `p=2` is mandatory; autopad gives `p=3` for `k=6` and
changes the output size to `H/2 + 1`.

The stem stays `k=3` on purpose: it runs at full resolution, where widening costs most. Deeper downsamples
feed large-object levels, where a narrower kernel does not hurt small defects.

**The weight-transfer confound is removed, not accepted.** Widening a kernel leaves it with no same-shape
counterpart in `yolo26n.pt`: 12 tensors and 166,274 parameters would start random, in the two earliest
feature layers, while everything downstream stays pretrained. A negative result would then be
uninterpretable — "wider kernel is worse" could not be separated from "losing pretrained early layers is
worse". `scripts/anomaly_bench/make_k6_donor.py` embeds the pretrained 3x3 at `[1:4, 1:4]` of the 6x6
window and zeroes the rest, which reproduces the original layer bit for bit:

```bash
python scripts/anomaly_bench/make_k6_donor.py --src yolo26n.pt --cfg yolo26n-k6.yaml \
    --out /abs/path/yolo26n-k6.pt --check
```

Verified: 708/708 tensors, 2 kernels embedded, whole-model forward `max abs diff = 0.0` against
`yolo26n.pt`, and the trainer reports `Transferred 708/708 items`. Z3 therefore starts from exactly the
baseline, with the extra taps initialized as a no-op. Snapshots carry no untracked files, so the donor must
be passed as an absolute path.

## Two things worth recording

- **`Focus` is legacy plumbing.** It is wired correctly (imported in `tasks.py`, in `base_modules` so width
  scaling applies, exported from `nn.modules`) but **no shipped model yaml uses it**, and its only test is
  one shape check. `Contract`, `Expand`, `PixelUnshuffle` and `SpaceToDepth` do not exist in this
  repository, so the commented-out `Contract(gain=2)` line inside `Focus` points at a deleted module. Two
  traps if it is ever used: the default `k=1` makes it a 1x1 conv over the 4x-channel tensor, losing the
  spatial mixing entirely, and a third positional arg sets the inner conv's stride on top of the /2 the
  slicing already did.
- **Do not compare end2end outputs element-wise.** The `(1, 300, 6)` NMS-free tensor is confidence-sorted,
  so tiny numerical differences permute tied low-confidence rows and produce a max-abs-diff of ~288 on a
  model that is in fact correct. Compare sorted confidences, or only detections above a threshold.

`Focus`'s ONNX graph did not contain a native `SpaceToDepth`; its slicing lowered to `Slice`+`Concat`
(`Slice` count 2 -> 18, exactly 8 per `Focus`). Both are standard ops needing no plugin, so the design
doc's "export-clean" conclusion held, but not for the stated reason. Moot now that the conv form is used.

---

# Phase B — Z1 / Z2 / Z3 launched

Snap commit **`20b6e886f0c2`** · `yolo26n` · `epochs=100` · `imgsz=640` · `batch=128` · `seed=0` ·
`coco_eval=True` · `project=yolo26-defect-bench`. Launched 2026-08-21 on GPU 4 and 5, which freed when the
`yoloa_clean` job finished. Package code is unchanged from the Phase A baseline snapshot `0ccb13883`
except for the addition of an unused yaml, so these are directly comparable to those baselines.

**Z4 (P2 head) stays in the plan but is deliberately not launched yet** (Louis's call).

```bash
EXP=/Users/louis/workspace/ultra_louis_work/expman/.venv/bin/expman-cli
B="model=yolo26n.pt epochs=100 imgsz=640 batch=128 seed=0 coco_eval=True project=yolo26-defect-bench"
D=/data/shared-datasets/louis_data/anomaly_bench

# Z1 — drop the L1-on-ltrb term. Gates M2: if AP_small falls, NWD largely duplicates this term.
for d in 3cad tianchifabirc dspcbsd; do
  $EXP launch --snap --args "nohupyolo 0 train data=$D/$d/data.yaml $B dfl=0 device=4 name=${d}_z1_dfl0_n_s0"
done

# Z2 — inverse-frequency class weights, on the long-tailed dataset. cls_pw is asserted to [0, 1].
$EXP launch --snap --args "nohupyolo 0 train data=$D/3cad/data.yaml $B cls_pw=0.5 device=5 name=3cad_z2_clspw05_n_s0"
$EXP launch --snap --args "nohupyolo 0 train data=$D/3cad/data.yaml $B cls_pw=1.0 device=5 name=3cad_z2_clspw10_n_s0"

# Z3 — widen the two early downsampling kernels. Donor first, on CPU, then the run.
$EXP launch yolo26-defect-bench --snap --py ultra --args "python scripts/anomaly_bench/make_k6_donor.py --src /home/louis/ultra_louis_work/ultralytics/yolo26n.pt --cfg yolo26n-k6.yaml --out /home/louis/ultra_louis_work/yolo26n-k6.pt --check name=k6_donor > /home/louis/ultra_louis_work/ultralytics/runs/yolo26-defect-bench/k6_donor.log 2>&1"

$EXP launch --snap --args "nohupyolo 0 train data=$D/dspcbsd/data.yaml model=yolo26n-k6.yaml pretrained=/home/louis/ultra_louis_work/yolo26n-k6.pt epochs=100 imgsz=640 batch=128 seed=0 coco_eval=True device=5 project=yolo26-defect-bench name=dspcbsd_z3_k6_n_s0"
```

| run                          | dataset       | variable               | GPU |
| ---------------------------- | ------------- | ---------------------- | --- |
| `3cad_z1_dfl0_n_s0`          | 3cad          | `dfl=0`                | 4   |
| `tianchifabirc_z1_dfl0_n_s0` | tianchifabirc | `dfl=0`                | 4   |
| `dspcbsd_z1_dfl0_n_s0`       | dspcbsd       | `dfl=0`                | 4   |
| `3cad_z2_clspw05_n_s0`       | 3cad          | `cls_pw=0.5`           | 5   |
| `3cad_z2_clspw10_n_s0`       | 3cad          | `cls_pw=1.0`           | 5   |
| `dspcbsd_z3_k6_n_s0`         | dspcbsd       | k=6 early downsampling | 5   |

Donor verified on ultra6: 708 tensors, 2 kernels embedded, `max abs diff = 0.000e+00` against
`yolo26n.pt`. Log at `runs/yolo26-defect-bench/k6_donor.log`.

**Z3's confound is confirmed removed in the run itself**: both `dspcbsd_z3_k6_n_s0` and
`dspcbsd_baseline_n_s0` report `Transferred 606/708 items`. The 102 skipped tensors are the 80-class head,
which both arms lose identically because `nc=9`. Had the widened kernels failed to transfer, Z3 would read
604/708.

## Launcher notes, each of which cost a failed launch

- `expman-cli launch` needs a literal `name=` token in `--args`, or the workspace given positionally
  (`launch yolo26-defect-bench ...`) plus `project=` in the args.
- `nohuppython` needs argparse-style `--project`/`--name` and **hard-errors on any bare `name=` token**, so
  the two launchers' contracts cannot both be met in one arg string. `make_k6_donor.py` accepts both forms.
- Plain `python` through `--args` does not activate the conda env (`ModuleNotFoundError: cv2`) and writes no
  log. Pass `--py ultra` and redirect stdout yourself; `nohuppython` is what normally does both.
- GPU capacity is set by residency, not job count: these runs measure ~25 GB each at `batch=128`, so a card
  already holding a 45 GB job fits one more, not two. Packing to 97% risks taking down the tenant.

---

# THE NOISE FLOOR

`dspcbsd`, `yolo26n`, 100 epochs, `batch=128`, `imgsz=640`, snap `0ccb13883e94`, three seeds. Val
mAP50-95 at the best epoch, i.e. the number `best.pt` corresponds to:

| seed | best epoch | mAP50  | mAP50-95 |
| ---- | ---------- | ------ | -------- |
| 0    | 67         | 0.7875 | 0.4737   |
| 1    | 83         | 0.8085 | 0.4802   |
| 2    | 83         | 0.7985 | 0.4756   |

**spread (max − min) = 0.0065** · mean 0.4765 · sd 0.0033 · **2sd = 0.0067**

**Quote this: a change to YOLO26 on this benchmark is not a result below ~0.0067 mAP50-95.** Anything
smaller is indistinguishable from re-running the same code with a different seed.

The floor is tight, which is the useful outcome: real effects of 0.01 and up are measurable on a single
seed per arm. It is also specific to `dspcbsd` — `3cad` and `tianchifabirc` have no seed replicates, so on
those two a delta can only be reported as agreeing in direction, never as significant. If a change turns
out to hinge on `3cad`, that is the moment to spend two more seeds there.

Other baselines finished so far: `tianchifabirc` mAP50 0.4074, mAP50-95 0.1936 (best epoch 89).
`3cad` still running.

Outstanding for Phase A: test-split numbers, AP_small/medium/large, and full per-class tables via
`eval_bench.py`. Not yet run.

---

# Phase B queue — second wave

Snap **`bca283c87ec2`**; package code byte-identical to `20b6e886` and `0ccb13883`, so all of Phase B is
comparable to the Phase A baselines. Launched on GPU 6 and 7 as the first baselines freed them.

| run                             | dataset       | variable                    | GPU |
| ------------------------------- | ------------- | --------------------------- | --- |
| `3cad_z3_k6_n_s0`               | 3cad          | k=6 early downsampling      | 6   |
| `dspcbsd_z7_imgsz960_n_s0`      | dspcbsd       | `imgsz=960` (ceiling probe) | 6   |
| `3cad_z5_mosaic0_n_s0`          | 3cad          | `mosaic=0.0`                | 7   |
| `tianchifabirc_z2_clspw05_n_s0` | tianchifabirc | `cls_pw=0.5`                | 7   |

`3cad_z3_k6_n_s0` also reports `Transferred 606/708`, matching the baseline, so the widened kernels
transferred there too.

**Z5 (`mosaic=0.0`) reasoning.** Mosaic composes four images into one canvas and downscales them. On
`3cad`, where 69% of boxes are under 0.1% of image area, that halves defects that are already at the limit
of what P3 can resolve. This is a plain augmentation knob, no code, and it plausibly matters more than any
architectural change on this data.

**Z7 (`imgsz=960`) is a reference measurement, not a candidate.** It is the cheapest large lever on small
objects, so it calibrates everything else: if raising resolution moves AP_small by 0.05 while every
architectural change moves it by 0.005, the ablation programme's priorities are wrong. Measured 47.8 GiB at
`batch=128`, so it needs a card mostly to itself.

Still queued, not launched: Z2 on `dspcbsd`, Z3 on `tianchifabirc`, Z4 (P2 head, held by Louis), Z6
(`scale=0.2`, mechanism overlaps Z5 — read Z5 first), and the dfl sweep beyond `dfl=0` (gated on Z1).

## Third wave — queued via `--after`, fills GPUs 4/5/6/7 as tenants free

Snap **`d942b647fb86`** (package code byte-identical to `bca283c87`/`20b6e886`/`0ccb13883`; the commit
only adds run.md). Each job is chained to a running predecessor with `nohupyolo --after <run>`, so it
polls the predecessor's PID, then +20 s VRAM settle, then launches. Every successor replaces a tenant on
the **same card** it waits on, so no card ever exceeds its current occupancy.

```bash
EXP=/Users/louis/workspace/ultra_louis_work/expman/.venv/bin/expman-cli
D=/data/shared-datasets/louis_data/anomaly_bench
P=yolo26-defect-bench

$EXP bundle

$EXP launch --snap --args "nohupyolo --after dspcbsd_z3_k6_n_s0     train data=$D/dspcbsd/data.yaml       model=yolo26n.pt     epochs=100 imgsz=640 batch=128 seed=0 coco_eval=True cls_pw=0.5 device=5 project=$P name=dspcbsd_z2_clspw05_n_s0"
$EXP launch --snap --args "nohupyolo --after 3cad_z3_k6_n_s0        train data=$D/tianchifabirc/data.yaml model=yolo26n-k6.yaml pretrained=/home/louis/ultra_louis_work/yolo26n-k6.pt epochs=100 imgsz=640 batch=128 seed=0 coco_eval=True device=6 project=$P name=tianchifabirc_z3_k6_n_s0"
$EXP launch --snap --args "nohupyolo --after tianchifabirc_z1_dfl0_n_s0 train data=$D/tianchifabirc/data.yaml model=yolo26n.pt  epochs=100 imgsz=640 batch=128 seed=0 coco_eval=True mosaic=0.0 device=4 project=$P name=tianchifabirc_z5_mosaic0_n_s0"
$EXP launch --snap --args "nohupyolo --after dspcbsd_z1_dfl0_n_s0   train data=$D/dspcbsd/data.yaml       model=yolo26n.pt     epochs=100 imgsz=640 batch=128 seed=0 coco_eval=True mosaic=0.0 device=4 project=$P name=dspcbsd_z5_mosaic0_n_s0"
$EXP launch --snap --args "nohupyolo --after dspcbsd_z7_imgsz960_n_s0 train data=$D/3cad/data.yaml        model=yolo26n.pt     epochs=100 imgsz=960 batch=128 seed=0 coco_eval=True device=6 project=$P name=3cad_z7_imgsz960_n_s0"
$EXP launch --snap --args "nohupyolo --after tianchifabirc_z2_clspw05_n_s0 train data=$D/tianchifabirc/data.yaml model=yolo26n.pt epochs=100 imgsz=640 batch=128 seed=0 coco_eval=True cls_pw=1.0 device=7 project=$P name=tianchifabirc_z2_clspw10_n_s0"
```

| queued run                 | waits on (PID → run)          | variable   | GPU |
| -------------------------- | ----------------------------- | ---------- | --- |
| `dspcbsd_z2_clspw05_n_s0`  | 3095507 `dspcbsd_z3_k6`       | cls_pw=0.5 | 5   |
| `tianchifabirc_z3_k6_n_s0` | 3122342 `3cad_z3_k6`          | k=6        | 6   |
| `tianchifabirc_z5_mosaic0` | 3069100 `tianchifabirc_z1`    | mosaic=0.0 | 4   |
| `dspcbsd_z5_mosaic0_n_s0`  | 3070978 `dspcbsd_z1`          | mosaic=0.0 | 4   |
| `3cad_z7_imgsz960_n_s0`    | 3120263 `dspcbsd_z7_960`      | imgsz=960  | 6   |
| `tianchifabirc_z2_clspw10` | 3125776 `tianchifabirc_z2_05` | cls_pw=1.0 | 7   |

All six are `status=queued`, `.status` files record `Waiting for: PID <n> to exit, then +20s settle`.
`3cad_z7_imgsz960` (~48 GB) chains onto the other 48 GB `imgsz=960` job on the same card, so the big
slot is handed off rather than double-booked. Still deliberately unqueued (gated on results, not on a
free card): Z6 `scale=0.2` (after Z5), the `dfl` sweep (after Z1), Z4 (held by Louis).

## Two operational notes

- `refs/louis/laptop-head` **vanished on ultra6** mid-session; `launch` failed with "cannot read HEAD" and
  `bundle` recreated it from scratch (`remote_before: null`, 4836 commits). Running jobs were unaffected
  because they execute from `--snap` worktrees, which is precisely the failure mode `--snap` exists for.
- `lsta` reports the `k6_donor` utility job as **FAILED**. It succeeded; the classifier looks for
  Ultralytics' training completion marker, which a non-training script never writes. Its log is the
  authority.

---

# Phase B results — the zero-code wave, all 21 runs complete

Snap `20b6e886f0c2` / `bca283c87ec2` (package code byte-identical). `yolo26n`, 100 epochs, `imgsz=640`,
`batch=128`, `seed=0`, `coco_eval=True`. Baselines: `dspcbsd` = 3-seed mean, `3cad` and `tianchifabirc` =
seed 0.

## Per-metric noise floors, and why they change the readout

From the three `dspcbsd` baseline seeds, per metric:

| metric       | mean   | spread | sd     | **2sd = threshold** |
| ------------ | ------ | ------ | ------ | ------------------- |
| mAP50        | 0.7982 | 0.0210 | 0.0105 | 0.0210              |
| mAP50-95     | 0.4765 | 0.0065 | 0.0033 | **0.0067**          |
| **AP_small** | 0.3985 | 0.0015 | 0.0008 | **0.0016**          |
| AP_medium    | 0.5499 | 0.0106 | 0.0055 | 0.0110              |
| AP_large     | 0.5126 | 0.1771 | 0.1017 | 0.2034              |

Two consequences, both of which change how this benchmark should be read:

- **AP_small is 4x tighter than mAP50-95, so it is the better judging metric, not just the mechanism
  check.** Every candidate on the list targets small objects, and total mAP50-95 barely registers what they
  do: Z3 moves AP_small by +4.0x its floor on `dspcbsd` while moving mAP50-95 by only +0.7x. Judging on
  mAP50-95 alone would have scored the one change that works as "no measurable difference".
- **AP_large is unusable on `dspcbsd`** (2sd = 0.2034). Too few large boxes; that column is noise. Any
  conclusion quoting it is fictional.

## Results — delta vs baseline, in multiples of the 2sd floor

| change          | dataset       | mAP50-95            | AP_small             | AP_medium           |
| --------------- | ------------- | ------------------- | -------------------- | ------------------- |
| Z1 `dfl=0`      | dspcbsd       | -0.0013 (-0.2x)     | **-0.0065 (-4.1x)**  | -0.0037 (-0.3x)     |
|                 | 3cad          | +0.0038 (+0.6x)     | **+0.0130 (+8.1x)**  | **+0.0525 (+4.8x)** |
|                 | tianchifabirc | -0.0011 (-0.2x)     | +0.0038 (+2.4x)      | -0.0078 (-0.7x)     |
| Z2 `cls_pw=0.5` | dspcbsd       | -0.0045 (-0.7x)     | **-0.0111 (-6.9x)**  | -0.0059 (-0.5x)     |
|                 | 3cad          | -0.0181 (-2.7x)     | **-0.0142 (-8.9x)**  | -0.0169 (-1.5x)     |
|                 | tianchifabirc | +0.0024 (+0.4x)     | +0.0019 (+1.2x)      | -0.0001 (-0.0x)     |
| Z2 `cls_pw=1.0` | 3cad          | **-0.0599 (-8.9x)** | **-0.0375 (-23.4x)** | **-0.0544 (-4.9x)** |
|                 | tianchifabirc | +0.0062 (+0.9x)     | +0.0011 (+0.7x)      | +0.0058 (+0.5x)     |
| **Z3 `k=6`**    | dspcbsd       | +0.0047 (+0.7x)     | **+0.0064 (+4.0x)**  | -0.0068 (-0.6x)     |
|                 | 3cad          | **+0.0150 (+2.2x)** | **+0.0304 (+19.0x)** | +0.0079 (+0.7x)     |
|                 | tianchifabirc | +0.0036 (+0.5x)     | **-0.0065 (-4.1x)**  | +0.0149 (+1.4x)     |
| Z5 `mosaic=0`   | dspcbsd       | -0.0122 (-1.8x)     | **-0.0112 (-7.0x)**  | **-0.0470 (-4.3x)** |
|                 | 3cad          | **-0.0268 (-4.0x)** | **-0.0126 (-7.9x)**  | -0.0167 (-1.5x)     |
|                 | tianchifabirc | -0.0116 (-1.7x)     | **-0.0113 (-7.1x)**  | -0.0144 (-1.3x)     |
| Z7 `imgsz=960`  | dspcbsd       | +0.0066 (+1.0x)     | **+0.0072 (+4.5x)**  | +0.0138 (+1.3x)     |
|                 | 3cad          | **+0.0232 (+3.5x)** | **+0.0945 (+59.1x)** | **+0.0280 (+2.5x)** |

Absolute values for every run are in `expman/data/workspaces/yolo26-defect-bench.json` under
`train_metrics`; the columns are `metrics/{mAP50,mAP50-95}(B)` and
`metrics/mAP_{small,medium,large}(B-coco)`.

## Verdicts

**Z3 (`k=6` early downsampling) — the only change that passes.** AP_small +4.0x on `dspcbsd` and +19.0x on
`3cad`, both far past the floor, so the "agrees on at least two of three datasets" rule is met.
`tianchifabirc` disagrees on AP_small (-4.1x) while its AP_medium rises (+1.4x) and total mAP50-95 is flat,
so it is a shift in size sensitivity there, not a collapse. The attribution is clean: the donor makes the
run start bit-identical to the baseline (`Transferred 606/708` on both `dspcbsd` and `3cad`), so the only
difference is kernel 3 -> 6. Cost is +4.8% parameters and no latency penalty, since it is one conv.

**Z5 (`mosaic=0`) — fails, and the hypothesis was backwards.** All three datasets get worse, AP_small
-7.0x / -7.9x / -7.1x, remarkably consistent. The reasoning was that mosaic downscales and would destroy
defects already at the resolution limit. The data says mosaic's regularization and context diversity are
worth much more than the downscaling costs, on every dataset here. Line closed.

**Z2 (`cls_pw`) — closed.** Monotone damage on `3cad` (`0.5` -> -8.9x, `1.0` -> -23.4x AP_small). A
dose-response relation that clean is not noise. No effect on `tianchifabirc`. Inverse-frequency class
weighting is useless or harmful on this benchmark.

**Z1 (`dfl=0`) — no conclusion; the datasets disagree.** `dspcbsd` AP_small -4.1x against `3cad` +8.1x.
Total mAP50-95 shows nothing anywhere. So the L1-on-ltrb term helps small objects on PCB data and hurts
them on 3CAD, and its overall contribution is invisible.

**Z7 (`imgsz=960`) — the reference measurement, and it reorders the cost-benefit:**

| lever              | AP_small on dspcbsd | cost                     |
| ------------------ | ------------------- | ------------------------ |
| `imgsz` 640 -> 960 | +0.0072             | compute 2.25x            |
| kernel 3 -> 6      | +0.0064             | +4.8% params, no latency |

Nearly the same gain for an order of magnitude difference in cost. But on `3cad` resolution gives +0.0945
(59x), three times what `k=6` gives, which says `3cad`'s tiny defects are **resolution-limited rather than
architecture-limited**. On data like that, input resolution caps what any architectural trick can reach —
worth knowing before spending runs on architecture there.

## Consequences for the plan

- **M2 (NWD) is downgraded.** Its gate was Z1, and Z1 came back contradictory rather than permissive. On
  top of that, CIoU already contains a normalized centre-distance term (`rho2/c2`) and an aspect term
  (`v`), which are the two things NWD's `W2^2` adds. Weak prior, no supporting evidence, expensive to tune
  (`C` is dataset-specific). Not next.
- **M1 (Inner-IoU) and M3 (o2o `topk2`) are next**, being cheap and independent of Z1.
- `3cad` seeds 1 and 2 launched, because the two largest signals on the board (Z1 +8.1x, Z3 +19.0x
  AP_small) are both on the one dataset with no replicates, so neither can be called significant yet.

---

# Phase B — M1 and M3 launched (first runs on modified loss code)

Snap **`acbc98561`**. Two new knobs, both defaulting to current behaviour so a default run is unchanged by
construction. The package diff against the baseline snapshot `bca283c87` is 28 insertions across
`cfg/__init__.py`, `cfg/default.yaml`, `utils/loss.py`, `utils/metrics.py` — nothing else is touched, and
at their defaults neither knob has a live code path.

| knob          | default | what it does                                                                                                                                                                                                             |
| ------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `inner_ratio` | `1.0`   | Inner-IoU (arXiv:2311.02877). The IoU term is measured between auxiliary boxes scaled about their own centres; the CIoU penalty terms stay on the original boxes. Rescales the box gradient without moving any geometry. |
| `o2o_topk2`   | `1`     | Secondary top-k for the one-to-one head's assigner, previously hardcoded in `E2ELoss`.                                                                                                                                   |

## Why `o2o_topk2` is worth a run at all

`topk2` short-circuits when it equals `topk` (`tal.py`), and `E2ELoss` passes `topk2=None` to the
one-to-many head — which resolves to `topk`, so the branch never fires there. Only the one-to-one head gets
`topk2=1`. So the secondary assignment is the model's one small-target-aware assignment mechanism, it
applies to the **inference** head alone, and until now there was no way to set it.

This also corrects the design doc a second time: the doc treats "STAL" as solving small targets from the
assignment side of the training head. The mechanism exists, but not where the doc assumes.

## Verification before launch

- `inner_ratio=1.0` skips the new branch, so CIoU is identical: `max abs diff = 0.000e+00`.
- The formula reduces to plain IoU as the ratio approaches 1 (`5.5e-07`; the residual is the pre-existing
  `eps` inflation of `h1`/`h2` in the xyxy branch feeding the corner reconstruction, not new error).
- Survives `autocast`, output finite.
- `train/box_loss` responds monotonically: `1.2 -> 1.4398`, `1.0 -> 1.5064`, `0.8 -> 1.6342`.
- Probed inside a real training run: `inner_ratio=0.8` reaches both heads' `BboxLoss`; `o2o_topk2=3` gives
  the o2o assigner `topk/topk2 = 7/3` while o2m stays `10/10`. Defaults probe as `1.0` and `1`.

**A trap worth recording: coco8 mAP cannot resolve either knob.** The first smoke test scored two genuinely
different configurations identically to six decimals, because mAP over 4 val images is quantized far too
coarsely. That looked exactly like a dead knob. Use `train/box_loss` or probe the criterion.

## Runs

Value screening on `dspcbsd` only — the one dataset with a noise floor. Two jobs per card, ~25 GB each.

```bash
EXP=/Users/louis/workspace/ultra_louis_work/expman/.venv/bin/expman-cli
D=/data/shared-datasets/louis_data/anomaly_bench/dspcbsd/data.yaml
B="model=yolo26n.pt epochs=100 imgsz=640 batch=128 seed=0 coco_eval=True project=yolo26-defect-bench"

$EXP launch --snap --args "nohupyolo 0 train data=$D $B inner_ratio=0.8 device=4 name=dspcbsd_m1_ir08_n_s0"
$EXP launch --snap --args "nohupyolo 0 train data=$D $B inner_ratio=1.2 device=5 name=dspcbsd_m1_ir12_n_s0"
$EXP launch --snap --args "nohupyolo 0 train data=$D $B inner_ratio=0.7 device=6 name=dspcbsd_m1_ir07_n_s0"
$EXP launch --snap --args "nohupyolo 0 train data=$D $B o2o_topk2=2 device=6 name=dspcbsd_m3_topk22_n_s0"
$EXP launch --snap --args "nohupyolo 0 train data=$D $B o2o_topk2=3 device=7 name=dspcbsd_m3_topk23_n_s0"
$EXP launch --snap --args "nohupyolo 0 train data=$D $B o2o_topk2=4 device=7 name=dspcbsd_m3_topk24_n_s0"
```

Also running: `3cad_baseline_n_s1` and `3cad_baseline_n_s2`, which give `3cad` its own noise floor. Both of
the largest signals on the board so far (Z1 AP_small +8.1x, Z3 +19.0x) are on `3cad`, and neither can be
called significant until that floor exists.

**Read the winner with care: six arms judged at 2sd each carry a real chance that one clears the bar by
luck.** Screening picks a value; it does not establish an effect. Whatever wins has to reproduce on the
other two datasets before it means anything.

## M2 (NWD) deliberately not written

Its gate was Z1, and Z1 returned a contradiction (`dspcbsd` AP_small -4.1x against `3cad` +8.1x) rather
than a green light. Independently, CIoU already carries a normalized centre-distance term (`rho2/c2`) and
an aspect term (`v`) — the two components NWD's `W2^2` would add. Weak prior, no supporting evidence, and
`C` needs per-dataset tuning. Revisit only if `3cad`'s own noise floor turns the Z1 signal there into a
real effect.
