# run.md — YOLO26 baselines on the defect benchmark

Logbook for branch `yolo26-defect-bench`. Assignment: [TASK.md](TASK.md).
All metrics rounded to 4 decimal places. Every entry states the `--snap` commit, weight, and dataset yaml.

# CURRENT STATE — the one place to read first (2026-08-23 12:10)

Everything below this section is chronological history; snapshots there were true when written and may be
superseded. THIS section is the current truth. Naming, metric, and table rules live in [CONVENTIONS.md](CONVENTIONS.md); new runs and tables follow it. Protocol locked: `yolo26n`, 100 ep, `imgsz=640`,
`batch=128`, `coco_eval=True`, 3 seeds/arm; best epoch by `metrics/mAP50-95(B)`, AP_small from the same row.

## Confirmed winners (3 seeds, p < 0.05)

| dataset       | config      | args                                              | mAP50-95                      | AP_small                         | cost                    |
| ------------- | ----------- | ------------------------------------------------- | ----------------------------- | -------------------------------- | ----------------------- |
| 3cad          | **A1 ms16** | `tal_min_side=16`                                 | 0.2972 (+1.51x fl, p=0.048)   | 0.2373 (**+2.39x fl, p=0.0062**) | zero                    |
| 3cad          | A1 ms32     | `tal_min_side=32`                                 | 0.3035 (+3.01x, p=0.031)      | 0.2281 (+1.90x, p=0.008)         | zero                    |
| 3cad          | k=6         | `yolo26n-k6.yaml` + donor                         | 0.3064 (+3.71x, p=0.049)      | 0.2393 (+2.50x, p=0.044)         | 1.35x FLOPs             |
| tianchifabirc | **A7**      | `tal_min_side=16 tal_prior=ar_rfla tal_ar_rfla=4` | 0.2161 (**+5.26x, p=0.0035**) | 0.1275 (+0.65x, p=0.14)          | zero, ~12% slower train |
| tianchifabirc | A2 rfla     | `tal_prior=rfla`                                  | 0.2158 (+5.19x, p=0.0069)     | 0.1335 (+1.36x, p=0.028)         | zero                    |
| dspcbsd       | —           | —                                                 | nothing moves it              | (saturated)                      | —                       |

Baselines (3 seeds): 3cad 0.2908 / 0.1925; tianchifabirc 0.1909 / 0.1220; dspcbsd 0.4765 / 0.3985.
Noise floors (2sd): 3cad 0.0042 / 0.0187; tianchifabirc 0.0048 / 0.0084; dspcbsd 0.0067 / 0.0016.
The two winners do not stack with k=6 (combinations equal the larger single arm on both datasets).

## Closed lines (with the one-line reason)

- **A2.2 `rfla_fill`** — no-op everywhere tested (3cad n=3 mAP +0.60x p=0.578; tianchifabirc n=1 +0.35x).
- **A3 `tal_metric=nwd` (all betas)** — collapses (beta 6, 2) or below baseline (beta 1: -7.07x).
  Control `ciou beta=1` is -1.57x, so nwd itself is the -5.5x.
- **A4 k6 x assigner** — no stacking (k6xms32 AP_S +2.56x p=0.041 ≈ k6 alone; k6xrfla p=0.158 vs rfla).
- **S1/S2 short-side inflation** — wrong lever: inflation floods P3 (floor 16/32/48/64 -> P3 160/320/480/640
  vs P5 0/0/40/40), and the model does not choose coarse anchors on its own. S2 cancelled before running.
- **Tie-bug fix alone (`inside_fix`)** — real default-code bug (pool<10 GTs lose positives to zero ties;
  670 tianchifabirc / 777 3cad GTs affected) but fixing it alone is a no-op on tianchifabirc (+0.63x).

## Laws and mechanism facts

- **3cad stability law**: any arm that removes the model's prediction from the topk ranking collapses
  (rfla ep9, geom_topk ep11, ar_rfla ep11, nwd beta<=2, A7 3/3). All arms that keep the ranking are stable
  (baseline, ms8-32, rfla_fill, inside_fix). No dataset-specific sliver treatment exists for 3cad.
- **Sliver mechanism (final, index-corrected)**: rfla reroutes slivers to P5 (11.5x637.5: inside pool
  160/40/0, rfla picks 0/0/10). Pool widening does NOT replicate this -- a converged model keeps picking
  P3 (64/30/6% P3/P4/P5) even when the pool is widened post hoc.
- **Bit-exact anchor**: coco8 3-epoch hash `512f46b7...` (drifted from 7086e13e due to environment, not code).

## In flight / queued

- **A8 `tal_prior=level_assign`** (Louis's rule: long side picks the level, pool = that level + finer
  neighbour, topk stays the model's): tianchifabirc (68 ep, healthy), 3cad (18 ep — the test of whether a
  hard LEVEL cut alone triggers the stability law), dspcbsd (5 ep, negative control).
- **yolo11 wave**: baseline + A7, 3 datasets x 3 seeds (architecture-generality check; ~2.1x wall time,
  long tail into 2026-08-24).

## Corrigendum ledger (superseded claims, newest first)

1. **"rfla routes slivers to coarse levels" is TRUE.** It was retracted in the A2.4 section based on a
   level-indexing bug in the measuring script; the index-corrected re-measure in the S1/S2 section
   re-establishes it (160/40/0 -> 0/0/10).
2. **"tie bug is half of A2's gain" is FALSE.** The A2.4 section attributed geom_topk's +2.69x to the bug
   fix; inside_fix (the fix alone) is +0.63x, so the +2.69x was the geometric ranking itself.
3. **"3cad_a25_arrfla4 alive at 62 ep" was a false negative** — the check counted rows, not mAP. It
   collapsed at ep11.
4. **dspcbsd AP_small x-floor readings overstate** — its AP_S floor is 0.0016; always read the p column.

---

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

---

# Phase B results — M1 fails, M3 is structurally invalid, and the reason reframes the whole plan

All six runs finished. Judged against the `dspcbsd` 2sd floors (AP_small 0.0016, mAP50-95 0.0067) and the
three-seed baseline means (mAP50-95 0.4765, AP_small 0.3985, AP_medium 0.5499).

| run                      | variable          | mAP50  | mAP50-95 | AP_S   | ΔAP_S (×floor)   |
| ------------------------ | ----------------- | ------ | -------- | ------ | ---------------- |
| baseline mean (3 seeds)  | --                | 0.7982 | 0.4765   | 0.3985 | --               |
| `dspcbsd_m1_ir07_n_s0`   | `inner_ratio=0.7` | 0.8059 | 0.4796   | 0.3915 | -0.0070 (-4.4×)  |
| `dspcbsd_m1_ir08_n_s0`   | `inner_ratio=0.8` | 0.7950 | 0.4701   | 0.3857 | -0.0128 (-8.0×)  |
| `dspcbsd_m1_ir12_n_s0`   | `inner_ratio=1.2` | 0.8001 | 0.4771   | 0.4003 | +0.0018 (+1.1×)  |
| `dspcbsd_m3_topk22_n_s0` | `o2o_topk2=2`     | 0.6226 | 0.3721   | 0.3185 | -0.0800 (-50.0×) |
| `dspcbsd_m3_topk23_n_s0` | `o2o_topk2=3`     | 0.5168 | 0.3184   | 0.2744 | -0.1241 (-77.6×) |
| `dspcbsd_m3_topk24_n_s0` | `o2o_topk2=4`     | 0.4742 | 0.2744   | 0.2415 | -0.1570 (-98.1×) |

**M1 (Inner-IoU): fail.** On mAP50-95 every setting is inside the floor (+0.5×, -1.0×, +0.1×) — a no-op. On
AP_small, shrinking the auxiliary box (`ratio < 1`) costs 4.4-8.0× the floor, and expanding it (`ratio = 1.2`)
lands inside the floor. Nothing here is worth carrying. The paper's mechanism is faster convergence on
high-IoU pairs; at 100 epochs these runs are converged, so there is no gap for it to close.

## M3 is not a bad hyperparameter, it is an invalid one — and the proof is cheap

`o2o_topk2` raises the positive count on the **one-to-one** head. That head is what runs at inference, with
no NMS. Training it to fire k times per object means k boxes per object reach the output. Precision reads as
0.7995 -> 0.5482 -> 0.4608 -> 0.4536 for topk2 = 1, 2, 3, 4 while recall barely moves
(0.7371 -> 0.6478 -> 0.6254 -> 0.5304): the classic duplicate signature, and `P ~= 0.52` at `topk2=2` is
suspiciously close to 1/2. The trajectory plateaus rather than diverging, so this is a ceiling, not instability.

Proved on the saved `predictions.json`, no GPU needed — post-hoc class-wise NMS at the same `iou=0.7`:

```bash
expman-cli pull yolo26-defect-bench yolo26-defect-bench__dspcbsd_m3_topk22_n_s0 --what all
expman-cli pull yolo26-defect-bench yolo26-defect-bench__dspcbsd_baseline_n_s0  --what all
P=/Users/louis/workspace/ultra_louis_work/expman/data/pulled/yolo26-defect-bench
python scripts/anomaly_bench/nms_probe.py $P/dspcbsd_baseline_n_s0 $P/dspcbsd_m3_topk22_n_s0
```

```
dspcbsd_baseline_n_s0  (23003 dets -> 14230 after NMS, 38.1% removed)
  raw    mAP50-95=0.4740  mAP50=0.7856  AP_S=0.3989  AP_M=0.5435  AP_L=0.3966
  +NMS   mAP50-95=0.4763  mAP50=0.8041  AP_S=0.4004  AP_M=0.5475  AP_L=0.4033

dspcbsd_m3_topk22_n_s0  (77867 dets -> 33508 after NMS, 57.0% removed)
  raw    mAP50-95=0.3736  mAP50=0.6224  AP_S=0.3189  AP_M=0.3899  AP_L=0.4190
  +NMS   mAP50-95=0.4416  mAP50=0.7786  AP_S=0.3733  AP_M=0.4662  AP_L=0.4634
```

`topk2=2` emits **3.4x** the boxes for the same 3167 objects. NMS recovers mAP50 0.6224 -> 0.7786, which is
0.156 of the 0.163 gap to the baseline's NMS'd 0.8041 — **96% of the loss was duplicates**. The residual
AP_small gap after NMS (0.3733 vs 0.4004) is the small real feature cost.

So `topk2=1` on the o2o head is not a tunable hyperparameter, it is the NMS-free contract. **M3 is closed
permanently**, and the design doc's premise — widen small-target positive coverage via `topk2` — was aimed at
the wrong head. The `o2o_topk2` knob stays in `default.yaml` because it is one line and documents the trap.

## The measurement that reframes the plan: anchor starvation

If positive coverage matters, the lever must sit on the **one-to-many** head (`tal_topk=10`), which is
discarded at inference. But `topk` only binds when more than 10 anchors are eligible at all —
`TaskAlignedAssigner.select_candidates_in_gts` keeps only anchors whose centre falls inside the GT box. So
measure the pool directly, from the `gt_val.json` that `coco_eval` already writes:

```bash
P=/Users/louis/workspace/ultra_louis_work/expman/data/pulled/yolo26-defect-bench
python scripts/anomaly_bench/anchor_pool.py $P/3cad_baseline_n_s0/gt_val.json    --imgsz 640 960 1280
python scripts/anomaly_bench/anchor_pool.py $P/dspcbsd_baseline_n_s0/gt_val.json --imgsz 640 960 1280
```

| dataset | levels    | imgsz | median pool | pool < topk(10) | pool <= 1 |
| ------- | --------- | ----- | ----------- | --------------- | --------- |
| 3cad    | P3-P5     | 640   | 7           | 55.0%           | 19.1%     |
| 3cad    | P3-P5     | 960   | 17          | 36.0%           | 9.4%      |
| 3cad    | P3-P5     | 1280  | 35          | 21.6%           | 3.7%      |
| 3cad    | **P2**-P5 | 640   | 35          | 21.6%           | 3.7%      |
| dspcbsd | P3-P5     | 640   | 55          | 6.7%            | 0.1%      |
| dspcbsd | P3-P5     | 960   | 134         | 0.3%            | 0.0%      |
| dspcbsd | **P2**-P5 | 640   | 254         | 0.1%            | 0.0%      |

This explains the entire results table so far:

- **On 3cad, 55% of objects are anchor-starved and 19% get at most one anchor.** No loss-side knob can
  reweight positives that do not exist, which is why `dfl`, `cls_pw`, Inner-IoU and `topk2` all came back
  neutral-to-harmful there — and why `imgsz=960` was the single biggest AP_small move on the whole branch
  (+0.0945, ~50% relative).
- **On dspcbsd the pool is already 55 with only 6.7% starved**, so `imgsz=960` bought almost nothing
  (+0.0072). Nothing to fix.
- Raising o2m `tal_topk` above 10 would therefore be a near-no-op on dspcbsd (pool 55 > 10 for 93% of boxes,
  assignment does change) but would still not reach the 55% of 3cad boxes that cannot supply 10 anchors.

**A P2 head at `imgsz=640` matches `imgsz=1280` on starvation (median 35, 21.6% starved) at far lower cost.**
That makes Z4 the highest-value untested candidate, and it moves the plan off loss knobs and onto
stride/resolution. Z4 is held by Louis.

Two probes kept, both CPU-only and both reusable: `scripts/anomaly_bench/nms_probe.py` and `scripts/anomaly_bench/anchor_pool.py`.

## Starvation does not explain `k=6`'s sign — aspect ratio does

Extending the pool measurement to the third dataset breaks the tidy version of the story above:

| dataset       | median pool @640 | starved (<10) | AR median | AR p90 | AR >= 5 | min side < 6px @640 | `k=6` ΔAP_S |
| ------------- | ---------------- | ------------- | --------- | ------ | ------- | ------------------- | ----------- |
| dspcbsd       | 55               | 6.7%          | 1.25      | 2.50   | 2.3%    | 0.0%                | +4.0×       |
| 3cad          | 7                | 55.0%         | 1.90      | 6.95   | 15.1%   | 8.7%                | +19.0×      |
| tianchifabirc | 6                | 55.5%         | 5.16      | 52.83  | 51.1%   | 31.1%               | -4.1×       |

`3cad` and `tianchifabirc` are **equally starved** (55.0% vs 55.5%, median pool 7 vs 6), yet `k=6` is the
branch's largest architectural win on one and a loss on the other. So starvation predicts _whether loss-side
knobs are reachable at all_, but it does **not** predict `k=6`'s sign.

Aspect ratio does. `tianchifabirc` defects are extreme slivers — median AR 5.16, p90 **52.8**, and **31.1% of
boxes have a side thinner than 6 px at `imgsz=640`, i.e. thinner than the `k=6` window itself**. A square 6x6
kernel averages across the thin dimension of a defect that is 2 px wide, which is exactly the signal that
matters. On `3cad` (8.7% thin) and `dspcbsd` (0.0% thin) there is almost nothing for it to smear.

Leading hypothesis, **not established**: `k=6` helps when the widened window still fits inside the defect and
hurts when it does not. The controlled test would be an anisotropic stem (`k=(6,3)` / `k=(3,6)`) on
`tianchifabirc`; a square-kernel sweep cannot separate "wider window" from "wider than the target".

Measured with the same probe:

```bash
P=/Users/louis/workspace/ultra_louis_work/expman/data/pulled/yolo26-defect-bench
for d in 3cad dspcbsd tianchifabirc; do
  python scripts/anomaly_bench/anchor_pool.py $P/${d}_baseline_n_s0/gt_val.json --imgsz 640
done
```

# The 3cad noise floor lands and invalidates every 3cad verdict above

`3cad_baseline_n_s1` and `_s2` completed. **3cad's own floor is an order of magnitude wider than
`dspcbsd`'s**, and the sections above judged 3cad deltas in `dspcbsd` floor units. Those readings are wrong
and are superseded here.

| metric   | s0     | s1     | s2     | mean   | sd     | **2sd** | dspcbsd 2sd | ratio |
| -------- | ------ | ------ | ------ | ------ | ------ | ------- | ----------- | ----- |
| mAP50    | 0.4826 | 0.4761 | 0.4884 | 0.4824 | 0.0062 | 0.0123  | 0.0210      | 0.6×  |
| mAP50-95 | 0.2900 | 0.2932 | 0.2893 | 0.2908 | 0.0021 | 0.0042  | 0.0067      | 0.6×  |
| AP_small | 0.1882 | 0.2032 | 0.1861 | 0.1925 | 0.0093 | 0.0187  | 0.0016      | 11.7× |
| AP_med   | 0.2699 | 0.2752 | 0.2573 | 0.2675 | 0.0092 | 0.0184  | 0.0110      | 1.7×  |

`3cad` AP*small carries sd/mean = **4.8%** relative noise, against 0.2% on `dspcbsd`. The lesson is that a
floor is per-dataset \_and* per-metric: `3cad` is actually **tighter** than `dspcbsd` on mAP50-95 (0.6×) and
wildly looser on AP_small. Borrowing a floor across datasets is not a conservative approximation, it is
arbitrary in both directions.

Re-judged against `3cad`'s own floors (baseline means AP_small 0.1925, mAP50-95 0.2908):

| change       | ΔAP_small | **×3cad floor** | previously reported | ΔmAP50-95 | ×3cad floor |
| ------------ | --------- | --------------- | ------------------- | --------- | ----------- |
| `imgsz=960`  | +0.0902   | **+4.82×**      | +59.1×              | +0.0224   | **+5.33×**  |
| `k=6`        | +0.0261   | **+1.40×**      | +19.0×              | +0.0142   | **+3.38×**  |
| `dfl=0`      | +0.0087   | **+0.47×**      | +8.1×               | +0.0030   | +0.71×      |
| `mosaic=0`   | -0.0169   | -0.90×          | -7.9×               | -0.0276   | -6.57×      |
| `cls_pw=0.5` | -0.0185   | -0.99×          | -8.9×               | -0.0189   | -4.50×      |
| `cls_pw=1.0` | -0.0418   | -2.24×          | -23.4×              | -0.0607   | -14.45×     |

Three corrections that change conclusions, not just numbers:

1. **`k=6` does not clear 2sd on 3cad AP_small (+1.40×).** The "+19×, largest architectural win on the
   branch" claim is withdrawn. What `k=6` does clear on 3cad is **mAP50-95, +3.38×** — a real effect, but not
   a small-object one, which is the opposite of the reason it was proposed.
2. **`dfl=0` is noise on 3cad (+0.47×).** The "positive on both hard datasets, negative on the easy one"
   pattern was an artifact of the borrowed floor. **Z1 is closed: no effect.**
3. **`mosaic=0` and `cls_pw=0.5` are noise on 3cad AP_small (~-1×)**, not the clear failures reported. Their
   real evidence is mAP50-95 (-6.57× and -4.50×), a different metric than the one that was cited.

Also now suspect: the aspect-ratio hypothesis rests on `k=6` reading -4.1× on `tianchifabirc` in **dspcbsd**
floor units. `tianchifabirc` has no seed replicates at all, so that -0.0065 may be noise and the hypothesis
may have no phenomenon to explain. The wave below measures it.

# Phase C — replicate the only surviving candidate, and buy the missing sigmas

Snap **`23be8da87374`**. Nine runs across GPUs 4-7, each card chained to refill itself.

**Comparability proven, not argued.** The `z3_k6_n_s0` runs predate the M1/M3 knobs (`20b6e886`,
`bca283c87e`, `d942b647fb`); the new seeds run at `23be8da87`. That diff touches `bbox_iou`, `BboxLoss` and
`E2ELoss`, so inertness at default args was verified by running coco8 (2 epochs, CPU, seed 0) under both
package trees and comparing `results.csv`: **all 14 columns bit-identical**, `time` excluded. Both branches
are gated off at defaults — `inner_ratio=1.0` skips the Inner-IoU block, and `tal_topk2` was already
hard-coded to 1.

```bash
EXP=/Users/louis/workspace/ultra_louis_work/expman/.venv/bin/expman-cli
D=/data/shared-datasets/louis_data/anomaly_bench
K=/home/louis/ultra_louis_work/yolo26n-k6.pt
P=yolo26-defect-bench

$EXP bundle

# Wave 1 — k=6 seed replicates. GPU 4/5 gated on the 3cad baselines so the cards never double up.
$EXP launch --snap --args "nohupyolo --after 3cad_baseline_n_s1 train data=$D/dspcbsd/data.yaml model=yolo26n-k6.yaml pretrained=$K epochs=100 imgsz=640 batch=128 seed=1 coco_eval=True device=4 project=$P name=dspcbsd_z3_k6_n_s1"
$EXP launch --snap --args "nohupyolo --after 3cad_baseline_n_s2 train data=$D/dspcbsd/data.yaml model=yolo26n-k6.yaml pretrained=$K epochs=100 imgsz=640 batch=128 seed=2 coco_eval=True device=5 project=$P name=dspcbsd_z3_k6_n_s2"
$EXP launch --snap --args "nohupyolo 0 train data=$D/3cad/data.yaml model=yolo26n-k6.yaml pretrained=$K epochs=100 imgsz=640 batch=128 seed=1 coco_eval=True device=6 project=$P name=3cad_z3_k6_n_s1"
$EXP launch --snap --args "nohupyolo 0 train data=$D/3cad/data.yaml model=yolo26n-k6.yaml pretrained=$K epochs=100 imgsz=640 batch=128 seed=2 coco_eval=True device=7 project=$P name=3cad_z3_k6_n_s2"

# Wave 2 — chained on the same card
$EXP launch --snap --args "nohupyolo --after dspcbsd_z3_k6_n_s1 train data=$D/3cad/data.yaml model=yolo26n-k6.yaml pretrained=$K epochs=100 imgsz=960 batch=128 seed=0 coco_eval=True device=4 project=$P name=3cad_z3xz7_k6_imgsz960_n_s0"
$EXP launch --snap --args "nohupyolo --after dspcbsd_z3_k6_n_s2 train data=$D/tianchifabirc/data.yaml model=yolo26n.pt epochs=100 imgsz=640 batch=128 seed=1 coco_eval=True device=5 project=$P name=tianchifabirc_baseline_n_s1"
$EXP launch --snap --args "nohupyolo --after tianchifabirc_baseline_n_s1 train data=$D/tianchifabirc/data.yaml model=yolo26n.pt epochs=100 imgsz=640 batch=128 seed=2 coco_eval=True device=5 project=$P name=tianchifabirc_baseline_n_s2"
$EXP launch --snap --args "nohupyolo --after 3cad_z3_k6_n_s1 train data=$D/tianchifabirc/data.yaml model=yolo26n-k6.yaml pretrained=$K epochs=100 imgsz=640 batch=128 seed=1 coco_eval=True device=6 project=$P name=tianchifabirc_z3_k6_n_s1"
$EXP launch --snap --args "nohupyolo --after 3cad_z3_k6_n_s2 train data=$D/tianchifabirc/data.yaml model=yolo26n-k6.yaml pretrained=$K epochs=100 imgsz=640 batch=128 seed=2 coco_eval=True device=7 project=$P name=tianchifabirc_z3_k6_n_s2"
```

| GPU | wave 1 (~h)              | wave 2 (~h)                                      | question the card answers                  |
| --- | ------------------------ | ------------------------------------------------ | ------------------------------------------ |
| 4   | `dspcbsd_z3_k6_n_s1` 1.8 | `3cad_z3xz7_k6_imgsz960_n_s0` 6.0                | is `k=6` additive with resolution?         |
| 5   | `dspcbsd_z3_k6_n_s2` 1.8 | `tianchifabirc_baseline_n_s1` -> `_s2` 1.3 + 1.3 | give tianchifabirc a sigma at last         |
| 6   | `3cad_z3_k6_n_s1` 2.8    | `tianchifabirc_z3_k6_n_s1` 1.6                   | is `k=6`'s only failure real? (3v3 with 5) |
| 7   | `3cad_z3_k6_n_s2` 2.8    | `tianchifabirc_z3_k6_n_s2` 1.6                   | same                                       |

**Two runs cancelled before they started.** `3cad_z1_dfl0_n_s1/_s2` were queued on GPUs 6/7 to resolve Z1;
the 3cad floor landed while they waited and put `dfl=0` at +0.47×, so they would only have confirmed a null.
`lsta --kill` does not match a `--after` waiter (its cmdline is the wrapper, not the `yolo` command), so the
PIDs came from `runs/yolo26-defect-bench/<name>.status`, were checked against `/proc/<pid>/cmdline` for the
run name before `kill -9`, and the `.status`/`.log` files were parked as `.cancelled`. Their expman records
stay at `queued` with the reason in `ai_notes` — there is no delete subcommand, and inventing one is not
worth it.

# Phase C results — k=6 replicated 3v3 on all three datasets

Eight of nine runs finished. With replicates on both arms the comparison finally becomes a two-sample test
instead of one run against a floor, so these Welch t-tests supersede every floor-multiple reading above.

| dataset       | metric   | k=6 (3 seeds)            | mean   | sd     | baseline mean | Δ           | t     | df   | p          | 95%?    |
| ------------- | -------- | ------------------------ | ------ | ------ | ------------- | ----------- | ----- | ---- | ---------- | ------- |
| dspcbsd       | mAP50-95 | 0.4812 / 0.4776 / 0.4802 | 0.4797 | 0.0019 | 0.4765        | **+0.0032** | +1.43 | 3.13 | 0.2434     | no      |
| dspcbsd       | AP_small | 0.4049 / 0.3997 / 0.4033 | 0.4026 | 0.0027 | 0.3985        | **+0.0041** | +2.58 | 2.35 | 0.1049     | no      |
| 3cad          | mAP50-95 | 0.3050 / 0.3004 / 0.3139 | 0.3064 | 0.0069 | 0.2908        | **+0.0156** | +3.77 | 2.36 | **0.0488** | **YES** |
| 3cad          | AP_small | 0.2186 / 0.2387 / 0.2607 | 0.2393 | 0.0211 | 0.1925        | **+0.0468** | +3.52 | 2.76 | **0.0444** | **YES** |
| tianchifabirc | mAP50-95 | 0.1972 / 0.1957 / 0.1899 | 0.1943 | 0.0039 | 0.1908        | **+0.0034** | +1.31 | 3.4  | 0.272      | no      |
| tianchifabirc | AP_small | 0.1190 / 0.1442 / 0.1241 | 0.1291 | 0.0133 | 0.1220        | **+0.0071** | +0.88 | 2.4  | 0.458      | no      |

> **Correction (was wrong when first written).** The two 3cad rows originally read "no, close". That came
> from comparing t against critical values for the wrong degrees of freedom — 2.776 is the df=4 value and
> 4.303 the df=2 value, while Welch gives fractional df of 2.36 and 2.76 here. Recomputed from the raw seed
> values with `scipy.stats.ttest_ind(..., equal_var=False)`, **both 3cad cells clear 95%** (p = 0.0488 and
> 0.0444). Read p directly; do not eyeball a t-table at fractional df.

**`k=6` is positive in all six dataset x metric cells, and clears 95% on both 3cad metrics.** The other four
cells do not clear at n=3. Six positives out of six is worth something on its own, but state it honestly: the
two metrics within a dataset are correlated, so the naive 1/64 = 1.6% is too generous; treating the three
datasets as independent gives 3/3 = **12.5%**, which on its own would not clear 95% either. The load-bearing
evidence is 3cad; the rest is consistent direction with underpowered tests.

Two corrections this forces, both to claims made earlier in this file:

1. **`k=6` does not hurt `tianchifabirc`.** The single-seed -0.0065 flipped sign to **+0.0071** with
   replicates and sits at t=0.88. **The aspect-ratio hypothesis is falsified** — there is no failure to
   explain, so `k=6 helps only when the window fits inside the defect` is withdrawn and the anisotropic-stem
   experiment is dropped. The AR/thin-side measurements stand as dataset facts; the causal story built on
   them does not.
2. **`k=6` on 3cad AP_small is larger than the single seed showed, not smaller.** 3-seed Δ = **+0.0468**
   against the s0-only +0.0261. The previous section judged it "does not clear" off one run; a single seed
   was unreliable in _both_ directions, which is the whole point of replicates.

**Cost of `k=6`: it makes training noisier.** Its seed sd exceeds the baseline's on every dataset —
AP_small 0.0211 vs 0.0093 on 3cad, 0.0133 vs 0.0042 on tianchifabirc, 0.0027 vs 0.0008 on dspcbsd. That
inflated variance is exactly why the t-tests fall short, and it is a real property of the change, not a
measurement artifact.

**Implication for the plan: buy seeds, not variants.** At 5v5 the same effect sizes would clear — dspcbsd
AP_small reaches t≈3.2 against a 2.45 threshold, 3cad AP_small t≈4.5 against 2.6. Adding variants instead
would generate more single-seed numbers of the kind this section just had to correct twice.

Still running: `3cad_z3xz7_k6_imgsz960_n_s0` (epoch 68/100, best so far AP_small 0.2795, mAP50-95 0.3016).
For reference `imgsz=960` alone finished at AP_small 0.2827 and `k=6` alone averages 0.2393, so the
combination is not yet ahead of resolution alone — but it is incomplete and must not be read yet.

## `k=6` vs `imgsz=960` — same direction, different price, and n=1 on one side

The two changes buy the same thing (more spatial detail reaching P3) by different means, so the honest
comparison needs the cost axis, not just AP. Params and FLOPs measured locally at batch 1 with
`torch.utils.flop_counter.FlopCounterMode`:

| config            | params            | GFLOPs   | vs baseline |
| ----------------- | ----------------- | -------- | ----------- |
| `yolo26n` @640    | 2,572,280         | 3.04     | 1.00×       |
| `yolo26n-k6` @640 | 2,696,696 (+4.8%) | **4.10** | **1.35×**   |
| `yolo26n` @960    | 2,572,280 (+0%)   | **7.01** | **2.31×**   |

`k=6` is a **weight** change (kernel 3→6 on layers 1 and 3, +4.8% params, same activations); `imgsz=960` is
an **activation** change (0 params, 2.25× the pixels through every layer). That is why `960` needed 47.8 GiB
at `batch=128` against ~25 GiB for the rest, and why it must have a card to itself.

Accuracy. Baseline is the first row of every block so the four arms are read on one scale — absolute
values, not just deltas. `sd` is across seeds; `n/a` means a single seed, which is the whole problem below.

**3cad** — the dataset where both changes actually do something

| arm         | n   | GFLOPs | mAP50-95   | sd     | Δ           | AP_small   | sd     | Δ           |
| ----------- | --- | ------ | ---------- | ------ | ----------- | ---------- | ------ | ----------- |
| baseline    | 3   | 3.04   | 0.2908     | 0.0021 | --          | 0.1925     | 0.0093 | --          |
| `k=6`       | 3   | 4.10   | 0.3064     | 0.0069 | +0.0156     | 0.2393     | 0.0211 | +0.0468     |
| `960`       | 1   | 7.01   | 0.3132     | n/a    | +0.0224     | **0.2827** | n/a    | **+0.0902** |
| `k=6`+`960` | 1   | 9.40   | **0.3254** | n/a    | **+0.0346** | 0.2470     | n/a    | +0.0545     |

**dspcbsd** — near saturation, everything is small. **`960` now has 3 seeds and its single-seed number was
the optimistic draw**, exactly the failure mode this section warned about.

| arm      | n   | GFLOPs | mAP50-95   | sd         | Δ       | AP_small   | sd         | Δ       |
| -------- | --- | ------ | ---------- | ---------- | ------- | ---------- | ---------- | ------- |
| baseline | 3   | 3.04   | 0.4765     | 0.0033     | --      | 0.3985     | 0.0008     | --      |
| `k=6`    | 3   | 4.10   | **0.4797** | 0.0019     | +0.0032 | 0.4026     | 0.0027     | +0.0041 |
| `960`    | 3   | 7.01   | 0.4788     | **0.0056** | +0.0023 | **0.4049** | **0.0115** | +0.0064 |

`960` seeds: mAP50-95 0.4831 / 0.4807 / 0.4725, AP_small 0.4057 / 0.4159 / 0.3930. The s0 draw used earlier
(0.4831 / 0.4057) sat at the top of both ranges, and with the other two seeds in, **`960`'s mAP50-95 mean
falls below `k=6`'s.** Welch, all three arms at n=3:

| comparison        | ΔmAP50-95 | p      | ΔAP_small | p      |
| ----------------- | --------- | ------ | --------- | ------ |
| `k=6` vs baseline | +0.0032   | 0.2434 | +0.0041   | 0.1049 |
| `960` vs baseline | +0.0023   | 0.5861 | +0.0064   | 0.4381 |
| `960` vs `k=6`    | -0.0009   | 0.8082 | +0.0022   | 0.7720 |

**Nothing clears on dspcbsd, and `960` is statistically indistinguishable from `k=6` there (p = 0.77-0.81).**
The reason is variance, not the mean: `960`'s AP_small seed sd is **0.0115, fourteen times the baseline's
0.0008** and four times `k=6`'s. Both interventions inflate run-to-run spread on this dataset, and `960`
inflates it far more.

**tianchifabirc** — `960` still queued, no combination run

| arm      | n   | GFLOPs | mAP50-95   | sd     | Δ       | AP_small   | sd     | Δ       |
| -------- | --- | ------ | ---------- | ------ | ------- | ---------- | ------ | ------- |
| baseline | 3   | 3.04   | 0.1908     | 0.0024 | --      | 0.1220     | 0.0042 | --      |
| `k=6`    | 3   | 4.10   | **0.1943** | 0.0039 | +0.0034 | **0.1291** | 0.0133 | +0.0071 |

Reading across the three blocks: the arms rank **baseline < `k=6` < `960`** on every metric of every dataset
measured so far, and the spread between them tracks how much headroom the dataset has — large on 3cad,
negligible on dspcbsd, unmeasurable on tianchifabirc. `k=6`+`960` is the only arm that breaks the ranking,
and it breaks it on AP_small only.

**Gain per extra GFLOP** — `k=6` costs +1.06 GFLOPs, `960` costs +3.97:

| cell             | `k=6` /GFLOP | `960` /GFLOP | `k=6` advantage |
| ---------------- | ------------ | ------------ | --------------- |
| 3cad AP_small    | 0.0442       | 0.0227       | 1.94×           |
| 3cad mAP50-95    | 0.0147       | 0.0056       | 2.61×           |
| dspcbsd AP_small | 0.0039       | 0.0018       | 2.13×           |
| dspcbsd mAP50-95 | 0.0030       | 0.0017       | 1.82×           |

The pattern is consistent across all four cells: **`960` wins on absolute AP, `k=6` wins ~2× on AP per FLOP.**
`960` buys roughly twice the gain for roughly four times the extra compute.

**Is this enough to conclude? No — every `960` number above is a single seed.** This section must not repeat
the mistake the section above had to correct twice. Concretely, on 3cad AP*small the gap between `960`
(0.2827) and the `k=6` 3-seed mean (0.2393) is 0.0434, which is **2.06× `k=6`'s own seed sd of 0.0211**. A
single draw from a distribution that wide cannot be compared to a 3-seed mean, and `960`'s own sd is unknown.
The \_ordering* is probably safe; the _magnitude_, and therefore the whole efficiency ratio, is not pinned.

**The combination is the sharpest warning.** `k=6`+`960` beats both on mAP50-95 (0.3254, close to the
additive prediction 0.2908+0.0156+0.0224 = 0.3288) but on AP_small lands at 0.2470 — **below `960` alone**.
Either the two are anti-additive on small objects, or one of these two n=1 points is a bad draw. With the
numbers in hand those are indistinguishable.

### Supplementary runs launched

Snap **`1a0ec3ba3ee8`**. Six runs; `960` needs ~48 GiB so one per card.

```bash
EXP=/Users/louis/workspace/ultra_louis_work/expman/.venv/bin/expman-cli
D=/data/shared-datasets/louis_data/anomaly_bench
P=yolo26-defect-bench

$EXP bundle
$EXP launch --snap --args "nohupyolo 0 train data=$D/3cad/data.yaml model=yolo26n.pt epochs=100 imgsz=960 batch=128 seed=1 coco_eval=True device=4 project=$P name=3cad_z7_imgsz960_n_s1"
$EXP launch --snap --args "nohupyolo 0 train data=$D/3cad/data.yaml model=yolo26n.pt epochs=100 imgsz=960 batch=128 seed=2 coco_eval=True device=5 project=$P name=3cad_z7_imgsz960_n_s2"
$EXP launch --snap --args "nohupyolo 0 train data=$D/dspcbsd/data.yaml model=yolo26n.pt epochs=100 imgsz=960 batch=128 seed=1 coco_eval=True device=6 project=$P name=dspcbsd_z7_imgsz960_n_s1"
$EXP launch --snap --args "nohupyolo 0 train data=$D/dspcbsd/data.yaml model=yolo26n.pt epochs=100 imgsz=960 batch=128 seed=2 coco_eval=True device=7 project=$P name=dspcbsd_z7_imgsz960_n_s2"
$EXP launch --snap --args "nohupyolo --after dspcbsd_z7_imgsz960_n_s1 train data=$D/3cad/data.yaml model=yolo26n-k6.yaml pretrained=/home/louis/ultra_louis_work/yolo26n-k6.pt epochs=100 imgsz=960 batch=128 seed=1 coco_eval=True device=6 project=$P name=3cad_z3xz7_k6_imgsz960_n_s1"
$EXP launch --snap --args "nohupyolo --after dspcbsd_z7_imgsz960_n_s2 train data=$D/tianchifabirc/data.yaml model=yolo26n.pt epochs=100 imgsz=960 batch=128 seed=0 coco_eval=True device=7 project=$P name=tianchifabirc_z7_imgsz960_n_s0"
```

| GPU | now (~h)                       | chained (~h)                         | what it buys                                  |
| --- | ------------------------------ | ------------------------------------ | --------------------------------------------- |
| 4   | `3cad_z7_imgsz960_n_s1` 4.2    | --                                   | 3cad `960` reaches n=3 -> real 2-sample test  |
| 5   | `3cad_z7_imgsz960_n_s2` 4.2    | --                                   | same                                          |
| 6   | `dspcbsd_z7_imgsz960_n_s1` 1.7 | `3cad_z3xz7_k6_imgsz960_n_s1` 4.5    | dspcbsd `960` to n=3; start seeding the combo |
| 7   | `dspcbsd_z7_imgsz960_n_s2` 1.7 | `tianchifabirc_z7_imgsz960_n_s0` 2.5 | fills the last hole in the 3x3 grid           |

After this, `baseline` / `k=6` / `960` are all n=3 on both dspcbsd and 3cad, so "`k=6` vs `960`" becomes a
Welch test between two replicated arms instead of a mean against a point. Code comparability across the
M1/M3 commit boundary is the same coco8 bit-identical A/B proven for Phase C.

## expman's `status` field is not evidence — check the process and the CSV

A status audit during this wave found **three of the six supplementary runs mislabelled at once**:

| run                              | expman said | actually                                      |
| -------------------------------- | ----------- | --------------------------------------------- |
| `3cad_z3xz7_k6_imgsz960_n_s1`    | failed      | **running** — PID 3530124 on GPU 6, log live  |
| `tianchifabirc_z7_imgsz960_n_s0` | failed      | **running** — PID 3531054 on GPU 7, log live  |
| `dspcbsd_z7_imgsz960_n_s2`       | running     | **completed** — 100 epochs, best.pt + last.pt |
| `k6_donor`                       | failed      | completed (known: writes no training marker)  |

Both "failed" runs were launched with `--after`, so while the waiter was queued there was no process
carrying the run name for expman to match, and it recorded a failure. The status never corrected once they
started. This is the same class of bug as `3cad_z3_k6_n_s2` above, where a healthy run parsed to empty
`train_metrics`.

**Rule: never conclude a run failed from expman alone.** The two cheap checks that settle it:

```bash
R=/home/louis/ultra_louis_work/ultralytics/runs/yolo26-defect-bench
# 1. is a process actually carrying this run name?
ssh ultra6 "nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader"
ssh ultra6 'for p in <pids>; do tr "\0" " " < /proc/$p/cmdline | grep -oE "name=[^ ]+"; done'
# 2. did it finish? 101 rows = header + 100 epochs, and both weights present
ssh ultra6 "wc -l < $R/<name>/results.csv; ls $R/<name>/weights/"
```

A genuinely dead run shows a log that stops mid-epoch with no traceback. Here both suspects had logs being
written to the same second the audit ran, which is what exposed the mislabel.

## Two operational notes from this wave

**`3cad_z3_k6_n_s2` trained fine but expman parsed no metrics from it.** The run is healthy: 100 epochs,
well-formed `results.csv` (25 columns x 100 rows, no empty cells), final mAP50 0.5266. `import-run` still
returns empty `train_metrics`, so its numbers above were read directly from `results.csv` using the stock
detect fitness (`0.1*mAP50 + 0.9*mAP50-95`), best row = epoch 99. Do not trust an empty `train_metrics` as
evidence a run failed.

**`coco_eval` breaks `results.png` on runs with 8 optimizer param groups.**
`ERROR ❌ Plotting error for results.csv: index 14 is out of bounds for axis 0 with size 14` — the plotter
sizes its axis grid before the five extra `(B-coco)` columns exist. Cosmetic only: metrics, CSV and
checkpoints are unaffected. Note that `optimizer=auto` picks 8 param groups on 3cad and 3 on dspcbsd, so
column counts differ **between** datasets (25 vs 20) but are consistent **within** one — and every
comparison on this branch is within a dataset.

# Scale bands: what AP_small actually measures, and pinning it to letterbox-640

## The images are rescaled before the model sees them, upward as well as downward

`LetterBox(scaleup=False)` in the val transform (`dataset.py:318`) suggests small images are never enlarged.
That reading is wrong. The resize happens earlier, in `BaseDataset.load_image` (`base.py:258`):

```python
r = self.imgsz / max(h0, w0)   # rect_mode
if r != 1:                     # this branch enlarges as well as shrinks
    im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
```

By the time `LetterBox` runs the image already matches `imgsz`, so `scaleup=False` is a no-op. Every dataset
on this branch is genuinely rescaled at both 640 and 960:

| dataset       | image size | @640                  | @960                   |
| ------------- | ---------- | --------------------- | ---------------------- |
| dspcbsd       | 226x226    | r=2.83, area **x8.0** | r=4.25, area **x18.0** |
| 3cad          | 1024x1024  | r=0.625, area x0.39   | r=0.938, area x0.88    |
| tianchifabirc | 640x640    | r=1.0, area x1.0      | r=1.5, area x2.25      |

dspcbsd is 1621 images at 226x226 plus 12 at 108x108; tianchifabirc is 1185 images all exactly 640x640;
3cad is mostly 1024x1024 across 126 distinct sizes.

## Band split in each frame

`yolo2coco_gt` writes `area` from the **original** image dimensions (`converter.py:399-409`), so the reported
AP_small/medium/large use original pixels against COCO's fixed 32^2 = 1024 and 96^2 = 9216 thresholds.

| dataset       | frame         | small     | medium | large | area p50 | p10  | p90   |
| ------------- | ------------- | --------- | ------ | ----- | -------- | ---- | ----- |
| dspcbsd       | original      | 67.2%     | 28.9%  | 3.9%  | 400      | 96   | 2900  |
| dspcbsd       | letterbox 640 | **16.2%** | 52.0%  | 31.8% | 3268     | 769  | 24243 |
| dspcbsd       | letterbox 960 | 2.5%      | 52.4%  | 45.1% | 7354     | 1731 | 54547 |
| 3cad          | original      | 42.6%     | 42.4%  | 14.9% | 1305     | 210  | 17112 |
| 3cad          | letterbox 640 | **63.7%** | 27.8%  | 8.5%  | 536      | 86   | 6984  |
| 3cad          | letterbox 960 | 45.8%     | 40.4%  | 13.8% | 1206     | 194  | 15715 |
| tianchifabirc | original      | 60.3%     | 23.8%  | 15.9% | 401      | 97   | 26802 |
| tianchifabirc | letterbox 640 | **60.3%** | 23.8%  | 15.9% | 401      | 97   | 26802 |
| tianchifabirc | letterbox 960 | 51.7%     | 20.6%  | 27.6% | 903      | 219  | 60305 |

Four consequences:

1. **dspcbsd is not a small-object dataset.** In original pixels 67.2% of its boxes are "small", but the
   model sees them enlarged 8x in area — only **16.2% are small and 31.8% are already large**. The reason
   `imgsz=960` bought nothing there (Δ mAP50-95 +0.0023, p=0.59) is not that resolution fails to help; it is
   that there was no small-object problem to fix.
2. **3cad's small-object problem is manufactured by the letterbox.** It is the least small dataset in
   original pixels (42.6%) and the most small at 640 (**63.7%**), because its 1024x1024 images are shrunk to
   0.39x area. `imgsz=960` restores it to 45.8%. So `imgsz=960` on 3cad does not add information — **it
   throws less away**, which is the mechanism behind its +0.0902 AP_small, the largest single move on this
   branch.
3. **An AP_small gain across `imgsz` does not mean the model got better at small objects.** The objects
   stopped being small. The grouping is fixed, so the comparison is valid; the _interpretation_ "better at
   small objects" is not.
4. **Two different axes have been conflated.** COCO bands are absolute pixels; anchor starvation is relative
   to image size. dspcbsd is absolutely-small but relatively-large (pool 55); 3cad is absolutely-medium but
   relatively-tiny (pool 7). The starvation table and the AP_small column are not measuring the same thing
   and must not be chained into one argument.

## Decision: keep original pixels for now; letterbox-640 is understood but NOT adopted

**Louis's call: no change. Every AP_small/medium/large on this branch stays in the original-pixel
convention, and the caveat below is recorded instead.** Switching mid-programme would split the tables
across two conventions and cost a val-only pass over every `best.pt` (see Migration). The analysis below is
kept because it is what makes the existing numbers readable, not because anything was changed.

### Caveat to carry with every dspcbsd AP_small number

**dspcbsd images are 226x226** (1621 of 1633; the other 12 are 108x108). At `imgsz=640` `load_image`
enlarges them 2.83x linear = **8.0x in area**, at 960 it is 4.25x / **18.0x**. So a dspcbsd box that COCO
files as "small" by original pixels is presented to the network far larger than its nominal size — in the
frame the model actually sees, only **16.2%** of dspcbsd boxes are small and **31.8% are already large**.

Two things follow, and both matter for how the results here are read:

1. **`dspcbsd AP_small` is not a small-object measurement.** It is a label attached to a set of boxes the
   model mostly resolves as medium or large. Do not cite it as evidence about small-object behaviour, and do
   not pool it with 3cad's AP_small as if the two measured the same regime.
2. **Enlarging adds no information.** A 226px image resampled to 640 still carries 226px of real detail;
   everything above native resolution is interpolation. So on dspcbsd `imgsz` is already saturated at the
   baseline and 960 is interpolating an interpolation — which is the mechanism behind its null result
   (Δ mAP50-95 +0.0023, p=0.59), not a failure of resolution as a lever.

Generalising that to the whole benchmark: the useful ceiling for `imgsz` is each dataset's native size —
226 for dspcbsd, 640 for tianchifabirc, 1024 for 3cad. **3cad is the only dataset whose native resolution
exceeds the 640 baseline**, so it is the only one where raising `imgsz` recovers real detail rather than
interpolating, and even there 960 is still below native. This predicts that `imgsz=1280` would add nothing
anywhere, and that 3cad's remaining headroom from resolution alone stops at 1024.

### The options, for when this is revisited

Recorded for later; **none of these is in force today**.

| option                     | verdict                                                                                                 |
| -------------------------- | ------------------------------------------------------------------------------------------------------- |
| **original pixels**        | **in force** — dataset-intrinsic; describes difficulty wrongly, which the caveat above covers instead   |
| letterbox-640, pinned      | the better definition, understood and validated, **not adopted** — see Migration for what it would cost |
| per-run `imgsz` frame      | **wrong**: the grouping would move with the arm, so "AP_small up" could just mean fewer boxes are small |
| relative area (% of image) | resolution-proof, but abandons the COCO convention and comparability with published numbers             |

The load-bearing property is that the reference is **pinned at 640 and does not follow `args.imgsz`** — that
is what keeps a 640 arm and a 960 arm comparable. Pinning also improves the statistics: under original
pixels dspcbsd had only 124 large boxes (3.9%), which is why its AP_large floor was an unusable 0.2034; at
letterbox-640 it has 513 / 1647 / 1007 boxes across the three bands and all three become usable.

## How it would be implemented, already verified so the work is not lost

Not applied. Recorded so that adopting it later is a code change, not a re-investigation.

Rescale **both** the GT boxes and the predictions by `640 / max(w, h)` per image. IoU is invariant under a
similarity transform applied to both sides, so AP and AP50 are untouched and only the band assignment moves.
Verified on `dspcbsd_baseline_n_s0` and `3cad_baseline_n_s0` — AP and AP50 come back **bit-identical**:

```bash
P=/Users/louis/workspace/ultra_louis_work/expman/data/pulled/yolo26-defect-bench
python scripts/anomaly_bench/band_probe.py $P/dspcbsd_baseline_n_s0 640
```

```
dspcbsd_baseline_n_s0  original frame: AP=0.473997 AP50=0.785642 S=0.398921 M=0.543504 L=0.396632
dspcbsd_baseline_n_s0  letterbox-640:  AP=0.473997 AP50=0.785642 S=0.290170 M=0.426146 L=0.562433
dspcbsd_baseline_n_s0  GT-area-only:   AP=0.473997 AP50=0.785642 S=0.112047 M=0.467612 L=0.682512
3cad_baseline_n_s0     original frame: AP=0.291428 AP50=0.484480 S=0.187804 M=0.268588 L=0.417248
3cad_baseline_n_s0     letterbox-640:  AP=0.291428 AP50=0.484480 S=0.166641 M=0.303539 L=0.396728
```

**The shortcut of rescaling only the GT `area` field is wrong** and the third line shows how wrong
(S = 0.1120 against the correct 0.2902). COCO ignores an _unmatched_ detection whose own area falls outside
the band, so false positives would be banded in the original frame while ground truth was banded in the
letterbox frame. Both sides must be transformed.

## Migration — deliberately not done

**Every AP_small/medium/large number on this branch is in the original-pixel convention, and stays that
way.** The two constraints that make switching expensive, and that decided it:

- Runs already in flight are pinned by `--snap` to older commits, so a code change now cannot corrupt them.
  The convention boundary would be the launch commit, and must be recorded per run.
- `predictions.json` is overwritten every epoch, so an offline recompute yields **last-epoch** bands, while
  every table here reports the **best** epoch. Restating the tables faithfully needs a val-only pass over
  each `best.pt`, not a rerun of training.

If it is ever adopted, the pass is: re-validate `best.pt` for the arms that matter (baseline / `k=6` / `960`
/ `k6+960` across the three datasets) with both sides rescaled — twelve val-only jobs, no retraining. Until
then the caveat above is the mitigation, and `dspcbsd AP_small` must not be read as a small-object result.

# WAVE A — label assignment (bottleneck b)

Three knobs, all training-only: zero params, zero inference FLOPs, zero export surface. Code in
`74d9df50f`; `E2ELoss` builds one `v8DetectionLoss` per head, so every knob reaches the one2many and the
one2one assigner together. Defaults are bit-exact with the pre-change code (3 epochs of deterministic
coco8 training give an identical `state_dict` hash), so **the existing baselines stay valid comparators
and no baseline was re-run.**

| knob | arg                               | what it changes                                                                                        |
| ---- | --------------------------------- | ------------------------------------------------------------------------------------------------------ |
| A1   | `tal_min_side`                    | monotone floor on GT sides, replacing the legacy clamp in `select_candidates_in_gts`                   |
| A2   | `tal_prior=rfla`, `tal_rf_scale`  | RFLA (arXiv:2208.08738): rank anchors by receptive-field distance, `topk` candidates per GT guaranteed |
| A3   | `tal_metric=nwd`, `tal_nwd_gamma` | scale-invariant Wasserstein similarity in place of CIoU inside the assigner                            |

## The legacy clamp is not monotone, and the old pool numbers were measured against the wrong clamp

`tal.py:300` inflates a GT side **below `stride[0]`=8 straight to `stride_val`=`stride[1]`=16**, and leaves
`[8, 16)` untouched. So a side just above 8 recruits _fewer_ anchors than one just below it. Measured on a
synthetic grid with the centre averaged over 16 sub-pixel offsets to cancel alignment noise, `topk=10`:

| GT side px    | 3    | 5    | 7.9  | 8.1     | 10   | 12   | 15.9 | 16.1 |
| ------------- | ---- | ---- | ---- | ------- | ---- | ---- | ---- | ---- |
| legacy        | 4.6  | 4.6  | 4.6  | **2.7** | 2.7  | 2.7  | 4.6  | 7.2  |
| `min_side=8`  | 1.3  | 1.3  | 1.3  | 2.7     | 2.7  | 2.7  | 4.6  | 7.2  |
| `min_side=16` | 4.6  | 4.6  | 4.6  | 4.6     | 4.6  | 4.6  | 4.6  | 7.2  |
| `min_side=24` | 10.6 | 10.6 | 10.6 | 10.6    | 10.6 | 10.6 | 10.6 | 10.6 |
| `min_side=32` | 18.8 | 18.8 | 18.8 | 18.8    | 18.8 | 18.8 | 18.8 | 18.8 |

`anchor_pool.py` had inflated to 8 rather than 16 (fixed in `46156e462`), which both mis-stated the pool
and erased this band. Re-measured on the real GT, `imgsz=640`, `topk=10`:

```bash
B=/Users/louis/workspace/ultra_louis_work/expman/data/pulled/yolo26-defect-bench
for d in 3cad tianchifabirc dspcbsd; do python scripts/anomaly_bench/anchor_pool.py $B/${d}_baseline_n_s0/gt_val.json --imgsz 640 --topk 10 --min-side 8 16 24 32; done
```

| dataset         | short side in [8,16) | starved (legacy) | pool<=1 (legacy) | **pool==0** | starved as previously reported |
| --------------- | -------------------- | ---------------- | ---------------- | ----------- | ------------------------------ |
| `3cad`          | **34.3%**            | **52.1%**        | **8.8%**         | **0.0%**    | 55.0% / 19.1%                  |
| `tianchifabirc` | **30.8%**            | **40.6%**        | **5.4%**         | **0.0%**    | 55.5% / 11.5%                  |
| `dspcbsd`       | 2.0%                 | 6.7%             | 0.1%             | **0.0%**    | 6.7% / 0.1%                    |

The previously-reported figures are reproduced exactly by the `min_side=8` row of the same script, which
confirms the old clamp was the only error. Three consequences:

1. **`pool == 0` is 0.0% everywhere, so no GT is ever left without a positive.** Since the one-to-one head
   runs `topk2=1` and therefore needs only a non-empty pool, **this whole wave is effectively a one2many
   experiment** even though the knobs are wired to both heads. Conclusions must say so.
2. **`tianchifabirc` is materially _less_ starved than `3cad`** (40.6% vs 52.1%), not equally starved.
3. **`dspcbsd` is a free negative control for A1**: only 2.0% of its boxes sit in the affected band, so
   `min_side=16` should not move it. If it does, the mechanism is not the one we think.

Dose grid per dataset (starved %, `topk=10`), which is what selected the arms:

| min_side | 3cad   | tianchifabirc | dspcbsd | character                                                                  |
| -------- | ------ | ------------- | ------- | -------------------------------------------------------------------------- |
| legacy   | 52.1%  | 40.6%         | 6.7%    | baseline                                                                   |
| 8        | 55.0%  | 55.5%         | 6.7%    | **reverse dose** — stricter than legacy; isolates monotonicity from dose   |
| 16       | 45.1%  | 34.2%         | 6.3%    | closes the discontinuity; pool<=1 -> 0%                                    |
| 24       | **0%** | **0%**        | **0%**  | starvation eliminated; median pool 14 just fills `topk=10`                 |
| 32       | 0%     | 0%            | 0%      | **over-dose** — buys no coverage, only competition; the precision-cost arm |

`min_side` 16/18/20/22 give an **identical** pool, because the pool counts `int(w // stride)` and so only
steps at multiples of 8. The grid {8, 16, 24, 32} is therefore complete and a finer sweep is meaningless.

## A1/A2/A3 — screening wave, 3cad only, 1 seed

Snap **`e391163211bf`**. Nine runs on GPUs 4-7, **co-located two-per-card with the running coco arms**
(56.9/97.9 GiB per card). Hard gate: `3cad_baseline_n_s0/args.yaml` differs from `default.yaml` in
`batch=128` and `coco_eval=True` only, and `$B` below reproduces both; nothing else is overridden.

```bash
EXP=/Users/louis/workspace/ultra_louis_work/expman/.venv/bin/expman-cli
D=/data/shared-datasets/louis_data/anomaly_bench
B="model=yolo26n.pt epochs=100 imgsz=640 batch=128 seed=0 coco_eval=True project=yolo26-defect-bench"

$EXP bundle
$EXP launch --snap --args "nohupyolo 0 train data=$D/3cad/data.yaml $B tal_min_side=16 device=4 name=3cad_a1_ms16_n_s0"
$EXP launch --snap --args "nohupyolo 0 train data=$D/3cad/data.yaml $B tal_min_side=24 device=5 name=3cad_a1_ms24_n_s0"
$EXP launch --snap --args "nohupyolo 0 train data=$D/3cad/data.yaml $B tal_prior=rfla tal_rf_scale=1 device=6 name=3cad_a2_rfla1_n_s0"
$EXP launch --snap --args "nohupyolo 0 train data=$D/3cad/data.yaml $B tal_metric=nwd device=7 name=3cad_a3_nwd10_n_s0"
$EXP launch --snap --args "nohupyolo --after 3cad_a1_ms16_n_s0 train data=$D/3cad/data.yaml $B tal_min_side=8 device=4 name=3cad_a1_ms08_n_s0"
$EXP launch --snap --args "nohupyolo --after 3cad_a1_ms24_n_s0 train data=$D/3cad/data.yaml $B tal_min_side=32 device=5 name=3cad_a1_ms32_n_s0"
$EXP launch --snap --args "nohupyolo --after 3cad_a2_rfla1_n_s0 train data=$D/3cad/data.yaml $B tal_prior=rfla tal_rf_scale=4 device=6 name=3cad_a2_rfla4_n_s0"
$EXP launch --snap --args "nohupyolo --after 3cad_a3_nwd10_n_s0 train data=$D/3cad/data.yaml $B tal_metric=nwd tal_nwd_gamma=2.0 device=7 name=3cad_a3_nwd20_n_s0"
$EXP launch --snap --args "nohupyolo --after 3cad_a1_ms08_n_s0 train data=$D/3cad/data.yaml $B tal_metric=nwd tal_nwd_gamma=0.5 device=4 name=3cad_a3_nwd05_n_s0"
```

| GPU | now                  | then                 | then                 |
| --- | -------------------- | -------------------- | -------------------- |
| 4   | `3cad_a1_ms16_n_s0`  | `3cad_a1_ms08_n_s0`  | `3cad_a3_nwd05_n_s0` |
| 5   | `3cad_a1_ms24_n_s0`  | `3cad_a1_ms32_n_s0`  | --                   |
| 6   | `3cad_a2_rfla1_n_s0` | `3cad_a2_rfla4_n_s0` | --                   |
| 7   | `3cad_a3_nwd10_n_s0` | `3cad_a3_nwd20_n_s0` | --                   |

**One seed here is screening, not evidence.** No conclusion may be drawn from this wave; it only picks
which arms get 3 seeds. Promotion bar is deliberately low -- 3cad AP_small delta above **1x** the floor
(0.0187), not 2x -- because at n=1 a single draw carries +-0.019 and the risk is a false negative, not a
false positive. A1 is read as a dose curve, whose shape is more robust to seed noise than any single point.

**The `time` column of these nine runs is void.** Co-location costs ~8% (100 s/epoch here vs the clean
baseline's 92.8 s/epoch), so wall-clock is still usable as a rough guide, but the clean per-run timings for
future budgeting are the baselines: 3cad 2.58 h, tianchifabirc 1.27 h, dspcbsd 1.26 h.

First-epoch losses already separate, which is the cheapest confirmation the knobs bite:

| run                  | box_loss | cls_loss  | l1_loss  |
| -------------------- | -------- | --------- | -------- |
| `3cad_a1_ms24_n_s0`  | 2.114    | 15.22     | 0.006761 |
| `3cad_a2_rfla1_n_s0` | 2.253    | 15.71     | 0.007244 |
| `3cad_a3_nwd10_n_s0` | 2.545    | **13.44** | 0.007268 |

`nwd`'s lower `cls_loss` is expected rather than encouraging: `loss_cls` is a BCE sum divided by
`target_scores.sum()`, and lifting the soft-label ceiling enlarges the denominator. It confirms A3 is
active; it says nothing about whether it helps.

# Phase D — COCO A/B: does `k=6` generalise off the defect benchmark?

Snap **`23be8da87374`** · `yolo26n` · `epochs=100` · `imgsz=640` · `batch=128` · `coco_eval=True` ·
launched 2026-08-21. COCO is the public-data question: every `k=6` positive so far is on the three defect
datasets, and none of those is a natural-image distribution. Data is `/data/shared-datasets/coco`
(118,287 train / 5,000 val / 80 classes), reached through a `datasets/coco` symlink created on ultra6
because the shared dir sits outside `datasets_dir`.

Scope is **1 baseline + 3 `k=6` seeds** (Louis's call). The `k=6` donor carries the full 80-class head
here, so `Transferred 708/708` means exactly the same bit-identical start as on the defect datasets.

```bash
EXP=/Users/louis/workspace/ultra_louis_work/expman/.venv/bin/expman-cli
K=/home/louis/ultra_louis_work/yolo26n-k6.pt
P=yolo26-defect-bench

$EXP launch --snap --args "nohupyolo 0 train data=coco.yaml model=yolo26n.pt epochs=100 imgsz=640 batch=128 seed=0 coco_eval=True device=7 project=$P name=coco_baseline_n_s0"
$EXP launch --snap --args "nohupyolo 0 train data=coco.yaml model=yolo26n-k6.yaml pretrained=$K epochs=100 imgsz=640 batch=128 seed=0 coco_eval=True device=4 project=$P name=coco_z3_k6_n_s0"
$EXP launch --snap --args "nohupyolo 0 train data=coco.yaml model=yolo26n-k6.yaml pretrained=$K epochs=100 imgsz=640 batch=128 seed=1 coco_eval=True device=5 project=$P name=coco_z3_k6_n_s1"
$EXP launch --snap --args "nohupyolo --after 3cad_z3xz7_k6_imgsz960_n_s1 train data=coco.yaml model=yolo26n-k6.yaml pretrained=$K epochs=100 imgsz=640 batch=128 seed=2 coco_eval=True device=6 project=$P name=coco_z3_k6_n_s2"
```

| GPU | run                  | note                                                     |
| --- | -------------------- | -------------------------------------------------------- |
| 7   | `coco_baseline_n_s0` | launched first so it finishes first                      |
| 4   | `coco_z3_k6_n_s0`    |                                                          |
| 5   | `coco_z3_k6_n_s1`    |                                                          |
| 6   | `coco_z3_k6_n_s2`    | chained after `3cad_z3xz7_k6_imgsz960_n_s1`, ~1.7 h wait |

**Readout differs from the defect datasets.** On COCO `is_coco=True`, so `coco_generic` is false and the
COCO eval feeds the **native** `(B)` columns every epoch — the results.csv header carries
`metrics/mAP{50,50-95,small,medium,large}(B)` and no `(B-coco)` columns. Those `(B)` values are the
official COCOeval numbers, the standard reporting convention, and `fitness` keeps the same formula the
stock COCO path uses, so `best.pt` semantics are unchanged and both arms are treated identically.

**Speed, corrected.** ~50 epochs in ~30 min ⇒ **~40-50 s/epoch, a full run is ~1-1.5 h**, not the 20-25 h
assumed when the seed count was decided (that estimate borrowed co-tenant defect-dataset speeds; these
cards are solo and the n model sustains ~20 it/s at batch 128 on Blackwell). Seeds are ~15x cheaper than
planned, so the 1v3 shape is only worth keeping if it reads as clearly null or clearly positive;
otherwise the obvious completion is two more baseline seeds, ~3 GPU-hours total. **Louis's call
(2026-08-21): hold at 1v3 and read the results first; baseline seeds are a follow-up decision.**

Also from the launch audit: `3cad_z7_imgsz960_n_s1/_s2` and `tianchifabirc_z7_imgsz960_n_s0` were recorded
queued/running in expman but had already completed (101-line CSVs, both weights) — the same stale-status
class as before. Process/CSV checks above beat expman.

## Wave A results

Best epoch by `metrics/mAP50-95(B)`, AP_small read from the same row -- the convention that reproduces
every published baseline figure here (`3cad_baseline_n_s0` -> 0.2900 / 0.1882 / 0.2699).

### A2 on tianchifabirc: the strongest arm measured on this benchmark so far

3 seeds, Welch against the 3-seed baseline. **Both metrics clear p<0.05.**

| metric   | arm                | baseline | Δ           | ×floor    | t     | p          |
| -------- | ------------------ | -------- | ----------- | --------- | ----- | ---------- |
| mAP50-95 | 0.2158 (sd 0.0054) | 0.1909   | **+0.0249** | **+5.19** | +7.28 | **0.0069** |
| AP_small | 0.1335 (sd 0.0041) | 0.1220   | +0.0114     | +1.36     | +3.37 | **0.0282** |

Per-seed mAP50-95 `[0.2211, 0.2159, 0.2103]`, AP_small `[0.1361, 0.1287, 0.1356]`.

For scale, the incumbent `k=6` peaks at 3cad mAP50-95 +0.0156 with t=3.77, p=0.0488, and costs 1.35x
FLOPs. **A2 is a larger and more significant effect at 1.00x inference cost**, and it lands on the dataset
section 2 calls the highest-variance and hardest to move. It does cost ~12% training wall clock
(123 vs 110 s/epoch): rfla runs a topk over all 8400 anchors per GT where `inside` runs a boolean test.

### A1 on 3cad: AP_small confirmed, dose plateaus, mAP50-95 not resolved

2 seeds each (third pair running). p at n=2 has df≈2 and is indicative only; the effect sizes are not.

| arm           | mAP50-95           | Δ       | ×floor | p      | AP_small           | Δ           | ×floor    | p          |
| ------------- | ------------------ | ------- | ------ | ------ | ------------------ | ----------- | --------- | ---------- |
| `min_side=16` | 0.2962 (sd 0.0035) | +0.0053 | +1.27  | 0.2394 | 0.2315 (sd 0.0063) | **+0.0390** | **+2.09** | **0.0123** |
| `min_side=32` | 0.3042 (sd 0.0066) | +0.0134 | +3.18  | 0.1961 | 0.2314 (sd 0.0024) | **+0.0389** | **+2.08** | **0.0127** |

**16 and 32 give the same AP_small gain to four decimals (+0.0390 vs +0.0389).** The dose response
plateaus, so the single-seed zigzag (16: +1.85, 24: +0.74, 32: +2.17) was seed noise, as suspected. Prefer
`min_side=16` on the principle that it is the smallest dose that buys the effect -- and note it is also the
dose that merely removes the clamp's discontinuity rather than over-inflating every GT.

`min_side=32` shows the larger mAP50-95 (+3.18x floor vs +1.27x) but neither is significant, and 3cad's
mAP50-95 floor is its tight one (0.0042), so this is the cell most likely to move with the third seed.

### A2/A3 on 3cad: reproducible collapse, three hypotheses refuted, mechanism unknown

`tal_prior=rfla` (both rf scales) and `tal_metric=nwd` (gamma 0.5 and 1.0) peak at epoch 6-22 and collapse
to mAP 0. `nwd gamma=2.0` is the only survivor of the five. Same configs are fine on the other two
datasets. Established, in order:

1. **Reproducible.** `scripts/anomaly_bench/assigner_probe.py` on `3cad` + `rfla` matches
   `3cad_a2_rfla1_n_s0` epoch for epoch: 0.0250 at ep1, 0.0576 peak at ep6, 0.0025 by ep9. args differ in
   `exist_ok` alone.
2. **Not a soft-label collapse.** The hypothesis was that `target_scores.sum()` falls under the
   `max(..., 1)` clamp in `v8DetectionLoss.loss`, turning `loss[1]` from a mean into a raw sum. Refuted:
   across 482 logged steps spanning the collapse, o2m held 9.4-9.6 positives per GT, o2o held **exactly
   1.000** (the `topk2=1` contract), `target_scores.sum()` ran 64-494, `tgt_max` stayed 0.90-0.99, and the
   clamp fired **0/482** times. The assignment is healthy the whole way through.
3. **Not the o2o head.** `tal_heads=o2m` (A2.1, `3cad_a21_rfla1o2m_n_s0`) leaves o2o on a stock assigner
   and collapses at **epoch 9 identically**. The damage reaches the inference head through the shared
   trunk that o2m trains. After ep9 the whole model diverges: box_loss saturates 2.19 -> 4.09, cls_loss
   climbs past 25.
4. **The signature is output mode collapse.** At an identical `conf=0.001`, the collapsed weights emit
   **300 boxes/image** -- saturating `max_det` -- against the baseline's 6.0, and all one size:

   |                | boxes/img | width median | width sd | height median | height sd | both sides in [6,11] px |
   | -------------- | --------- | ------------ | -------- | ------------- | --------- | ----------------------- |
   | rfla collapsed | 300       | 15.5         | **4.67** | 23.6          | **5.01**  | 0.0%                    |
   | baseline       | 6.0       | 19.0         | 103.44   | 24.0          | 52.56     | 7.9%                    |

   It predicts roughly the dataset-average box everywhere. With no NMS on o2o that annihilates precision.
   **Not** the "predict your own receptive field" shortcut -- 0.0% of boxes are cell-sized.

5. **Not a representational limit.** At `reg_max=1` the box branch is a bare `Conv2d(c2, 4, 1)` with
   `dfl = nn.Identity()`, so ltrb may go negative and an anchor outside its GT can represent that GT.
6. **It is schedule-dependent.** The identical config at `epochs=20` never collapses and reaches AP 0.226.
   `lrf` decays over `epochs`, so a 20-epoch run holds a far lower LR at epoch 6 than the 100-epoch arm --
   which points at sustained post-warmup learning rate rather than at anything in the assignment.

**Parked here.** Three refuted hypotheses is enough; A2 on 3cad is unusable at this benchmark's schedule
and the cause is open. The one cheap lead left is (6): `cos_lr=True` or a lower `lr0` on 3cad + rfla, which
would deviate from the fixed protocol and so could only support the claim "A2 needs a gentler schedule",
not a comparable number.

**Do not read this as "A2 fails on 3cad."** A2 on 3cad is _unmeasured_ -- the arm never trained.

## A1 at 3 seeds, and the A2/A3 stability fixes

### A1 on 3cad confirmed: both doses, both metrics, p<0.05

| arm               | mAP50-95           | Δ       | ×floor | t    | p          | AP_small           | Δ           | ×floor    | t    | p          | FLOPs |
| ----------------- | ------------------ | ------- | ------ | ---- | ---------- | ------------------ | ----------- | --------- | ---- | ---------- | ----- |
| `min_side=16`     | 0.2972 (sd 0.0031) | +0.0063 | +1.51  | 2.98 | **0.0482** | 0.2373 (sd 0.0110) | **+0.0448** | **+2.39** | 5.39 | **0.0062** | 1.00× |
| `min_side=32`     | 0.3035 (sd 0.0048) | +0.0127 | +3.01  | 4.18 | **0.0306** | 0.2281 (sd 0.0059) | +0.0356     | +1.90     | 5.60 | **0.0081** | 1.00× |
| `k=6` (incumbent) | --                 | +0.0156 | +3.71  | 3.77 | 0.0488     | --                 | +0.0468     | +2.50     | 3.52 | 0.0444     | 1.35× |

Per-seed mAP50-95 `[0.2987, 0.2937, 0.2992]` for 16 and `[0.2995, 0.3089, 0.3021]` for 32; AP_small
`[0.2270, 0.2360, 0.2488]` and `[0.2331, 0.2297, 0.2216]`.

**`min_side=16` reaches 96% of `k=6`'s AP_small gain (+0.0448 vs +0.0468) at 1.00× FLOPs against 1.35×,
with a tighter p-value on both metrics.** No clean winner between the doses: 16 takes AP_small, 32 takes
mAP50-95, and 16 carries the larger AP_small spread (sd 0.0110 vs 0.0059). Since the two knobs act on
different bottlenecks, the `A1 × k=6` cell in §5 is now the interesting one.

### A2.2 fixes the collapse

`tal_prior=rfla_fill` on 3cad, seed 0. Monotone throughout where `rfla` was dead by epoch 9:

| epoch                     | 1      | 6      | 9          | 12     | 20     | 40     | 60     | 80         |
| ------------------------- | ------ | ------ | ---------- | ------ | ------ | ------ | ------ | ---------- |
| cls_loss                  | 15.27  | 2.66   | 2.36       | 2.15   | 1.92   | 1.58   | 1.37   | 1.12       |
| mAP50-95                  | 0.0238 | 0.0519 | 0.0647     | 0.0844 | 0.1270 | 0.2224 | 0.2663 | **0.2943** |
| `rfla` mAP for comparison | 0.0250 | 0.0576 | **0.0025** | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000     |

At epoch 80 it is already above the baseline's best (0.2900). So the junk-positive diagnosis holds: the
damage was `rfla` handing every starved GT `topk - inside_pool` outside-GT anchors **and** stripping
healthy GTs of candidates. Restricting the top-up to what is actually short removes both.

### A3.1 does not fix it -- `tal_beta` is a real lever but not the cause

| arm                    | ep20                  | ep40                  | ep60          | ep80          | verdict                                                               |
| ---------------------- | --------------------- | --------------------- | ------------- | ------------- | --------------------------------------------------------------------- |
| `beta=6` (original A3) | cls 1.65 / mAP 0.0120 | collapsed             |               |               | dies ~ep25                                                            |
| `beta=2`               | cls 1.47 / mAP 0.0583 | cls 2.97 / mAP 0      | cls **34633** | cls **19742** | dies ~ep30, explodes far worse                                        |
| `beta=1`               | cls 1.43 / mAP 0.0173 | cls 1.18 / mAP 0.0640 | running       |               | no explosion yet, but mAP ≈ 0.06 against the baseline's ~0.17 at ep30 |

**Lowering beta moves the failure rather than removing it**, and `beta=2` explodes an order of magnitude
harder than `beta=6`. `beta=1` is the first nwd arm to reach epoch 55 without exploding, but at a third of
the baseline's mAP, so "stable" is not yet "working".

The hypothesis was that `beta=6` is calibrated to CIoU and over-sharpens `bbox_nwd`, which sits close to
1.0 for any near hit. That is **half right at best**: beta demonstrably changes when and how the arm fails,
so it is a live lever, but it is not the root cause. A3 stays open. Note the two fixes were independent by
construction -- A2 runs `tal_metric=ciou` so beta cannot touch it, and A3 runs `tal_prior=inside` so the
junk-positive story cannot -- and only one of the two diagnoses survived contact.

## Wave A results — final tables

Best epoch by `metrics/mAP50-95(B)`, AP_small read from the same row; Welch against the 3-seed baseline.

### A1 (tal_min_side) — 3cad-specific, now final on all three datasets

| dataset                 | mAP50-95 Δ (×floor) | t / p             | AP_small Δ (×floor) | t / p             | verdict             |
| ----------------------- | ------------------- | ----------------- | ------------------- | ----------------- | ------------------- |
| 3cad `min_side=16`      | +0.0063 (+1.51)     | 2.98 / **0.0482** | +0.0448 (+2.39)     | 5.39 / **0.0062** | **confirmed**       |
| 3cad `min_side=32`      | +0.0127 (+3.01)     | 4.18 / **0.0306** | +0.0356 (+1.90)     | 5.60 / **0.0081** | **confirmed**       |
| tianchifabirc `16`      | +0.0035 (+0.73)     | 2.00 / 0.1208     | +0.0035 (+0.42)     | 1.22 / 0.3022     | **no effect**       |
| dspcbsd `16` (neg ctrl) | +0.0036 (+0.53)     | 1.13 / 0.3244     | +0.0070 (+4.35)     | 1.70 / 0.2282     | **passes as no-op** |

3cad per-seed AP_small for `16`: `.2270/.2360/.2488` against baseline `.1882/.2032/.1861` — all three
seeds above the baseline's max. The two doses specialise rather than plateau: `16` favours AP_small, `32`
favours mAP50-95. The dspcbsd +4.35x AP_small reading is not an effect — its AP_small floor is 0.0016,
so the ×floor column overstates, and p=0.23 says noise. A1 is therefore **3cad-only**: the mechanism is
specific to that dataset's geometry, not a universal small-object lever.

### A2 (rfla) vs A2.2 (rfla_fill) — the compensating prior kills the effect

`3cad_a22_rflafill_n_s0` read +1.49x on mAP50-95 at n=1; the second seed closed that: per-seed
`.2971/.2858` mAP, `.2012/.1864` AP_S → mean Δ +0.0006 (+0.15x) mAP, +0.0013 (+0.07x) AP_S, t≈0.1.
**A2.2 is a no-op on 3cad.** Training is stable, so the compensation design works as code; it simply
buys nothing there.

The decisive remaining question is whether A2.2 keeps A2's tianchifabirc win (+5.19x, p=0.0069). If it
drops there too, the win came specifically from the unconditional replacement — meaning A2's benefit is
the _removal_ of the inside-GT choice from well-fed GTs, not the feeding of starved ones, and the whole
line is dataset-specific like A1. `tianchifabirc_a22_rflafill_n_s0` is queued.

### A3.1 (nwd + beta) — stable at beta=1, but below baseline

`nwd beta=6` (original A3) and `beta=2` collapse; `beta=1` completes 100 epochs. `beta` therefore does
govern stability, and smaller is more stable. But `nwd beta=1` scores mAP50-95 0.2612, **−7.07x floor**
against baseline. The arm confounds two changes (metric swapped AND beta lowered); attributing the loss
needs a `ciou beta=1` control, which has not been run. Parked.

### A4 (k6 × ms16, 3cad) — first seed only

Seed 0: mAP50-95 0.3004 (+2.26x), AP_small 0.2085 (+0.86x). Below either single arm's AP_small
(k6 +2.50x, ms16 +2.39x), but k6's AP_small sd across seeds is 0.021, so one draw carries ±0.011 and no
additivity claim is possible yet. Two seeds queued.

### Phase D (COCO k=6) — logged here for completeness

COCO mAP50-95, best epoch: baseline 0.38646; `k6` seeds 0.38870 / 0.38870 / 0.38870. Δ +0.0022. No COCO
noise floor was ever established, so no significance claim; direction is positive and the size is small
against the 3cad-scale effects above.

## Wave A — final: A2.4, the tie bug, and the stability law

### A2.4 (`tal_prior=geom_topk`): A2's gain splits into tie-bug repair + unconditional replacement

`geom_topk` keeps the inside-GT pool and only swaps the topk RANKING from the prediction-aligned
`align_metric` to receptive-field distance, zeroed outside the pool. It was built after the earlier
"sliver re-routing" story was refuted (that came from a broken analysis script; rfla picks 10/0/0 across
P3/P4/P5 for real slivers, same as inside) -- and after a real bug was found in the default path.

**The tie bug (default code).** `select_topk_candidates` always takes the top-10 of `align_metric`, even
when a GT's inside pool is smaller than 10. The missing slots fill with all-zero ties from OUTSIDE the
pool, and `mask_pos = mask_topk * mask_in_gts` silently kills those positions -- each zero that out-ranks
an in-box anchor steals that anchor's positive. Synthetic: a 16x16 GT (pool 4) ends with 3 positives under
default, 4/4 under geom_topk. Real scale: tianchifabirc 670 GTs with pool<10 (median 5, 3511 in-box
anchors worst-case at risk), 3cad 777, dspcbsd only 213 (6.7% -- which matches A2's no-op there).

| dataset       | geom_topk (n=1)                            | vs A2 rfla       |
| ------------- | ------------------------------------------ | ---------------- |
| tianchifabirc | mAP 0.2038 (+2.69x floor), stable          | A2 +5.19x (n=3)  |
| dspcbsd       | +0.16x, no-op (negative control passes)    | A2 +0.57x        |
| 3cad          | **collapsed at ep11** (box 5.77, cls 1257) | A2 collapsed ep9 |

So the tie-bug repair is worth roughly half of A2's tianchifabirc win and the rest requires the
unconditional replacement. The 3cad collapse falsifies the commit-time prediction that keeping the inside
pool would prevent it.

### The stability law (3cad): remove the model's prediction from the topk ranking and it diverges

Every assigner change on 3cad now sorts cleanly by WHO decides the topk:

| arm                               | topk ranking                          | result                    |
| --------------------------------- | ------------------------------------- | ------------------------- |
| baseline, ms8/16/24/32, rfla_fill | prediction alignment (`align_metric`) | all stable                |
| rfla                              | none (fixed 10)                       | collapse ep9              |
| geom_topk                         | geometric distance                    | collapse ep11             |
| nwd beta<=2                       | swapped metric                        | collapse ep22 or earlier  |
| nwd beta=1, gamma=2               | softened swapped metric               | stable but below baseline |

Candidate-POOL changes (inflation, top-up) are stable because they leave the ranking alone. Anything that
removes or replaces the model's own alignment feedback diverges: 3cad needs that feedback to keep its
confidence branch from drifting into emitting everything (the collapsed outputs were 300 boxes/image at
max_det, one size). tianchifabirc is the opposite -- fixed geometry beats the model's own picks, plausibly
because prediction alignment on slivers is unreliable. This is the mechanism behind A2's dataset
specificity, replacing the refuted re-routing story.

### A4 close-out: k6 combinations do not stack on either dataset

tianchifabirc k6 x rfla, 3 seeds: mAP 0.2223 (+6.55x floor, t=15.43, p=0.0001) -- but vs rfla alone
Delta+0.0066, t=1.90, p=0.158. 3cad k6 x ms32 (3 seeds): AP_small +2.56x p=0.0405, essentially the max of
the two single arms. Bottlenecks (a) and (b) hit the same ceiling on both datasets; there is no additive
combination to harvest from stacking.

### Corrigendum

The "A2 routes slivers to coarse levels" mechanism recorded earlier is wrong -- it came from a script bug
(anchor-pool level indexing). Re-measured rfla picks are 10/0/0 (P3/P4/P5) for real slivers, identical to
inside. The correct mechanism is the topk-ranking control described above. The run.md sections that repeat
the routing story should be read with this in mind; the experimental numbers were never affected.

## S1/S2 — short-side inflation is the wrong lever for slivers; a boundary bug hid it

Two findings from the S1/S2 line, both measured on tianchifabirc:

### The pool inflation mostly buys P3, not the coarse levels

Per-column accounting of the pool for a 5x634 sliver (anchor centres at (i+0.5)\*stride, strict
inside-box test, box centred at 320):

| floor | P3      | P4  | P5  |
| ----- | ------- | --- | --- |
| 16    | 160     | 0   | 0   |
| 32    | 320     | 80  | 0   |
| 48    | **480** | 80  | 40  |
| 64    | **640** | 160 | 40  |

Every +8px of short side adds a full column of ~80 P3 anchors but only ~20 P5 per +32px. The
inflation is weighted by grid density, so the money goes to the finest level -- exactly where the
sliver already has hundreds of anchors. That is why the model keeps choosing P3: the pool is
flooded with them, not because coarse anchors are unselectable in principle.

### The boundary bug: floor == target stride yields ZERO anchors on that level

With the strict inside-box test, raising the short side to exactly the stride of the level we want
puts that level's single anchor column ON the boundary, and the strict inequalities kill it:
5x634 with floor=32 -> P5 = 0. The S2 rule `floor = stride of the long side's level` therefore
never delivers its own target level (it hits 32 exactly). A floor of 2x the stride (48-64) is
required before P5 anchors enter at all.

### Measured behaviour on the trained models

`scripts`-level probe (train-mode forward + assigner, 30 sliver GTs): a baseline checkpoint shows
64/30/6% positives on P3/P4/P5 and does not change when the S2 floor is applied post hoc --
converged models do not re-rank when the pool widens. An S1-trained checkpoint (side=32) shows
42/54/4% even with the floor off: the model DID learn a coarser preference during training, it
just did not turn it into mAP (S1 side=16: -0.11x floor; side=32: -1.69x floor). Combined with
the P3-flooding accounting, the lesson: **widening the pool is the wrong lever. Slivers need the
pool REBALANCED across levels (or hard-level assignment), not enlarged.**

## Wave B (S1/S2/A8) — the sliver line, and the P3-flood + boundary findings

Started from the question "why does A2 help tianchifabirc but collapse 3cad, and can one config
get the sliver gain without the collapse". A7 (ms16 + ar_rfla@4, 3 seeds) confirmed the gain on
tianchifabirc (mAP +5.26x floor, p=0.0035) and collapsed 3/3 seeds on 3cad -- the stability law
holds: on 3cad, any arm that removes the model's prediction from the topk ranking collapses
(ar_rfla alone also collapsed at ep11; the earlier 62-ep "still training" check counted rows but
not mAP and was wrong). A7 is the final tianchifabirc config.

Two measurement findings killed the "widen the pool" alternative:

1. P3 flooding. Raising the sliver's short side buys mostly P3 anchors: floor 16/32/48/64 on a
   5x634 sliver gives P3 pools 160/320/480/640 against P5 0/0/40/40. Inflation is weighted by
   grid density, so it floods the level that already has hundreds and the model keeps choosing
   P3. Trained-model probe confirms: baseline ckpt picks 64/30/6% P3/P4/P5 and does not re-rank
   when the pool widens post hoc; the S1(side=32)-trained ckpt moved to 42/54/4% but got no mAP
   out of it (S1 side=16: -0.11x floor; side=32: -1.69x floor on tianchifabirc).

2. Boundary bug. With the strict inside test, a floor equal to the target level's stride puts
   that level's single anchor column exactly on the box boundary and yields ZERO anchors of it
   (floor=32 -> P5=0). S2's floor = long-side stride therefore never delivered its own target
   level. A floor of 2x the stride is required. This also means S1 side=32 only ever offered P4.

S2 (floor='long') was queued and then cancelled on both datasets once the S1 data killed the
mechanism; the finding is recorded rather than run out.

### A8 (`tal_prior=level_assign`) — Louis's rule, parameter-free: hard level, soft within

Long side picks the target level (largest stride s with 2s <= long); the GT pools the inside-box
anchors of that level plus the finer neighbour; short side inflated to 2x the target stride
(boundary bug); topk stays the model's align_metric within the allowed levels. Measured pools:
5x634 -> 0/160/40 (P3 flood gone), 40x40 -> 16/4/0 (squares naturally stay fine, no AR gate),
100x100 -> 0/36/16, 40x10 -> 64/16/0, 8x8 -> 4/0/0. Runs: tianchifabirc (68 ep),
3cad (18 ep -- the stability-law judgment), dspcbsd (5 ep, negative control). The 3cad run is the
first arm to test whether a hard LEVEL cut alone destabilizes when the within-level ranking is
left to the model; if it trains, the stability law's precise boundary is the within-level choice.

### Queue state 2026-08-23 12:10

Training: tianchifabirc_a8 (68 ep, gpu3), 3cad_a8 (18 ep, gpu4), 3cad_s2_long (19 ep, gpu4 —
leftover waiter race, will be killed), 3cad_y11_base_s2 (19 ep, gpu5), dspcbsd_a8 (5 ep, gpu5),
coco_a7_s2 (14 ep, gpu6), coco_baseline_s2 (25 ep, gpu7). Queued yolo11 wave: base + A7 for
3cad/tianchifabirc/coco, s0/s1 each (s2 chains already running/launched separately). yolo11
per-arm wall time is ~2.1x yolo26n, so the wave is the long tail; results expected late
2026-08-23 / 24.

# EXPERIMENT INDEX — maintained, canonical

**Every run on this branch, one row each. Keep this current: add a row when a run is launched (status
`running`, metrics `--`) and fill it in when the run completes.** Sections above hold the reasoning and the
verbatim commands; this table is the inventory.

Common to all rows unless stated: `yolo26n`, 100 epochs, `imgsz=640`, `batch=128`, `seed=0`,
`coco_eval=True`, `project=yolo26-defect-bench`, data under
`/data/shared-datasets/louis_data/anomaly_bench/<dataset>/data.yaml`.

`ΔAP_S (×floor)` and `ΔmAP (×floor)` are changes against that dataset's own baseline mean, in multiples of
**that dataset's own 2sd floor**. Earlier revisions of this table expressed 3cad in `dspcbsd` floor units;
that was wrong and the 3cad rows are restated here (see the floor section above).

| dataset       | baseline seeds | 2sd mAP50-95 | 2sd AP_small | baseline mAP50-95 | baseline AP_small |
| ------------- | -------------- | ------------ | ------------ | ----------------- | ----------------- |
| dspcbsd       | 3              | 0.0067       | 0.0016       | 0.4765            | 0.3985            |
| 3cad          | 3              | 0.0042       | **0.0187**   | 0.2908            | 0.1925            |
| tianchifabirc | 1 (2 queued)   | --           | --           | 0.1936            | 0.1255            |

**`tianchifabirc` still has no floor, so every multiple in its rows is `n/a`, not "small".** Nothing in that
column is a significance claim until `tianchifabirc_baseline_n_s1/_s2` land.

| run                              | dataset       | variable          | status    | mAP50-95 | AP_S   | AP_M   | ΔAP_S (×floor)      | verdict                                 |
| -------------------------------- | ------------- | ----------------- | --------- | -------- | ------ | ------ | ------------------- | --------------------------------------- |
| `dspcbsd_baseline_n_s0`          | dspcbsd       | baseline          | completed | 0.4737   | 0.3982 | 0.5437 | --                  | baseline                                |
| `3cad_baseline_n_s0`             | 3cad          | baseline          | completed | 0.2900   | 0.1882 | 0.2699 | --                  | baseline                                |
| `tianchifabirc_baseline_n_s0`    | tianchifabirc | baseline          | completed | 0.1936   | 0.1255 | 0.1870 | --                  | baseline                                |
| `dspcbsd_baseline_n_s1`          | dspcbsd       | baseline seed 1   | completed | 0.4802   | 0.3979 | 0.5543 | --                  | baseline                                |
| `3cad_baseline_n_s1`             | 3cad          | baseline seed 1   | completed | 0.2932   | 0.2032 | 0.2752 | --                  | baseline                                |
| `dspcbsd_baseline_n_s2`          | dspcbsd       | baseline seed 2   | completed | 0.4756   | 0.3994 | 0.5516 | --                  | baseline                                |
| `3cad_baseline_n_s2`             | 3cad          | baseline seed 2   | completed | 0.2893   | 0.1861 | 0.2573 | --                  | baseline                                |
| `dspcbsd_z1_dfl0_n_s0`           | dspcbsd       | `dfl=0`           | completed | 0.4752   | 0.3920 | 0.5462 | -0.0065 (-4.1×)     | inconclusive, datasets disagree         |
| `3cad_z1_dfl0_n_s0`              | 3cad          | `dfl=0`           | completed | 0.2938   | 0.2012 | 0.3224 | +0.0087 (+0.47×)    | noise — Z1 closed                       |
| `tianchifabirc_z1_dfl0_n_s0`     | tianchifabirc | `dfl=0`           | completed | 0.1925   | 0.1293 | 0.1792 | n/a (no floor)      | noise — Z1 closed                       |
| `dspcbsd_z2_clspw05_n_s0`        | dspcbsd       | `cls_pw=0.5`      | completed | 0.4720   | 0.3874 | 0.5440 | -0.0111 (-6.9×)     | fail                                    |
| `3cad_z2_clspw05_n_s0`           | 3cad          | `cls_pw=0.5`      | completed | 0.2719   | 0.1740 | 0.2530 | -0.0185 (-0.99×)    | fail (real evidence is mAP50-95, -4.5×) |
| `tianchifabirc_z2_clspw05_n_s0`  | tianchifabirc | `cls_pw=0.5`      | completed | 0.1960   | 0.1274 | 0.1869 | n/a (no floor)      | fail                                    |
| `3cad_z2_clspw10_n_s0`           | 3cad          | `cls_pw=1.0`      | completed | 0.2301   | 0.1507 | 0.2155 | -0.0418 (-2.24×)    | fail                                    |
| `tianchifabirc_z2_clspw10_n_s0`  | tianchifabirc | `cls_pw=1.0`      | completed | 0.1998   | 0.1266 | 0.1928 | n/a (no floor)      | fail                                    |
| `dspcbsd_z3_k6_n_s0`             | dspcbsd       | `k=6` downsample  | completed | 0.4812   | 0.4049 | 0.5431 | +0.0064 (+4.0×)     | **pass**                                |
| `3cad_z3_k6_n_s0`                | 3cad          | `k=6` downsample  | completed | 0.3050   | 0.2186 | 0.2778 | +0.0261 (+1.40×)    | **mAP50-95 +3.4×; AP_S does not clear** |
| `tianchifabirc_z3_k6_n_s0`       | tianchifabirc | `k=6` downsample  | completed | 0.1972   | 0.1190 | 0.2019 | n/a (no floor)      | no floor yet — 3v3 queued               |
| `dspcbsd_z5_mosaic0_n_s0`        | dspcbsd       | `mosaic=0.0`      | completed | 0.4643   | 0.3873 | 0.5029 | -0.0112 (-7.0×)     | fail                                    |
| `3cad_z5_mosaic0_n_s0`           | 3cad          | `mosaic=0.0`      | completed | 0.2632   | 0.1756 | 0.2532 | -0.0169 (-0.90×)    | fail (real evidence is mAP50-95, -6.6×) |
| `tianchifabirc_z5_mosaic0_n_s0`  | tianchifabirc | `mosaic=0.0`      | completed | 0.1820   | 0.1142 | 0.1726 | n/a (no floor)      | fail                                    |
| `dspcbsd_z7_imgsz960_n_s0`       | dspcbsd       | `imgsz=960`       | completed | 0.4831   | 0.4057 | 0.5637 | +0.0072 (+4.5×)     | reference, not a candidate              |
| `3cad_z7_imgsz960_n_s0`          | 3cad          | `imgsz=960`       | completed | 0.3132   | 0.2827 | 0.2979 | +0.0902 (+4.82×)    | reference, not a candidate              |
| `dspcbsd_m1_ir07_n_s0`           | dspcbsd       | `inner_ratio=0.7` | completed | 0.4796   | 0.3915 | 0.5500 | -0.0070 (-4.4×)     | fail                                    |
| `dspcbsd_m1_ir08_n_s0`           | dspcbsd       | `inner_ratio=0.8` | completed | 0.4701   | 0.3857 | 0.5402 | -0.0128 (-8.0×)     | fail                                    |
| `dspcbsd_m1_ir12_n_s0`           | dspcbsd       | `inner_ratio=1.2` | completed | 0.4771   | 0.4003 | 0.5377 | +0.0018 (+1.1×)     | fail (inside the floor)                 |
| `dspcbsd_m3_topk22_n_s0`         | dspcbsd       | `o2o_topk2=2`     | completed | 0.3721   | 0.3185 | 0.3915 | -0.0800 (-50.0×)    | **structurally invalid**                |
| `dspcbsd_m3_topk23_n_s0`         | dspcbsd       | `o2o_topk2=3`     | completed | 0.3184   | 0.2744 | 0.3340 | -0.1241 (-77.6×)    | **structurally invalid**                |
| `dspcbsd_m3_topk24_n_s0`         | dspcbsd       | `o2o_topk2=4`     | completed | 0.2744   | 0.2415 | 0.2949 | -0.1570 (-98.1×)    | **structurally invalid**                |
| `dspcbsd_z3_k6_n_s1`             | dspcbsd       | `k=6` seed 1      | completed | 0.4776   | 0.3997 | 0.5488 | +0.0041 (3v3 mean)  | k=6 arm                                 |
| `dspcbsd_z3_k6_n_s2`             | dspcbsd       | `k=6` seed 2      | completed | 0.4802   | 0.4033 | 0.5512 | +0.0041 (3v3 mean)  | k=6 arm                                 |
| `3cad_z3_k6_n_s1`                | 3cad          | `k=6` seed 1      | completed | 0.3004   | 0.2387 | 0.2884 | +0.0468 (3v3 mean)  | k=6 arm                                 |
| `3cad_z3_k6_n_s2`                | 3cad          | `k=6` seed 2      | completed | 0.3139   | 0.2607 | 0.2852 | +0.0468 (3v3 mean)  | k=6 arm; read from results.csv          |
| `3cad_z3xz7_k6_imgsz960_n_s0`    | 3cad          | `k=6` + `960`     | completed | 0.3254   | 0.2470 | --     | --                  | best 3cad mAP50-95 so far               |
| `3cad_z7_imgsz960_n_s1`          | 3cad          | `imgsz=960` s1    | completed | --       | --     | --     | --                  | completed (expman status was stale)     |
| `3cad_z7_imgsz960_n_s2`          | 3cad          | `imgsz=960` s2    | completed | --       | --     | --     | --                  | completed (expman status was stale)     |
| `dspcbsd_z7_imgsz960_n_s1`       | dspcbsd       | `imgsz=960` s1    | completed | 0.4807   | 0.4159 | 0.5581 | --                  | completed                               |
| `dspcbsd_z7_imgsz960_n_s2`       | dspcbsd       | `imgsz=960` s2    | completed | 0.4725   | 0.3930 | 0.5565 | --                  | completed                               |
| `3cad_z3xz7_k6_imgsz960_n_s1`    | 3cad          | `k=6` + `960` s1  | running   | --       | --     | --     | --                  | running (GPU 6, PID 3530124)            |
| `tianchifabirc_z7_imgsz960_n_s0` | tianchifabirc | `imgsz=960`       | completed | --       | --     | --     | --                  | completed (expman status was stale)     |
| `tianchifabirc_baseline_n_s1`    | tianchifabirc | baseline seed 1   | completed | 0.1891   | 0.1173 | 0.1782 | --                  | baseline                                |
| `tianchifabirc_baseline_n_s2`    | tianchifabirc | baseline seed 2   | completed | 0.1898   | 0.1233 | 0.1826 | --                  | baseline                                |
| `tianchifabirc_z3_k6_n_s1`       | tianchifabirc | `k=6` seed 1      | completed | 0.1957   | 0.1442 | 0.2004 | +0.0071 (3v3 mean)  | k=6 arm — no harm, t=0.88               |
| `tianchifabirc_z3_k6_n_s2`       | tianchifabirc | `k=6` seed 2      | completed | 0.1899   | 0.1241 | 0.1949 | +0.0071 (3v3 mean)  | k=6 arm — no harm, t=0.88               |
| `coco_baseline_n_s0`             | coco          | baseline          | running   | --       | --     | --     | n/a (no coco floor) | baseline (n=1)                          |
| `coco_z3_k6_n_s0`                | coco          | `k=6` downsample  | running   | --       | --     | --     | n/a (no coco floor) | k=6 arm                                 |
| `coco_z3_k6_n_s1`                | coco          | `k=6` seed 1      | running   | --       | --     | --     | n/a (no coco floor) | k=6 arm                                 |
| `coco_z3_k6_n_s2`                | coco          | `k=6` seed 2      | queued    | --       | --     | --     | n/a (no coco floor) | queued (GPU 6, after the 960 job)       |
| `3cad_z1_dfl0_n_s1`              | 3cad          | `dfl=0` seed 1    | cancelled | --       | --     | --     | --                  | cancelled before start                  |
| `3cad_z1_dfl0_n_s2`              | 3cad          | `dfl=0` seed 2    | cancelled | --       | --     | --     | --                  | cancelled before start                  |

| `3cad_a1_ms16_n_s0` | 3cad | `tal_min_side=16` | running | -- | -- | -- | -- | screening, n=1 -- not evidence |
| `3cad_a1_ms24_n_s0` | 3cad | `tal_min_side=24` | running | -- | -- | -- | -- | screening, n=1 -- not evidence |
| `3cad_a2_rfla1_n_s0` | 3cad | `tal_prior=rfla` rf1 | running | -- | -- | -- | -- | screening, n=1 -- not evidence |
| `3cad_a3_nwd10_n_s0` | 3cad | `tal_metric=nwd` g1.0 | running | -- | -- | -- | -- | screening, n=1 -- not evidence |
| `3cad_a1_ms08_n_s0` | 3cad | `tal_min_side=8` | queued | -- | -- | -- | -- | screening, n=1 -- not evidence |
| `3cad_a1_ms32_n_s0` | 3cad | `tal_min_side=32` | queued | -- | -- | -- | -- | screening, n=1 -- not evidence |
| `3cad_a2_rfla4_n_s0` | 3cad | `tal_prior=rfla` rf4 | queued | -- | -- | -- | -- | screening, n=1 -- not evidence |
| `3cad_a3_nwd20_n_s0` | 3cad | `tal_metric=nwd` g2.0 | queued | -- | -- | -- | -- | screening, n=1 -- not evidence |
| `3cad_a3_nwd05_n_s0` | 3cad | `tal_metric=nwd` g0.5 | queued | -- | -- | -- | -- | screening, n=1 -- not evidence |
Also on disk, excluded above because they carry no interpretable numbers: `smoke_{3cad,dspcbsd,tianchifabirc}_n`
and `_n_v2` (Phase 0 data-loading checks) and `k6_donor` (the CPU job that builds the Z3 donor; `lsta`
reports it FAILED, which is a misclassification — it writes no training completion marker).

## Not launched

| candidate              | why it is waiting                                                                               |
| ---------------------- | ----------------------------------------------------------------------------------------------- |
| Z4 P2 head             | held by Louis, and now the top candidate — the starvation table above is the case for it        |
| Z6 `scale=0.2`         | mechanism overlaps Z5, and Z5 failed on all three datasets — low prior now                      |
| M2 NWD blend           | closed: its gate Z1 is now measured as noise, and CIoU already carries both terms NWD would add |
| M3 `o2o_topk2`         | closed permanently, not waiting — breaks the NMS-free contract, proven above                    |
| M4 o2m `tal_topk`      | the inference-safe version of M3; near-no-op on 3cad by the pool table, so low prior            |
| anisotropic stem       | `k=(6,3)`/`k=(3,6)`; only worth it if the tianchifabirc 3v3 shows `k=6` really does lose there  |
| Z3 position ablation   | only if `k=6` survives the seed replicates — widen layer 3 only, or the stem too                |
| Z3 x Z7                | `k=6` at `imgsz=960` on 3cad, to test whether the two are additive                              |
| test-split + per-class | `eval_bench.py` over the finished runs; Phase A still owes these                                |

---

## Warm-pool probe — can the pretrained box head vote on pool MEMBERSHIP? (2026-08-23)

**Question.** Every A-wave knob moves stage 1 (the inside-GT pool) because the 3cad stability law
kills any arm that takes stage-2 ranking away from the model. One combination was untried: let the
*warm* box head (transferred by `intersect_dicts` whenever the shape matches; the cls head is not,
since `nc` differs from COCO) decide **membership** instead of order. This measures whether such a
vote would have anything to say — per GT at epoch 0, how many anchors sit OUTSIDE the inside-GT pool
while `bbox_nwd(pred_box, gt) >= tau`. NWD, not IoU: IoU vanishes at these sizes, and its flatness
is the point when it is used as a gate rather than a ranking.

Script: `scripts/anomaly_bench/warm_pool.py` (commit `3362cde5b`), o2m head, 10 batches, epoch 0,
`yolo26n.pt` donor, `imgsz=640 batch=128`. Logs + CSVs in `runs/yolo26-defect-bench/warm_<ds>/`.

```bash
# on ultra6, after `expman-cli bundle` from the laptop worktree
cd ~/ultra_louis_work/ultralytics
for d in 3cad tianchifabirc dspcbsd; do
  mkdir -p runs/yolo26-defect-bench/warm_$d
  ~/miniconda3/envs/ultra/bin/python scripts/anomaly_bench/warm_pool.py \
    --data /data/shared-datasets/louis_data/anomaly_bench/$d/data.yaml \
    --device 0 --steps 10 --project runs/yolo26-defect-bench --name warm_$d \
    > runs/yolo26-defect-bench/warm_$d/run.log 2>&1
done
```

### Result 1 — the starved fraction ranks the datasets exactly as the A-wave outcomes do

| dataset   | GTs measured | starved (pool<10) | starved short side | starved pool (P3/P4/P5) | A-wave outcome                    |
| --------- | ------------ | ----------------- | ------------------ | ----------------------- | --------------------------------- |
| `3cad`    | 1207         | **44.3%**         | 8.5 px             | 4.86 (3.66/0.93/0.26)   | ms16, ms32, k6 all confirmed      |
| `tianchi` | 3166         | **31.4%**         | 7.1 px             | 5.20 (3.95/1.00/0.25)   | rfla, A7 confirmed (+5.19/+5.26x) |
| `dspcbsd` | 3606         | **14.3%**         | 14.1 px            | 6.32 (4.65/1.36/0.30)   | "nothing moves it"                |

44% / 31% / 14% is the same order as responsiveness to assignment knobs, and dspcbsd's flatness now
has a measured cause rather than a guess. This is the first quantity that predicts, before training,
whether a dataset can respond to an assigner change at all.

### Result 2 — the warm gate fires, but weakly, and weakest where the biggest win already is

Anchors per GT gained (outside the pool and `nwd >= tau`), against the warm anchors already inside:

| dataset   | group   | tau=0.3 out/in | tau=0.5 out/in | tau=0.7 out/in | mean best nwd outside |
| --------- | ------- | -------------- | -------------- | -------------- | --------------------- |
| `3cad`    | starved | 6.10 / 2.46    | 2.34 / 1.93    | 0.74 / 1.21    | 0.4345                |
| `3cad`    | healthy | 145.16 / 74.72 | 17.83 / 38.60  | 2.01 / 10.06   | 0.6693                |
| `tianchi` | starved | 3.78 / 1.88    | 1.07 / 1.12    | 0.22 / 0.44    | 0.3247                |
| `tianchi` | healthy | 355.63 / 283.75| 12.23 / 74.85  | 0.66 / 6.86    | 0.4802                |
| `dspcbsd` | starved | 11.77 / 3.53   | 3.32 / 2.37    | 0.57 / 1.14    | 0.5205                |
| `dspcbsd` | healthy | 354.40 / 211.30| 28.87 / 101.60 | 2.59 / 23.15   | 0.6891                |

Readings:

- **The gate does not fill a starved pool.** At `tau=0.5` a starved 3cad GT goes 4.86 -> ~7.2, still
  under `topk=10`. Filling it needs `tau=0.3`, which simultaneously hands healthy GTs 145-355 extra
  anchors — the same flood failure mode that closed S1/S2.
- **The warm head's opinion about tiny defects is mediocre.** Mean best outside-pool NWD for starved
  GTs is only 0.4345 / 0.3247 / 0.5205, so at `tau=0.5` only 43.7% / 24.4% / 58.1% of starved GTs
  gain even one anchor. COCO-warm localisation does not transfer strongly to 8 px defects.
- **tianchi is the weakest**, yet it holds the largest confirmed win (A7, +5.26x). Consistent: A7/rfla
  win by rerouting slivers to P5, not by counting. This gate cannot reproduce that mechanism.

### Result 3 — tie-bug exposure at epoch 0, by dataset

Zero-`align_metric` anchors inside the pool (they lose the topk tie to out-of-pool anchors that
`mask_in_gts` then deletes): starved GTs carry 1.17 (3cad) / 2.06 (tianchi) / 0.74 (dspcbsd) of a
~5-anchor pool. On tianchi that is 40% of the pool dead at step 0 — matching the 670-GT count
measured earlier, and still consistent with `inside_fix` alone being a no-op (+0.63x).

### Verdict

The warm-union gate as specified is **not worth a 3-seed arm**: it cannot fill the starved pool at a
`tau` that leaves healthy GTs alone, and it is weakest on the dataset with the most headroom. The
probe's value is Result 1.

The one variant the data does support is **starvation-conditional** top-up — apply the warm gate only
where `pool < topk`, which makes the healthy-GT flood impossible by construction. That is exactly the
existing `topup_candidates_by_rfd` path with the top-up ranked by `bbox_nwd(pred_box, gt)` instead of
receptive-field distance. Prior against it: geometric top-up (`rfla_fill`) was already measured as a
no-op (+0.60x, p=0.578), so the whole "fill starved GTs" family may simply not be the lever.

### Aspect-ratio split of the same probe (2026-08-23)

Small square defects already have a cure (`tal_min_side`, confirmed on 3cad), so the open population
is the slivers. Re-analysing the same CSVs by aspect ratio, no new GPU time:

```bash
for d in 3cad tianchifabirc dspcbsd; do
  python scripts/anomaly_bench/anchor_pool.py --csv runs/yolo26-defect-bench/warm_$d/warm_pool.csv
done
```

| dataset   | AR 1-2 | AR 2-4 | AR 4-8 | AR 8-16 | AR >=16 | **AR >= 4** |
| --------- | ------ | ------ | ------ | ------- | ------- | ----------- |
| `3cad`    | 44.1%  | 29.8%  | 15.3%  | 7.8%    | 3.0%    | **26.1%**   |
| `tianchi` | 23.0%  | 22.5%  | 21.3%  | 13.4%   | 19.9%   | **54.6%**   |
| `dspcbsd` | 73.5%  | 18.9%  | 6.0%   | 1.4%    | 0.2%    | **7.6%**    |

Per-group detail (median short/long px, median pool, mean per-GT level share, % starved):

| dataset   | AR      | short | long  | pool | P3/P4/P5 per GT | P3-only | starved |
| --------- | ------- | ----- | ----- | ---- | --------------- | ------- | ------- |
| `3cad`    | 1-2     | 15.8  | 21.5  | 8.0  | 77/19/5         | 10.2%   | 54.1%   |
| `3cad`    | 4-8     | 9.7   | 50.8  | 14.0 | 77/18/5         | 6.5%    | 31.9%   |
| `3cad`    | 8-16    | 9.6   | 103.9 | 25.0 | 76/19/5         | 3.2%    | 4.3%    |
| `3cad`    | >=16    | 9.8   | 247.4 | 57.0 | 75/21/4         | 0.0%    | 0.0%    |
| `tianchi` | 1-2     | 16.4  | 23.6  | 8.0  | 76/20/5         | 10.3%   | 52.3%   |
| `tianchi` | 4-8     | 7.6   | 47.2  | 13.0 | 76/19/5         | 2.7%    | 34.2%   |
| `tianchi` | 8-16    | 9.2   | 107.2 | 26.0 | 76/20/5         | 2.4%    | 7.8%    |
| `tianchi` | >=16    | 7.5   | 291.2 | 76.5 | 75/20/5         | 3.0%    | 0.0%    |
| `dspcbsd` | 1-2     | 37.5  | 48.0  | 38.0 | 76/19/5         | 0.9%    | 14.9%   |
| `dspcbsd` | >=16    | 9.2   | 192.1 | 36.0 | 69/28/2         | 0.0%    | 0.0%    |

**Sliver share predicts the sliver-knob effect size.** AR>=4 is 54.6% / 26.1% / 7.6% for
tianchi / 3cad / dspcbsd, matching where `rfla` and A7 pay: tianchi +5.19x/+5.26x, 3cad collapses
(its 26% cannot pay for the ranking damage), dspcbsd nothing. Together with the starvation table
above, the two populations now have separate, measured causes.

**Law — the pool's level composition is constant at 76/19/5.** A GT's per-level pool is its area over
that level's stride squared, so the shares are `(1/64):(1/256):(1/1024)` = 76:19:5 for *every* GT:
tiny, huge, square or sliver (measured range across all 15 groups: 69-77 / 18-28 / 2-5%). Two
consequences:

- **No box-widening knob can change which level answers for a sliver.** Widening scales all three
  levels together and leaves the share fixed — this is the mechanism behind the S1/S2 P3 flood, and
  it makes the whole "raise the short side" family dead for slivers, not just badly dosed.
- Only a prior that *replaces* the pool with a level-aware rule (`rfla`, `ar_rfla`) can move a sliver
  to a coarse level. A7 (`ms16` + `ar_rfla4`) is exactly that split — geometric cure for the square
  population, pool replacement for the sliver population — which is now explained rather than found.

**Correction to an earlier assumption.** A short side under one cell does not zero that level:
centres sit at `(i + 0.5) * stride`, so a 11.5 px side still catches a P4 row whenever it straddles
one. `P3-only` never exceeds 10.3% in any group. The recorded `11.5x637.5 -> 160/40/0` measurement is
consistent with this (P4 = 40, not 0); the P5 zero there is the `floor` of a 637.5 px long side, not
the short side vanishing.

**Slivers are not starved.** By AR>=8 starvation is 4.3% / 7.8% / 18.0% and by AR>=16 it is 0%
everywhere. Candidate *count* is not the sliver problem, so `topk`, `min_side` and the warm-union
gate are all aimed at the wrong population.
