# TASK — YOLO26 baselines on the defect benchmark

**Branch** `yolo26-defect-bench` (based on upstream `main`) · **Host** ultra6
**Report results in** `run.md` at this repo root.

---

## Objective

Establish **baseline numbers and the run-to-run noise floor** for YOLO26 on three defect
datasets. Nothing else.

This is the prerequisite for the real work that follows — changing YOLO26's loss and head
for defect data. Without a noise floor, a later "+0.8 mAP" cannot be distinguished from
seed variance, and the whole ablation programme is worthless. This task is not a warm-up;
it is the measuring instrument.

**Do not modify any loss, head, or model code in this task.** Baselines must come from
unmodified upstream `main`. If you find yourself editing `ultralytics/utils/loss.py`, stop.

---

## The benchmark

Location on ultra6 — data, audit scripts, and a copy of this description all live here:

```
/data/shared-datasets/louis_data/anomaly_bench/
~/ultra_louis_work/datasets/anomaly_bench          # same thing, symlink
```

| dataset | train / val / test | classes | labels | normal imgs | what it tests |
|---|---|---|---|---|---|
| `3cad` | 21,424 / 2,575 / 2,652 | 24 | yolo-seg | 58% | object defects, multi-class, **very small targets** (69% of boxes < 0.1% of image area), **false positives on normal images** |
| `tianchifabirc` | 10,332 / 1,185 / 1,195 | 20 | yolo-detect | ~0 | **texture defects**, **extreme aspect ratio** (45% of boxes AR>5, 26% AR>10) |
| `dspcbsd` | 6,575 / 1,633 / 2,051 | 9 | yolo-detect | 0 | small targets (58% of boxes < 1% area), PCB domain, zero augmentation |

```
/data/shared-datasets/louis_data/anomaly_bench/3cad/data.yaml
/data/shared-datasets/louis_data/anomaly_bench/tianchifabirc/data.yaml
/data/shared-datasets/louis_data/anomaly_bench/dspcbsd/data.yaml
```

~46k images. Chosen so the three together cover four axes a loss change can plausibly
move: **texture vs object**, **small / elongated targets**, **multi-class**, and **false
positives**. `tianchifabirc` stresses IoU/DFL regression hardest — long thin fabric
defects are where box regression breaks.

### Why only these three

Two requirements drove the selection:

1. **Defect images in both train and val.** Many anomaly datasets follow the MVTec
   paradigm — `train/` holds only normal images. Those are built for training-free /
   memory-bank methods; a supervised detector trained on them yields a val metric that
   says nothing. Excluded: `visa`, `goodsad`, `mvtec-ad`, `mpdd`, `btad`.
2. **No split leakage.** This turned out to be the real problem.

Many public defect datasets are small (a few hundred to ~1k real images) and were
**offline augmented — rotation × lighting × flips — and only then split**. The same source
image then appears in both train and val, so the val metric measures memorisation and
every ablation on top of it is noise. Measured leakage (share of val *source* images also
present in train, after stripping augmentation and multi-view markers):

| dataset | train imgs | unique sources | leak% |
|---|---|---|---|
| `realiad` (30 products) | 121k total | — | **100%** |
| `wtbd-mhsa` | 2,508 | 330 | 98.6% |
| `metal-defect-bristol` | 11,606 | 1,491 | 96.2% |
| `crack` | 1,290 | 514 | 84.0% |
| `pcb-defect` | 20,365 | 935 | 76.3% |
| `metal-surface-defect` | 11,845 | 2,307 | 9.3% |
| `fabric1` | 1,450 | 1,325 | 8.9% |

`pcb-defect` looked like the best candidate at first — 24,896 images, 6 classes. It is
about **1,100 real images** augmented ~20× and then split. Replaced by `dspcbsd`, same PCB
small-target domain with zero augmentation.

`realiad` leaks for a different reason and is **recoverable later**: the data is excellent
(121k images, uniform 512×512, stable good/defect ratio), but each part is photographed
from 5 cameras (C1–C5) and the split is per *image* instead of per *sample*, so the same
part with the same physical defect lands in train and val. Re-splitting on the `S####`
sample id would fix it. Not done.

Also excluded despite being leak-free: `dagm` and `texturead-*` (1 defect class per
product, 11–39 val defect images), `neu-det` (1.8k images, median box area 11.8% — all
large targets), `casedefec` (val is 24 images).

**Before adding any new dataset to this benchmark, run the leakage audit on it first.**
Seven of the candidates we examined leaked, including the two largest.

### How `3cad` was built

`3cad` is a **real merged copy**; the other two are symlinks to their originals. It had to
be, because the 8 per-product 3CAD exports use inconsistent class spaces — same class
name, different id per product:

```
camera-cover           10 classes   bruise = id 1
aluminum-ipad           4 classes   bruise = id 1
aluminum-middle-frame   6 classes   bruise = id 2
aluminum-pc            15 classes   bruise = id 3
copper-stator           1 class     (wire_damage only)
```

Concatenating them would have silently corrupted every class id and made multi-class mAP
meaningless. `merge_3cad.py` remaps everything to one **24-class vocabulary**. Three other
decisions:

- **`multiple_defects` images dropped whole — 388 images, 1.4%.** It is a meta-label
  meaning "this image has several defect types", not a defect type, and **no image
  carrying it carries any other real class**. Dropping only the boxes would have left
  real, visible defects unlabelled, teaching the model to call them normal — worse than
  keeping a useless class.
- **Filenames prefixed with the product name** — 4,398 basenames collide across products.
- **Normal (empty-label) images kept** — 58% of the data, and the only reason this
  benchmark can measure false positives at all.

Verified post-merge: 26,651 images, 0 unpaired image/label, class ids 0–23 all present,
polygon arity valid, 0 cross-split filename overlap.

| | train | val | test |
|---|---|---|---|
| good | 12,544 | 1,471 | 1,562 |
| defect | 8,880 | 1,104 | 1,090 |

---

## Environment

Training runs on **ultra6** (8× RTX PRO 6000). This worktree is on the laptop; you write
and commit code here, then ship it with the **`expman` skill**. Read that skill before you
run anything — every ultra6 action (sync, launch, status, kill, GPU check, pulling weights
and logs) goes through `expman-cli`. **Do not hand-roll ssh / scp / rsync / nohup /
nvidia-smi.**

The datasets are on ultra6 only. You cannot train locally.

### How code reaches ultra6

Two steps, run **adjacently** — `launch --snap` pins whatever commit `bundle` just pushed,
so another `bundle` slipping in between would pin the wrong one:

```bash
# from THIS worktree (bundle mirrors the repo at your cwd)
expman-cli bundle
expman-cli launch --snap --args "nohupyolo 0 train data=... model=... project=... name=..."
```

- `bundle` pushes local HEAD to ultra6 and **refuses a dirty tracked tree** — committing
  first is a hard precondition, not a style rule. Untracked files are fine.
- `--snap` materialises that commit as an immutable detached worktree at
  `~/ultra_louis_work/ultralytics_snap/<hash12>` and runs there. **Always use it.** This is
  also your identity binding: the snapshot dir name *is* the commit, so a result can never
  drift away from the code that produced it. Same commit reuses the same snapshot.
- ultra6's own branch/checkout state is irrelevant to you. Never `git checkout` on ultra6.
- Run `expman-cli gcsnap` occasionally — snapshots never self-delete.

### Three gotchas that will cost you a run each

1. **Snapshots carry no untracked files.** The `*.pt` weights sitting in ultra6's main
   checkout root are *not* in the snapshot. Pass an **absolute path** for the weights, or
   the job dies seconds after launch.
2. **`nohupyolo` already wraps the `yolo` CLI.** The args start with `train` / `val`, not
   `yolo train`. `nohupyolo 0 yolo train ...` double-wraps and crashes.
3. **`launch` refuses if the run dir already exists** (otherwise it would auto-delete it,
   weights included). Bump `name=` rather than reaching for `--force`.

Check for a free GPU with `expman-cli cuda` before picking `device=N`. Before `bundle`,
run `expman-cli lsta` — if any job is running from the *main checkout* rather than a
snapshot, `bundle` would interrupt it.

---

## What to run

### Step 0 — smoke test before anything expensive

One short run (2 epochs, `fraction=0.05`) per dataset. Purpose is only to confirm the data
loads, not to get numbers. Verify specifically:

- **`3cad` has yolo-seg polygon labels but you are training `task=detect`.** Ultralytics
  derives boxes from the polygons — confirm the derived boxes are sane (dump a few training
  batch images and look at them) before spending GPU hours on it.
- class counts match the yaml (24 / 20 / 9)
- no missing-label or corrupt-image warnings

Report what you saw. If anything looks wrong, stop and say so — do not work around it.

### Step 1 — baselines

One run per dataset, identical settings, fixed seed:

| setting | value |
|---|---|
| weights | `yolo26s.pt` (absolute path — see gotcha 1) |
| epochs | 100 |
| imgsz | 640 |
| seed | 0 |
| task | detect |
| project | `runs/anomaly_bench` |
| name | `<dataset>_baseline_s0` |

Report `mAP50` and `mAP50-95` on **val** and on **test** for each.

### Step 2 — noise floor

Repeat the **`dspcbsd`** baseline with `seed=1` and `seed=2` (smallest dataset, so this is
cheap). Three seeds total including step 1.

Report the **spread** (max − min) of mAP50-95 across the three seeds. That number is the
noise floor: any later change smaller than it is not a result. State it as a single
explicit number — everyone downstream will quote it.

### Assumptions you may correct with Louis

My choices, not requirements — ask if you think they are wrong:

- **`yolo26s.pt` rather than `l`.** Many loss ablations will follow, so iteration speed
  matters more than headline numbers. Headline numbers on `l` would be a second pass.
- **100 epochs.** Not tuned. If the loss curves are clearly still improving at 100, say so
  rather than silently extending.
- **Noise floor on `dspcbsd` only.** Cheapest of the three. If the spread comes out large,
  propose repeating on `3cad` too.

---

## How to report

Everything goes in **`run.md` at this repo root** — one logbook for this branch. Per
experiment:

1. The **verbatim command**, copy-pasteable, in the same section as its results.
2. **Full per-class tables**, not summaries. Per-class AP costs GPU hours to regenerate; a
   prose summary destroys the shape. Compress your interpretation instead, not the numbers.
3. **Identity binding** — the `--snap` hash12, weight path, and dataset yaml for every
   number. Record the hash explicitly even though the snapshot dir encodes it, so the
   logbook still stands once `gcsnap` has reclaimed the snapshot.
4. Raw log on disk at `runs/anomaly_bench/<name>/run.log`, alongside the CSVs. Never leave
   the only copy in `/tmp` or in chat.

All metrics rounded to **4 decimal places**.

---

## Reading the results — do not skip this

- **`3cad` classes are long-tailed.** `bump` alone is 9,064 of 18,134 boxes (50%). Nine
  classes have under 70 boxes: `ink` 6, `fracture` 9, `wear_crack` 12, `fluff` 25,
  `pinhole` 33, `abrasion` 58, `bright_shadow` 61, `knife_mark` 67. **Per-class AP on
  those is noise.** Report it, but do not interpret it.
- **Report per dataset, never a single average.** The three deliberately stress different
  things; averaging destroys exactly the signal the benchmark was built to produce.
- **`tianchifabirc` and `dspcbsd` have no normal images** and cannot measure false
  positives. Only `3cad` can.
- **`3cad` is yolo-seg; the other two are detect-only.** Segmentation experiments can only
  run on `3cad`.
- **Attribution needs a controlled comparison.** Before claiming a change caused a delta,
  confirm the two runs differ in that one thing alone, and state whether the delta clears
  the step-2 noise floor. If it doesn't, the honest report is "no measurable difference".

---

## Rules

- **Never launch with uncommitted changes.** `bundle` enforces it, but the reason matters:
  an uncommitted change is simply absent from the snapshot, so the run would silently
  execute different code than you think.
- Commit in small units; every commit passes a smoke test.
- All outputs under `runs/` (already gitignored). Never commit weights or logs.
- Temporary scripts go in `runs/tests/`, not the repo root.
- Code, commits, comments, docs: **English**.
- Don't add files to `ultralytics/` for this task. Keep your footprint to `run.md` and, if
  you need scripts, `scripts/anomaly_bench/`.

---

## Rebuilding the benchmark (reference — you should not need this)

```bash
ssh ultra6
cd /data/shared-datasets/louis_data/AnomalyDataset
python3 /data/shared-datasets/louis_data/anomaly_bench/leak_audit.py   # leakage audit
python3 /data/shared-datasets/louis_data/anomaly_bench/bbox_stats.py   # bbox geometry
python3 /data/shared-datasets/louis_data/anomaly_bench/merge_3cad.py --dry-run
```

`leak_audit.py` and `bbox_stats.py` use paths relative to `AnomalyDataset/`, so run them
from there. `merge_3cad.py` uses absolute paths. Merge log: `anomaly_bench/merge_3cad.log`.
Those scripts are version-controlled in the `anomaly_data` repo under `scripts/`.
