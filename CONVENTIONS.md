# CONVENTIONS.md — naming and reporting rules for the defect benchmark

The one file that defines how experiments are named, identified, compared, and reported on branch
`yolo26-defect-bench`. The CURRENT STATE section of [run.md](run.md) is the truth source for results;
this file is the truth source for names and tables. Rules apply to NEW runs and NEW tables; historical
entries keep their original names and are mapped below.

## 1. Datasets — fixed short IDs

| short ID  | full / historical name          | note                   |
| --------- | ------------------------------- | ---------------------- |
| `3cad`    | `3cad`                          |                        |
| `tianchi` | `tianchifabirc` (old run names) | new runs use `tianchi` |
| `dspcbsd` | `dspcbsd`                       |                        |
| `coco`    | `coco`                          | generalisation check   |

Historical run names are never renamed; only new runs use the short ID.

## 2. Metric convention — one iron rule

- **Best epoch is chosen by `metrics/mAP50-95(B)`; `AP_small` is read from the SAME row in the
  `(B-coco)` columns.** This reproduces every published baseline number in run.md.
- Every result reports **Δ (delta vs the 3-seed baseline mean), Δ×floor, and p** together. A result
  without p is a screening observation, not a conclusion.
- `n < 3` is always marked `?` and never enters a "confirmed" status.
- Noise floors (2sd, 3 seeds), fixed for the benchmark:

| dataset   | mAP50-95 floor | AP_small floor |
| --------- | -------------- | -------------- |
| `3cad`    | 0.0042         | 0.0187         |
| `tianchi` | 0.0048         | 0.0084         |
| `dspcbsd` | 0.0067         | **0.0016**     |

- **dspcbsd special rule:** its AP_small floor is tiny, so the ×floor column overstates. Always read
  the p column for dspcbsd AP_small, never ×floor alone. (Also: dspcbsd images are 226×226, rescaled
  2.83× to reach 640, so `dspcbsd AP_small` is not a small-object measurement — see run.md.)

## 3. Change (knob) registry — the only allowed tokens in run names

A run name is `<dataset>__<treatment>__s<seed>` where `<treatment>` is one or more knob codes joined
by `+`. Architecture switches are a prefix on the first code. The registry is closed: a new knob gets
a code here FIRST, then runs may use it.

| code                                 | meaning                                                               | example                |
| ------------------------------------ | --------------------------------------------------------------------- | ---------------------- |
| `base`                               | baseline, no knobs                                                    | `3cad__base__s0`       |
| `k6`                                 | `model=yolo26n-k6.yaml` + donor `pretrained=.../yolo26n-k6.pt`        | `3cad__k6__s0`         |
| `ms16` / `ms32` / `ms8` / `ms24`     | `tal_min_side=<px>`                                                   | `3cad__ms16__s0`       |
| `rf1` / `rf4`                        | `tal_prior=rfla` + `tal_rf_scale=<n>`                                 | `tianchi__rf1__s0`     |
| `fill`                               | `tal_prior=rfla_fill`                                                 | `3cad__fill__s0`       |
| `arf4` / `arf2` / `arf8`             | `tal_prior=ar_rfla` + `tal_ar_rfla=<n>`                               | `tianchi__arf4__s0`    |
| `gm`                                 | `tal_prior=geom_topk`                                                 | `3cad__gm__s0`         |
| `ifx`                                | `tal_prior=inside_fix`                                                | `tianchi__ifx__s0`     |
| `nwd` / `nwdg2` / `nwdb1`            | `tal_metric=nwd` (+ `tal_nwd_gamma=2` / `tal_beta=1`)                 | `3cad__nwdb1__s0`      |
| `la`                                 | `tal_prior=level_assign`                                              | `3cad__la__s0`         |
| `si`                                 | `tal_score_inflate=True` (1A: slivers scored vs their surrogate box)  | `tianchi__la+si__s0`   |
| `s1_16` / `s1_32`                    | `tal_sliver_side=<px>` (S1, closed line)                              | `tianchi__s1_32__s0`   |
| `s2`                                 | `tal_sliver_floor=long` (S2, closed line)                             | —                      |
| `o2m` / `o2o`                        | head-scope suffix: `tal_heads=<head>` (only that head gets the knobs) | `tianchi__rf1_o2m__s0` |
| `y11`                                | architecture prefix: `model=yolo11n.pt`                               | `3cad__y11base__s0`    |
| `topk22` / `ir08` / `clspw` / `dfl0` | historical Z/M-wave knobs                                             | (read-only)            |

Combinations join with `+`: `k6+ms16`, `arf4+ms16`, `y11arf4+ms16`.

### Historical name map (for reading old run.md entries, never for new runs)

| old name                             | new spelling             |
| ------------------------------------ | ------------------------ |
| `3cad_a1_ms16_n_s0`                  | `3cad__ms16__s0`         |
| `3cad_a4_k6ms16_n_s0`                | `3cad__k6+ms16__s0`      |
| `tianchifabirc_a7_ms16_arrfla4_n_s0` | `tianchi__arf4+ms16__s0` |
| `3cad_z3_k6_n_s0`                    | `3cad__k6__s0`           |

## 4. Launch identification — project and name correspondence

Every launch line carries BOTH identifiers, and they must stay consistent:

- `project=yolo26-defect-bench` is fixed for this branch, every run, no exceptions.
- `name=<dataset>__<treatment>__s<seed>` as defined above.
- The expman record, the run dir, the `<name>.log`, and the run.md table row all use the same string.
- `--snap` is mandatory on every launch; the snap commit is recorded in the run.md entry with the run.

```bash
EXP=/Users/louis/workspace/ultra_louis_work/expman/.venv/bin/expman-cli
D=/data/shared-datasets/louis_data/anomaly_bench
B="model=yolo26n.pt epochs=100 imgsz=640 batch=128 coco_eval=True project=yolo26-defect-bench"
$EXP launch --snap --args "nohupyolo 0 train data=$D/3cad/data.yaml $B seed=0 tal_min_side=16 device=4 name=3cad__ms16__s0"
```

## 5. Comparison table — one canonical 9-column format, one table per dataset

```
| baseline + | n | mAP50-95 | Δ×floor | p | AP_small | Δ×floor | p | cost | status |
```

- First column literally reads `baseline + <treatment>` — the table answers "what did we change on top
  of baseline" at a glance. The baseline row itself is `— (baseline)`.
- `Δ×floor` and `p` are a pair; never one without the other.
- `cost` is stated when non-zero (`1.35× FLOPs` for k6; `train ~12% slower` for rf-family), else `0`.
- `status` takes exactly one of: `confirmed` / `screening` / `closed` / `collapsed` / `in flight`.
- `collapsed` rows put the collapse epoch in the mAP column (`collapsed ep9`).
- One table per dataset; never pool datasets in one table (floors and baselines differ per dataset).

Worked example (3cad, live numbers from run.md CURRENT STATE):

| baseline +   | n   | mAP50-95      | Δ×floor | p     | AP_small | Δ×floor | p     | cost        | status    |
| ------------ | --- | ------------- | ------- | ----- | -------- | ------- | ----- | ----------- | --------- |
| — (baseline) | 3   | 0.2908        | —       | —     | 0.1925   | —       | —     | 0           | —         |
| ms16         | 3   | 0.2972        | +1.51   | 0.048 | 0.2373   | +2.39   | 0.006 | 0           | confirmed |
| ms32         | 3   | 0.3035        | +3.01   | 0.031 | 0.2281   | +1.90   | 0.008 | 0           | confirmed |
| k6           | 3   | 0.3064        | +3.71   | 0.049 | 0.2393   | +2.50   | 0.044 | 1.35× FLOPs | confirmed |
| k6+ms16      | 3   | 0.3020        | +2.66   | 0.064 | 0.2076   | +0.81   | 0.098 | 1.35×       | closed    |
| rf1          | 1?  | collapsed ep9 |         |       |          |         |       | 0           | collapsed |
| fill         | 3   | 0.2934        | +0.60   | 0.578 | 0.1931   | +0.03   | 0.936 | 0           | closed    |
| la           | 1?  | in flight     |         |       |          |         |       | 0           | in flight |

## 6. Wording rules for prose (so history stops contradicting itself)

- A knob arm is named by its registry code (`ms16`), never by a wave letter. Wave letters (Z, M, A, B)
  are historical only; they describe when things were launched, not what they were.
- "Confirmed" is reserved for `n=3` AND `p<0.05` on at least one metric.
- A collapsed arm on 3cad is reported as **"unmeasured on 3cad"**, not "failed on 3cad" — the arm
  never trained. "Failed" is reserved for arms that trained and underperformed.
- When a mechanism claim is retracted, the retraction says which script/measurement was wrong; the
  corrigendum ledger in run.md CURRENT STATE is the running list. Fix the ledger, don't silently edit
  the old section.
