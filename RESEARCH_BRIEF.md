# Research brief — beat `k=6` on small-defect detection in YOLO26

**Audience:** an agent doing a literature survey. You have no access to the conversation that produced this
file. Everything you need is here; the raw logbook is [run.md](run.md) and the assignment is
[TASK.md](TASK.md).

**The ask.** We have benchmarked nine candidate modifications to YOLO26 on three industrial defect datasets.
Exactly one survived: widening two early downsampling kernels from 3×3 to 6×6 (`k=6`). Find published work
that should beat it, or that addresses a bottleneck it does not touch. Return a ranked, falsifiable shortlist
mapped onto this codebase — not a reading list.

**Why this is not a generic "small object detection" survey.** Most of that literature was written against
anchor-based, NMS-using, DFL-having detectors. YOLO26 is none of those. Half the standard toolbox is either
already present, provably identical to `k=6`, or structurally illegal here. Sections 3 and 4 tell you which,
with measurements. Proposing something from that list is the main failure mode — read them before searching.

---

## 1. The model you must fit into

`yolo26n`, from `ultralytics/cfg/models/26/yolo26.yaml`. Non-obvious properties, all verified in code:

- **`reg_max=1`, so there is no DFL.** Box supervision is **CIoU + an L1 term on imgsz-normalized ltrb**, not
  CIoU alone. The arg still named `dfl` scales that L1 term. Training logs print `train/l1_loss`, never
  `train/dfl_loss`. Anything that "replaces DFL" has nothing to replace.
- **`end2end: True`, a dual head.** `E2ELoss` in `ultralytics/utils/loss.py` builds two assigners:
  - `one2many` — `tal_topk=10`. Trains the shared backbone/neck. Discarded at inference.
  - `one2one` — `tal_topk=7`, then `topk2=1`. **This is the inference head and it runs with no NMS.**
- **Detection levels are P3/8, P4/16, P5/32.** Backbone stride-2 convs sit at layers 0 (P1/2), 1 (P2/4),
  3 (P3/8), 5 (P4/16), 7 (P5/32). Only layers 0, 1, 3 are upstream of P3, so only they can affect what the
  finest detection level ever sees.
- **The assigner requires the anchor centre to fall inside the GT box**
  (`TaskAlignedAssigner.select_candidates_in_gts`, `ultralytics/utils/tal.py`), with a clamp that inflates
  any GT side below the smallest stride up to that stride. `topk` therefore only binds when more than `topk`
  anchors are eligible at all. This matters enormously — see §5.
- Must survive ONNX/TensorRT export. Extra op count and dynamic shapes are real costs, not footnotes.

**Fixed experimental protocol** (do not propose changing it; comparability across ~50 runs depends on it):
`yolo26n`, 100 epochs, `imgsz=640`, `batch=128`, `coco_eval=True`, 3 seeds minimum per arm.

---

## 2. The benchmark, characterized

Three datasets, chosen after a leakage audit (see TASK.md). What matters for your search is that they sit in
**three different regimes**, and a change that helps one may be invisible or harmful in another.

Measured with `scripts/anomaly_bench/anchor_pool.py` over the COCO GT that `coco_eval` writes. "Pool" is the
number of anchors whose centre falls inside a GT box across P3+P4+P5 at `imgsz=640`; "starved" means the pool
is smaller than the o2m head's `topk=10`.

| dataset       | cls | GT boxes | median pool | starved | pool ≤ 1 | AR median | AR p90 | side < 6px @640 |
| ------------- | --- | -------- | ----------- | ------- | -------- | --------- | ------ | --------------- |
| dspcbsd       | 9   | 3,167    | 55          | 6.7%    | 0.1%     | 1.25      | 2.50   | 0.0%            |
| 3cad          | 24  | 1,801    | 7           | 55.0%   | 19.1%    | 1.90      | 6.95   | 8.7%            |
| tianchifabirc | 20  | 1,973    | 6           | 55.5%   | 11.5%    | 5.16      | 52.83  | 31.1%           |

- **dspcbsd** (PCB): compact defects, healthy anchor pool, and **near saturation** — baseline AP_small is
  0.3985 and nothing we tested moved it by more than 0.01. A change that only works here is not interesting.
- **3cad** (aluminium/3C parts, 58% of images are defect-free): **over half the objects cannot supply even 10
  anchors, and 19% get at most one.** This is where every real effect showed up.
- **tianchifabirc** (fabric): equally starved, but the defects are **slivers** — 31% have a side thinner than
  6 px at 640. Highest variance, hardest to move.

**Noise floors (2sd across 3 baseline seeds).** These are the significance bar.

| dataset       | mAP50-95 | AP_small   | baseline mAP50-95 | baseline AP_small |
| ------------- | -------- | ---------- | ----------------- | ----------------- |
| dspcbsd       | 0.0067   | **0.0016** | 0.4765            | 0.3985            |
| 3cad          | 0.0042   | **0.0187** | 0.2908            | 0.1925            |
| tianchifabirc | 0.0048   | 0.0084     | 0.1908            | 0.1220            |

**A floor is per-dataset AND per-metric, and borrowing one is wrong in both directions.** 3cad's AP_small
floor is 11.7× dspcbsd's, while its mAP50-95 floor is 0.6× — tighter. We made this mistake once and it
inverted three conclusions. `n=1` is not evidence: a single 3cad AP_small draw carries ±0.019.

---

## 3. The incumbent: `k=6`

Two-line diff from `yolo26.yaml` (`ultralytics/cfg/models/26/yolo26-k6.yaml`):

```yaml
- [-1, 1, Conv, [128, 6, 2, 2]] # 1-P2/4   was [128, 3, 2]
- [-1, 1, Conv, [256, 6, 2, 2]] # 3-P3/8   was [256, 3, 2]
```

Kernel 3→6 (taps 9→36) on the two downsampling convs **between the stem and P3**. Stride, output size,
channel count and topology are unchanged; `p=2` is mandatory because autopad would give 3 for k=6 and change
the output size. Layer 0 is deliberately untouched — it is the only conv that sees raw pixels.

**Mechanism.** At stride 2 with k=3, adjacent output positions read almost disjoint input windows, so
information between them is skipped. k=6 makes the windows overlap by half. Nothing between the stem and P3
is stepped over.

**Pretrained transfer is not a confound.** `scripts/anomaly_bench/make_k6_donor.py` embeds the pretrained 3×3
at `[1:4, 1:4]` of each 6×6 window and refuses to write if any tensor would stay random; both arms report
`Transferred 606/708`.

**Cost**, measured with `torch.utils.flop_counter.FlopCounterMode` at batch 1:

| config            | params            | GFLOPs | vs baseline |
| ----------------- | ----------------- | ------ | ----------- |
| `yolo26n` @640    | 2,572,280         | 3.04   | 1.00×       |
| `yolo26n-k6` @640 | 2,696,696 (+4.8%) | 4.10   | **1.35×**   |
| `yolo26n` @960    | 2,572,280 (+0%)   | 7.01   | 2.31×       |

`k=6` is a **weight** change — same activations, so unchanged memory (~25 GiB at batch=128). `imgsz=960` is
an **activation** change and needs 47.8 GiB.

**Results**, 3 seeds per arm, Welch two-sample t-test against the 3-seed baseline:

| dataset       | metric   | Δ       | t     | p          |
| ------------- | -------- | ------- | ----- | ---------- |
| 3cad          | mAP50-95 | +0.0156 | +3.77 | **0.0488** |
| 3cad          | AP_small | +0.0468 | +3.52 | **0.0444** |
| dspcbsd       | AP_small | +0.0041 | +2.58 | 0.1049     |
| dspcbsd       | mAP50-95 | +0.0032 | +1.43 | 0.2434     |
| tianchifabirc | AP_small | +0.0071 | +0.88 | 0.4601     |
| tianchifabirc | mAP50-95 | +0.0034 | +1.31 | 0.2736     |

**Positive in 6/6 cells; only 3cad clears 95%.** The load-bearing evidence is 3cad; the other two are
same-direction but underpowered. `k=6` also inflates seed variance (3cad AP_small sd 0.0211 vs the baseline's
0.0093), which is why the t-tests fall short — the effect is not small, the spread is large.

**The bar you must beat.** Reference point on 3cad: `imgsz=960` alone reaches AP_small 0.2827 vs `k=6`'s
0.2393, but at 2.31× FLOPs and ~2× memory. Per extra GFLOP, `k=6` wins ~2× in all four measured cells. So:

> **A candidate is interesting if it beats `+0.0468` AP_small on 3cad at ≤ 1.35× baseline FLOPs and no extra
> activation memory — or if it targets bottleneck (b) in §5, which `k=6` cannot touch.**

---

## 4. Already falsified — do not propose these

Each of these was tested or proved in this repo. If a paper you find reduces to one of them, say so and move
on; do not re-recommend it.

| candidate                                        | verdict                                                                                                                                                                                                                                                                                                                                                                                        |
| ------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **SPD-Conv / space-to-depth**                    | **Provably identical to `Conv(k=6,s=2)`** — verified in float64, max abs diff < 1e-12, same parameter count (`scripts/anomaly_bench/spd_equivalence.py`). Not an alternative to `k=6`, the same operator with worse plumbing: +12.6% CPU latency at batch 1 and 16 extra ONNX `Slice` nodes. Any SPD-Conv paper is already covered.                                                            |
| **`Focus` (YOLOv5 stem)**                        | Same operator class; dropped upstream in favour of a strided conv for exactly the plumbing reason above.                                                                                                                                                                                                                                                                                       |
| **NWD (Normalized Wasserstein)**                 | Closed. CIoU already carries a normalized centre-distance (`rho2/c2`) and an aspect term (`v`), and NWD's `W₂²` is a direct-coordinate L2 that overlaps the **existing L1-on-ltrb** term. Its gate experiment (`dfl=0`) measured as noise.                                                                                                                                                     |
| **Inner-IoU** (arXiv:2311.02877)                 | Tested at ratio 0.7 / 0.8 / 1.2. No-op on mAP50-95 (all inside the floor); ratio < 1 costs 4.4–8.0× the dspcbsd AP_small floor. Its mechanism is convergence speed, and at 100 epochs there is no gap left to close.                                                                                                                                                                           |
| **More positives on the o2o head** (`topk2 > 1`) | **Structurally invalid, not merely bad.** Precision 0.7995 → 0.5482 → 0.4608 → 0.4536 for topk2 = 1,2,3,4 while recall barely moves. `topk2=2` emits **3.4× the boxes** for the same 3,167 objects, and post-hoc NMS recovers **96%** of the lost mAP50 (`scripts/anomaly_bench/nms_probe.py`). The o2o head has no NMS: k positives per GT means k boxes per object. `topk2=1` is a contract. |
| **Raising o2m `tal_topk` above 10**              | Near-no-op by measurement: dspcbsd's pool is already 55 (`topk` is not the constraint), and on 3cad it cannot reach the 55% of boxes that cannot supply 10 anchors in the first place.                                                                                                                                                                                                         |
| **`cls_pw`** (inverse-frequency class weights)   | Fails with a monotone dose-response in the wrong direction: 3cad mAP50-95 −4.50× floor at 0.5, −14.45× at 1.0.                                                                                                                                                                                                                                                                                 |
| **`mosaic=0`**                                   | Fails on all three datasets (mAP50-95 −1.8× to −6.6× floor). The hypothesis was that mosaic's downscaling destroys tiny defects; it is wrong.                                                                                                                                                                                                                                                  |
| **Aspect-ratio-aware stem** (`k=(6,3)`)          | Not falsified but **de-motivated**: it existed to explain `k=6` allegedly hurting tianchifabirc, and at 3 seeds that −0.0065 flipped to **+0.0071**. The AR measurements in §2 stand; the causal story did not. Only revive it with a new argument.                                                                                                                                            |

**Pattern worth internalizing: every loss-side knob we tried came back neutral-or-harmful, and it is not a
coincidence.** On 3cad and tianchifabirc, 55% of objects cannot fill the assigner's candidate list. You
cannot reweight positives that do not exist. Loss-reweighting proposals should be treated as low prior here
unless they explain how they reach starved objects.

---

## 5. The two bottlenecks — keep them separate

Conflating these is the second failure mode.

**(a) Information destroyed before P3.** Three stride-2 steps happen upstream of the finest detection level,
and with k=3 each one skips pixels. This is what `k=6` fixes — cheaply, on the weight side. Open question:
is there a better operator than a wide strided conv for lossless-ish downsampling at this budget?

**(b) Anchor starvation.** The assigner can only pick anchors whose centre is inside the GT box, so a small
object has a hard cap on positives regardless of any `topk`. No weight-side change can lift it — only finer
stride or more pixels.

| dataset | levels    | imgsz | median pool | starved | pool ≤ 1 |
| ------- | --------- | ----- | ----------- | ------- | -------- |
| 3cad    | P3–P5     | 640   | 7           | 55.0%   | 19.1%    |
| 3cad    | P3–P5     | 960   | 17          | 36.0%   | 9.4%     |
| 3cad    | P3–P5     | 1280  | 35          | 21.6%   | 3.7%     |
| 3cad    | **P2**–P5 | 640   | 35          | 21.6%   | 3.7%     |
| dspcbsd | P3–P5     | 640   | 55          | 6.7%    | 0.1%     |
| dspcbsd | **P2**–P5 | 640   | 254         | 0.1%    | 0.0%     |

**A P2 head at `imgsz=640` matches `imgsz=1280` on starvation at a fraction of the cost.** `yolo26-p2.yaml`
already exists and is untested here — it is the obvious next move, which is precisely why we want to know
whether the literature has something better than a plain extra P2 level (the usual objections: P2 is
expensive in activations, and it is the noisiest level).

Both bottlenecks are real, and the data says so: `k=6` clears p<0.05 on 3cad on both metrics — so (a) matters
— yet pure resolution buys roughly twice the absolute AP_small there — so (b) matters more. A candidate that
addresses (b) at weight-side cost would be the most valuable result you can bring back.

---

## 6. What to return

A ranked shortlist. For **each** candidate, all seven fields — a candidate missing the mount point or the
prediction is not actionable and will be dropped:

1. **Paper** — title, venue, year, link. Prefer work with released code.
2. **Mechanism**, one paragraph, in terms of §5(a) or §5(b). Say which.
3. **Mount point in this repo** — file and class, or the exact yaml layer index. E.g. "replaces layers 1 and
   3 of the backbone", "wraps `TaskAlignedAssigner.select_candidates_in_gts`", "new module in
   `ultralytics/nn/modules/block.py` referenced by name from a yaml".
4. **Cost estimate** — Δparams, ΔGFLOPs at 640, and whether it changes **activation** memory (that is the
   axis that forced `960` onto its own card).
5. **Export risk** — op count added, any dynamic shape, any op ONNX/TensorRT handles badly.
6. **Compatibility check against §1** — in particular: does it assume DFL, anchors, or NMS? If it adds
   predictions per object, it is dead on arrival.
7. **Falsifiable prediction** — which dataset × metric cell it should move and roughly how much, against the
   floors in §2. "Improves small object detection" is not a prediction; "3cad AP_small +0.03 or more, no
   change on dspcbsd" is.

**Rank by expected AP gain per extra GFLOP**, since that is the axis on which `k=6` beats brute-force
resolution ~2×.

**Reject on sight**, and say which rule fired:

- mathematically identical to `Conv(k=6,s=2)` (§4);
- adds positives or predictions to the one-to-one head, or needs NMS at inference (§1, §4);
- only reshapes the box-regression loss (§4 — CIoU+L1 already covers centre distance, aspect, and scale);
- assumes DFL, an anchor-based head, or a separate objectness branch;
- can only be validated on a dataset outside this benchmark.

**Also welcome, if you find it:** evidence that we are wrong about something in §4. Every entry there is a
measurement or a proof, and each cites the script that produced it — check them rather than trusting them.

---

## 7. Reproduction pointers

| what                             | where                                                                                        |
| -------------------------------- | -------------------------------------------------------------------------------------------- |
| Full logbook, every run verbatim | [run.md](run.md)                                                                             |
| Assignment, dataset provenance   | [TASK.md](TASK.md)                                                                           |
| `k=6` model                      | `ultralytics/cfg/models/26/yolo26-k6.yaml`                                                   |
| `k=6` pretrained donor builder   | `scripts/anomaly_bench/make_k6_donor.py`                                                     |
| SPD-Conv ≡ `Conv(k=6,s=2)` proof | `scripts/anomaly_bench/spd_equivalence.py`                                                   |
| o2o duplicate-box proof          | `scripts/anomaly_bench/nms_probe.py`                                                         |
| Anchor-pool / AR measurement     | `scripts/anomaly_bench/anchor_pool.py`                                                       |
| Loss and assigner                | `ultralytics/utils/loss.py` (`E2ELoss`, `BboxLoss`), `ultralytics/utils/tal.py`              |
| Extra AP columns                 | `coco_eval` in `ultralytics/cfg/default.yaml`, default off, purely additive to `results.csv` |

Open at the time of writing: `imgsz=960` seed replicates on 3cad and tianchifabirc, and a second
`k=6`+`960` combination seed. The single-seed `960` numbers in §3 are marked as such and must not be
treated as settled.
