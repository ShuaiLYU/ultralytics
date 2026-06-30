# YOLOA Documentation Plan

## Context

YOLOA (YOLO Anomaly) — training-free anomaly detection on YOLO checkpoints. Two workflows: `YOLOA` wrapper (fit→predict) and supervised training (`task="anomaly_v2"`).

## Completed

- [x] `docs/en/tasks/anomaly.md`
- [x] `docs/en/models/yoloa.md`
- [x] `mkdocs.yml` — Models + Tasks nav entries

---

## Remaining Work

### A. CODE FIXES (must do before or with docs)

**A1. CRITICAL: Default model YAML path broken (3 places)**

Three defaults all point to nonexistent files:
- `TASK2MODEL["anomaly_v2"]` in `cfg/__init__.py:79` → `"yolo26m-anomaly-v2.yaml"` (nonexistent)
- `YOLOA.__init__` in `yoloa.py:106` → `"yolo26m-anomaly-v2.yaml"`
- `YOLOAnomalyV2Model.__init__` in `nn/tasks.py:570` → `"yolo26-anomaly-v2.yaml"`

Real files are `yolo26-anomaly.yaml` and `yolo26-anomaly-seg.yaml`. Stem rewriting also fails. Fix: update all defaults to `yolo26-anomaly.yaml` (or create the missing file as an alias).

**A2. HIGH: `prior` vs `prior_mode` naming inconsistency**

`YOLOA.predict(prior="heatmap")` vs `YOLO(task="anomaly_v2").predict(prior_mode="heatmap")`. Unify to `prior` everywhere or document the difference clearly.

**A3. HIGH: Export is likely broken for anomaly_v2**

`Model.export()` inherited with no anomaly_v2 awareness — dynamic mask paths + memory bank k-NN will fail ONNX export. Options: (a) block with clear error, (b) implement fused inference-mode export path. Regardless, the export section in the task doc should note limitations.

**A4. HIGH: YOLOA save() may lose fit_args/fit_data**

`Model.save()` uses `self.ckpt` + model `state_dict`. `fit_args`/`fit_data` are set as plain attrs on `YOLOAnomalyV2Model` — verify they round-trip through `.pt` save/load. If not, override `save()` in YOLOA.

**A5. MEDIUM: No graceful error for nonexistent pretrained .pt**

Users who try `YOLO("yolo26m-anomaly.pt")` get confusing `FileNotFoundError`. Add a check: if anomaly task and .pt doesn't exist locally or in asset registry → clear error: "YOLOA models are not distributed as pretrained .pt files."

**A6. LOW: YOLOA not in top-level exports**

`from ultralytics import YOLOA` doesn't work. Not critical (users can `from ultralytics.yoloa import YOLOA`), but discoverability gap.

---

### B. DOCUMENTATION (cross-reference updates)

**B1. API Reference docs (P0)**
- Create: `reference/models/yolo/anomaly_v2/{predict,train,val}.md`
- Update: `mkdocs.yml` reference→yolo subsection

**B2. Cross-reference updates (P1)**

| File | Fix |
|------|-----|
| `usage/cfg.md` | Add `anomaly_v2` to TASK list (line 50, 68) |
| `models/yolo26.md` | Add anomaly row to Supported Tasks table; update FAQ |
| `models/index.md` | Add YOLOA to Featured Models list; fix "five tasks" |
| `tasks/index.md` | Add Anomaly Detection section + link |
| `modes/index.md` | Mention anomaly detection |
| `modes/train.md` | Mention `task="anomaly_v2"` workflow |
| `modes/predict.md` | Mention anomaly predict with memory bank/prior |
| `modes/val.md` | Mention anomaly OOD eval |
| `platform/train/cloud-training.md` | Fix "5 sizes × 5 tasks" |
| `help/FAQ.md` | Add anomaly FAQ entry |

**B3. Datasets docs (P2)**
- `datasets/anomaly/index.md` + `datasets/anomaly/mvtec-ad.md`
- Update `datasets/index.md`

**B4. Performance macro (P2 — blocked)**
- `macros/yolo-anomaly-perf.md` + include in task doc

---

### C. IMAGES (P3)

| Image | For | Status |
|-------|-----|--------|
| Banner (hero) | `tasks/anomaly.md` + `models/yoloa.md` | Need to prepare |
| Architecture diagram | `models/yoloa.md` | Need to prepare |
| Benchmark chart | TBD (placeholder) | Blocked on MVTec data |

Images → `cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/`

---

## Blockers
- B4 (perf macro) + benchmark chart → MVTec benchmark data
- B3 (datasets) → dataset download/path details from Louis
- C (images) → Louis to prepare
