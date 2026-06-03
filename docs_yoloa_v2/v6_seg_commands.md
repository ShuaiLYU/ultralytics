# Anomaly v2 Seg — ultra6 Training Commands

**Branch:** `yoloa_v2_seg`
**Spec:** `docs_yoloa_v2/specs/2026-06-03-anomaly-v2-seg-design.md`

## Preconditions
1. `conda activate ultra`
2. `set_wandb_true`
3. Branch `yoloa_v2_seg` checked out, clean
4. Softhint runs completed and false-prompt evaluated

## 1. Softhint-seg main on v6_binary

```bash
nohupyolo train task=anomaly_v2_seg \
  model=yolo26m-anomaly-v2-seg.yaml \
  pretrained=yolo26m.pt \
  data=/data/shared-datasets/louis_data/AnomalyDataset/merge_data_v6_binary/data.yaml \
  epochs=20 batch=96 close_mosaic=20 device=0,1,2 \
  optimizer=MuSGD lr0=0.00125 lrf=0.5 momentum=0.9 weight_decay=0.0005 \
  scale=0.1 copy_paste=0.1 mixup=0.0 save_json=True \
  project=yoloa_v2_seg name=26m_yoloav2seg_v6binary_pd50_v1
```

## 2. Vanilla yolo26m-seg baseline on v6_binary (for comparison)

```bash
nohupyolo train task=segment \
  model=yolo26m-seg.yaml \
  pretrained=yolo26m.pt \
  data=/data/shared-datasets/louis_data/AnomalyDataset/merge_data_v6_binary/data.yaml \
  epochs=20 batch=96 close_mosaic=20 device=3,4,5 \
  optimizer=MuSGD lr0=0.00125 lrf=0.5 momentum=0.9 weight_decay=0.0005 \
  scale=0.1 copy_paste=0.1 mixup=0.0 save_json=True \
  project=yoloa_v2_seg name=26m_seg_v6binary_v1
```

## 3. Pass criteria

- Softhint-seg mAP50-mask within 0.02 of vanilla `yolo26m-seg` baseline.
- β values at e20 are finite.
- Architecture non-destructive confirmed.

## 4. v6_multiclass (later)

Once v6_binary validated, swap `data=...v6_multiclass/data.yaml`.
