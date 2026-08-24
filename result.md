# result.md — confirmed and screening gains, per dataset

精选视图 of run.md's CURRENT STATE：only arms with a positive Δ on either metric. 主表 8 列 + 逐 seed 明细子表（mAP50-95 / AP_small，best epoch 同口径）。Rules and knob meanings in [CONVENTIONS.md](CONVENTIONS.md). 26s line in flight — its screening rows firm up or drop when seeds land.

## 3cad

**26n**（baseline n=3：mAP50-95 0.2908 / AP_small 0.1925；floor 0.0042 / 0.0187）

| baseline +   | n   | mAP50-95 | Δ×floor | AP_small | Δ×floor | cost        | status    |
| ------------ | --- | -------- | ------- | -------- | ------- | ----------- | --------- |
| — (baseline) | 3   | 0.2908   | —       | 0.1925   | —       | 0           | —         |
| ms16         | 3   | 0.2972   | +1.51   | 0.2373   | +2.39   | 0           | confirmed |
| ms32         | 3   | 0.3035   | +3.01   | 0.2281   | +1.90   | 0           | confirmed |
| k6           | 3   | 0.3065   | +3.72   | 0.2394   | +2.51   | 1.35× FLOPs | confirmed |
| k6+la        | 3   | 0.3040   | +3.14   | 0.2206   | +1.50   | 1.35× FLOPs | confirmed |
| la           | 1?  | 0.3051   | +3.39   | 0.2126   | +1.08   | 0           | screening |
| s2           | 1?  | 0.2966   | +1.36   | 0.2233   | +1.64   | 0           | screening |

逐 seed（mAP50-95 / AP_small；s0/s1/s2 = seed 0/1/2，唯一例外 `3cad__si__s1` 实为 seed 1）：

| arm      | s0              | s1              | s2              |
| -------- | --------------- | --------------- | --------------- |
| baseline | 0.2900 / 0.1882 | 0.2932 / 0.2032 | 0.2893 / 0.1861 |
| ms16     | 0.2987 / 0.2270 | 0.2937 / 0.2360 | 0.2992 / 0.2488 |
| ms32     | 0.2995 / 0.2331 | 0.3089 / 0.2297 | 0.3021 / 0.2216 |
| k6       | 0.3050 / 0.2186 | 0.3004 / 0.2387 | 0.3139 / 0.2607 |
| k6+la    | 0.3005 / 0.2112 | 0.3110 / 0.2248 | 0.3004 / 0.2260 |
| la       | 0.3051 / 0.2126 | —               | —               |
| s2       | 0.2966 / 0.2233 | —               | —               |

**26s**（baseline n=3：mAP50-95 0.2895 / AP_small 0.2341；floor 0.0477 / 0.0271）

| baseline +   | n   | mAP50-95 | Δ×floor | AP_small | Δ×floor | cost        | status    |
| ------------ | --- | -------- | ------- | -------- | ------- | ----------- | --------- |
| — (baseline) | 3   | 0.2895   | —       | 0.2341   | —       | 0           | —         |
| k6+la        | 1?  | 0.3081   | +0.39   | 0.2446   | +0.39   | 1.35× FLOPs | screening |

逐 seed（mAP50-95 / AP_small；s0/s1/s2 = seed 0/1/2，唯一例外 `3cad__si__s1` 实为 seed 1）：

| arm      | s0              | s1              | s2              |
| -------- | --------------- | --------------- | --------------- |
| baseline | 0.2753 / 0.2252 | 0.2762 / 0.2273 | 0.3171 / 0.2497 |
| k6+la    | 0.3081 / 0.2446 | —               | —               |

## tianchi

**26n**（baseline n=3：mAP50-95 0.1909 / AP_small 0.1220；floor 0.0048 / 0.0084）

| baseline +   | n   | mAP50-95 | Δ×floor | AP_small | Δ×floor | cost                 | status    |
| ------------ | --- | -------- | ------- | -------- | ------- | -------------------- | --------- |
| — (baseline) | 3   | 0.1909   | —       | 0.1220   | —       | 0                    | —         |
| rf1          | 3   | 0.2158   | +5.19   | 0.1335   | +1.36   | 0                    | confirmed |
| arf4+ms16    | 3   | 0.2161   | +5.26   | 0.1275   | +0.65   | 0, train ~12% slower | confirmed |
| arf4         | 1?  | 0.2190   | +5.85   | 0.1387   | +1.99   | 0                    | screening |
| k6+la        | 3   | 0.1981   | +1.51   | 0.1305   | +1.01   | 1.35× FLOPs          | confirmed |
| k6+rf1       | 3   | 0.2223   | +6.55   | 0.1289   | +0.81   | 1.35× FLOPs          | confirmed |

逐 seed（mAP50-95 / AP_small；s0/s1/s2 = seed 0/1/2，唯一例外 `3cad__si__s1` 实为 seed 1）：

| arm       | s0              | s1              | s2              |
| --------- | --------------- | --------------- | --------------- |
| baseline  | 0.1936 / 0.1255 | 0.1891 / 0.1173 | 0.1898 / 0.1233 |
| rf1       | 0.2211 / 0.1361 | 0.2159 / 0.1287 | 0.2103 / 0.1356 |
| arf4+ms16 | 0.2108 / 0.1263 | 0.2186 / 0.1293 | 0.2189 / 0.1268 |
| arf4      | 0.2190 / 0.1387 | —               | —               |
| k6+la     | 0.2008 / 0.1365 | 0.1948 / 0.1258 | 0.1988 / 0.1293 |
| k6+rf1    | 0.2245 / 0.1318 | 0.2229 / 0.1291 | 0.2195 / 0.1257 |

**26s**（baseline n=3 完成，floor 0.0038 / 0.0104）— no positive arm yet; seeds in flight.

## dspcbsd

**26n**（baseline n=3：mAP50-95 0.4765 / AP_small 0.3985；floor 0.0067 / 0.0016）

| baseline +   | n   | mAP50-95 | Δ×floor | AP_small | Δ×floor | cost        | status    |
| ------------ | --- | -------- | ------- | -------- | ------- | ----------- | --------- |
| — (baseline) | 3   | 0.4765   | —       | 0.3985   | —       | 0           | —         |
| k6+la        | 3   | 0.4790   | +0.38   | 0.4099   | +7.14   | 1.35× FLOPs | confirmed |

逐 seed（mAP50-95 / AP_small；s0/s1/s2 = seed 0/1/2，唯一例外 `3cad__si__s1` 实为 seed 1）：

| arm      | s0              | s1              | s2              |
| -------- | --------------- | --------------- | --------------- |
| baseline | 0.4737 / 0.3982 | 0.4802 / 0.3979 | 0.4756 / 0.3994 |
| k6+la    | 0.4796 / 0.4116 | 0.4747 / 0.4090 | 0.4827 / 0.4092 |

**26s**（baseline n=3 完成，floor 0.0066 / 0.0199）— no positive arm yet; seeds in flight.

## Knob legend — baseline behaviour vs the change

| code           | baseline 怎么做                                                                                                                 | 改动后                                                                                                                        |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| ms16 / ms32    | `select_candidates_in_gts` 的 legacy clamp：边 <8px 抬到 16px，[8,16) 原样不动 —— **非单调**（8.1px 的框比 7.9px 的框候选更少） | 短边统一 floor 到 16/32px（单调）；仍只改候选池，topk 排序还是模型自己的预测                                                  |
| k6             | 层 1/3 的下采样 Conv 是 k=3,s=2：相邻输出窗口几乎不重叠，窗口之间的像素被跳过                                                   | k=6,s=2（p=2 保输出尺寸）：窗口重叠一半，stem 到 P3 之间没有信息被跳过；1.35× FLOPs（n），越大模型占比越小                    |
| la             | 候选池 = 所有检测层上框内 anchor 全部收进来；topk 由模型 align_metric 排序                                                      | 按 GT **长边**定层（≥64→P5，32-64→P4，否则 P3），池 = 该层 + 相邻更细一层；短边膨胀到 2×层 stride（补边界 bug）；排序仍是模型 |
| rf1            | topk 排序完全由模型预测（align_metric = cls×CIoU）决定 —— 细条上 CIoU 退化、排序漂移                                            | 候选 = 几何固定的 top-10（anchor 责任区 vs GT 的中心距离+大小匹配），不看模型预测                                             |
| arf4           | （同 rf1 的 baseline 侧）所有 GT 一条规则                                                                                       | 只对 AR≥4 的长条 GT 走 rf1 几何固定；compact GT 保持 baseline 原样                                                            |
| arf4+ms16      | 同 baseline                                                                                                                     | arf4 的门控钉死 + ms16 的池膨胀，两者叠加                                                                                     |
| k6+rf1 / k6+la | 各自单臂的 baseline 行为                                                                                                        | 特征侧（k6）+ 分配侧（rf1/la）的组合                                                                                          |

| code             | what it does                                                                                                                                                                                                                                                                                                    |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ms16 / ms32      | `tal_min_side=<px>`: raise every GT's short side to this floor before the assigner's inside-GT test, so small defects recruit more candidate anchors. The topk ranking stays the model's own prediction.                                                                                                        |
| k6               | `yolo26n-k6.yaml` + donor: widen the two early downsampling convs from 3x3 to 6x6 (k=6, s=2), so nothing between the stem and P3 is stepped over. Costs 1.35x FLOPs on n, less on larger models.                                                                                                                |
| la               | `tal_prior=level_assign`: assign each GT a detection level by its LONG side (P5 if >=64px, P4 if 32-64px, else P3), pool = that level + the finer neighbour, short side inflated to 2x the level stride. Ranking stays the model's. Does NOT include ms16/ms32 — that floor only applies to the compact branch. |
| rf1              | `tal_prior=rfla`: replace the candidate pool with the top-10 anchors by receptive-field distance (centre distance + size match) — geometric pinning, no model ranking. All GTs.                                                                                                                                 |
| arf4 / arf4+ms16 | `tal_prior=ar_rfla` + `tal_ar_rfla=4`: the rf1 pinning applied ONLY to sliver GTs (aspect ratio >= 4); compact GTs keep the default pool. `+ms16` adds the ms16 floor on top.                                                                                                                                   |
| k6+rf1 / k6+la   | combinations: k6 backbone plus the assignment knob.                                                                                                                                                                                                                                                             |
