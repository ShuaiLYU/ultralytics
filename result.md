# result.md — confirmed and screening gains, per dataset

精选视图 of run.md's CURRENT STATE：only arms with a positive Δ on either metric. Rules and knob meanings in [CONVENTIONS.md](CONVENTIONS.md); floors per dataset per model size. 26s line is in flight — its screening rows will firm up or drop when its seeds land.

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

**26s**（baseline n=3：mAP50-95 0.2895 / AP_small 0.2341；floor 0.0477 / 0.0271）

| baseline +   | n   | mAP50-95 | Δ×floor | AP_small | Δ×floor | cost        | status    |
| ------------ | --- | -------- | ------- | -------- | ------- | ----------- | --------- |
| — (baseline) | 3   | 0.2895   | —       | 0.2341   | —       | 0           | —         |
| k6+la        | 1?  | 0.3081   | +0.39   | 0.2446   | +0.39   | 1.35× FLOPs | screening |

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

**26s**（baseline n=3 已跑完，floor 已建：mAP 0.0038 / AP_small 0.0104）— no positive arm yet; k6+la s1/s2 and attribution controls in flight.

## dspcbsd

**26n**（baseline n=3：mAP50-95 0.4765 / AP_small 0.3985；floor 0.0067 / 0.0016）

| baseline +   | n   | mAP50-95 | Δ×floor | AP_small | Δ×floor | cost        | status    |
| ------------ | --- | -------- | ------- | -------- | ------- | ----------- | --------- |
| — (baseline) | 3   | 0.4765   | —       | 0.3985   | —       | 0           | —         |
| k6+la        | 3   | 0.4790   | +0.38   | 0.4099   | +7.14   | 1.35× FLOPs | confirmed |

**26s**（baseline n=3 已跑完，floor 已建：mAP 0.0066 / AP_small 0.0199）— no positive arm yet; k6+la s1/s2 and attribution controls in flight.

## Knob legend

| code             | what it does                                                                                                                                                                                                                        |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ms16 / ms32      | `tal_min_side=<px>`: raise every GT's short side to this floor before the assigner's inside-GT test, so small defects recruit more candidate anchors. The topk ranking stays the model's own prediction.                            |
| k6               | `yolo26n-k6.yaml` + donor: widen the two early downsampling convs from 3x3 to 6x6 (k=6, s=2), so nothing between the stem and P3 is stepped over. Costs 1.35x FLOPs on n, less on larger models.                                    |
| la               | `tal_prior=level_assign`: assign each GT a detection level by its LONG side (P5 if >=64px, P4 if 32-64px, else P3), pool = that level + the finer neighbour, short side inflated to 2x the level stride. Ranking stays the model's. |
| rf1              | `tal_prior=rfla`: replace the candidate pool with the top-10 anchors by receptive-field distance (centre distance + size match) — geometric pinning, no model ranking. All GTs.                                                     |
| arf4 / arf4+ms16 | `tal_prior=ar_rfla` + `tal_ar_rfla=4`: the rf1 pinning applied ONLY to sliver GTs (aspect ratio >= 4); compact GTs keep the default pool. `+ms16` adds the ms16 floor on top.                                                       |
| k6+rf1 / k6+la   | combinations: k6 backbone plus the assignment knob.                                                                                                                                                                                 |
