# 1-week full-scope ablation

Canonical plan. Work is GitHub issue [#1](https://github.com/DarkPhoenix645/groupim-lightgcn-ablation/issues/1) and its sub-issues. Blocked-by links on those issues are the live gate (ADR-0006).

**Hardware:** one RTX 3080 (10 GB). **Budget:** the ~450 runs / 80–120 GPU hours figure was sized for a 4080; remeasure on this card before locking the grid. 2–3 concurrent jobs. Queue what fits and leave it overnight.

**Reproduction:** [#3](https://github.com/DarkPhoenix645/groupim-lightgcn-ablation/issues/3) on the author Weeplaces files. That is the only Table 2 comparison (ADR-0008).

**Variants:** [#5](https://github.com/DarkPhoenix645/groupim-lightgcn-ablation/issues/5), [#6](https://github.com/DarkPhoenix645/groupim-lightgcn-ablation/issues/6), and [#7](https://github.com/DarkPhoenix645/groupim-lightgcn-ablation/issues/7) are reconstructed Yelp, Gowalla, and Douban. Counts: [table1-comparison.md](table1-comparison.md). They do not block [#3](https://github.com/DarkPhoenix645/groupim-lightgcn-ablation/issues/3).

**Gates**

- Weeplaces MLP attention, at least three seeds, before the LightGCN comparison is interpreted.
- Variant metrics are not reported as Table 2 reproductions.
- A small or null LightGCN gain is still a finding.

## Tickets

Sub-issues of [#1](https://github.com/DarkPhoenix645/groupim-lightgcn-ablation/issues/1). The blocked-by column mirrors the GitHub dependency, which is the one that gates the work.

| # | Title | Blocked by |
| --- | --- | --- |
| [2](https://github.com/DarkPhoenix645/groupim-lightgcn-ablation/issues/2) | Port CrowdDynamicsLab/GroupIM onto uv + Torch 2.x CUDA | — |
| [3](https://github.com/DarkPhoenix645/groupim-lightgcn-ablation/issues/3) | Audit Weeplaces and reproduce MLP GroupIM baseline | 2 |
| [4](https://github.com/DarkPhoenix645/groupim-lightgcn-ablation/issues/4) | Experiment launcher: config in, one CSV row out, artifacts saved | 2 |
| [5](https://github.com/DarkPhoenix645/groupim-lightgcn-ablation/issues/5) | Yelp reconstructed variant: Yin LA groups, record Table 1 gap | — |
| [6](https://github.com/DarkPhoenix645/groupim-lightgcn-ablation/issues/6) | Gowalla reconstructed variant: outing-as-group, record Table 1 gap | — |
| [7](https://github.com/DarkPhoenix645/groupim-lightgcn-ablation/issues/7) | Douban reconstructed variant: event items, record Table 1 gap | — |
| [8](https://github.com/DarkPhoenix645/groupim-lightgcn-ablation/issues/8) | LightGCN user encoder + Weeplaces smoke | 3 |
| [9](https://github.com/DarkPhoenix645/groupim-lightgcn-ablation/issues/9) | LightGCN user-loss pretraining analogue | 8 |
| [10](https://github.com/DarkPhoenix645/groupim-lightgcn-ablation/issues/10) | Ablation switches for every Table-3-style row | 8, 9 |
| [11](https://github.com/DarkPhoenix645/groupim-lightgcn-ablation/issues/11) | MLP + LightGCN smoke; freeze the grid from measured 3080 times | 3, 4, 8 |
| [12](https://github.com/DarkPhoenix645/groupim-lightgcn-ablation/issues/12) | Main grid: MLP vs LightGCN K=1,2,3 × pooling × 3 seeds × 4 datasets | 11 |
| [13](https://github.com/DarkPhoenix645/groupim-lightgcn-ablation/issues/13) | Table-3-style ablation for MLP and best LightGCN | 10, 11 |
| [14](https://github.com/DarkPhoenix645/groupim-lightgcn-ablation/issues/14) | LightGCN extras: pretrain on/off, dim {32,64,128}, tied item embeddings | 9, 11 |
| [15](https://github.com/DarkPhoenix645/groupim-lightgcn-ablation/issues/15) | λ sweep 2^-4..2^6 for MLP and best LightGCN | 10, 11 |
| [16](https://github.com/DarkPhoenix645/groupim-lightgcn-ablation/issues/16) | Extra seeds (5–10) for headline comparisons, leftover GPU | 12 |
| [17](https://github.com/DarkPhoenix645/groupim-lightgcn-ablation/issues/17) | Per-dataset results tables: mean±std and paired LightGCN vs MLP | 12–15 |
| [18](https://github.com/DarkPhoenix645/groupim-lightgcn-ablation/issues/18) | Group-size, coherence, and diversity quartile analysis | 4, 12 |
| [19](https://github.com/DarkPhoenix645/groupim-lightgcn-ablation/issues/19) | MI-variation and short-history hypothesis checks | 4, 12, 13 |
| [20](https://github.com/DarkPhoenix645/groupim-lightgcn-ablation/issues/20) | λ sensitivity plots and final figures | 15, 17 |
| [21](https://github.com/DarkPhoenix645/groupim-lightgcn-ablation/issues/21) | Report: method, datasets, results, ablations, limitations | 5–7, 17–20 |
| [22](https://github.com/DarkPhoenix645/groupim-lightgcn-ablation/issues/22) | Slides or demo if the course requires them | 21 |
| [23](https://github.com/DarkPhoenix645/groupim-lightgcn-ablation/issues/23) | Clean repo: README, exact commands, results CSVs | 17 |

## Experiment grid (once #11 is green)

Upper bound, not a commitment, until #11 records times on the 3080.

- **Main:** MLP vs LightGCN K∈{1,2,3} × {maxpool, meanpool, attention} × 3 seeds × 4 datasets (144). The Weeplaces slice is the comparison that can sit next to Table 2. The other three datasets are the variant track.
- **Ablation:** 7 Table-3-style rows × MLP + best LightGCN × attention × 3 seeds × 4 datasets (168; reuse overlap with main).
- **LightGCN extras:** pretrain on/off, dim {32,64,128}, tied item embeddings.
- **λ:** 2^-4 … 2^6 for MLP and best LightGCN.
- **Extra seeds:** 5–10 on headline pairs, smallest datasets first, leftover GPU only.

Weeplaces MLP attention baseline (seeds 1111–1113, mean ± sample std): N@20 0.509 ± 0.002, N@50 0.532 ± 0.002, R@20 0.693 ± 0.005, R@50 0.783 ± 0.004. Paper Table 2 targets: 0.521, 0.546, 0.716, 0.813. Detail: [dataset-notes/weeplaces.md](dataset-notes/weeplaces.md).
