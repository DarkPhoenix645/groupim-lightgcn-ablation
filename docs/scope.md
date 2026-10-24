# 1-week full-scope ablation

Canonical plan. Work is GitHub issue [#1](https://github.com/DarkPhoenix645/groupim-lightgcn-ablation/issues/1) and its sub-issues. Blocked-by links on those issues are the live gate (ADR-0006).

**Hardware:** one RTX 3080 (10 GB). **Budget:** one full Weeplaces run is 2.0 GPU hours, measured on this card. The locked queue below is 94 GPU hours. One job was timed. A second Weeplaces job may share the card only after a paired run finishes.

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

## Locked schedule (#11)

Measured on one RTX 3080 (10 GB), one job, Weeplaces, author budget (100 user-pretrain epochs, 10 MI epochs, 200 group epochs, dim 64, batch 256, attention). MLP seeds 1111–1113 took 7,124 s, 7,090 s, and 7,216 s. LightGCN K=3 seed 1111 took 7,198 s (`613d5f01c0af8e14`). The unit for every full Weeplaces run below is **2.0 GPU hours**. K=1 and K=2 use that same unit: group eval epochs in the K=3 log stay near 36 s, the same as MLP.

Yelp, Gowalla, and Douban have no smoke on this card. Their full grids are outside this queue. A later 1-epoch smoke can spend leftover time on a variant. It does not extend the 94 hours.

Runs already on disk are reused. The three MLP attention seeds and the LightGCN K=3 attention seed 1111 are not trained again.

| Ticket | What is queued | New runs | GPU hours |
| --- | --- | ---: | ---: |
| [#12](https://github.com/DarkPhoenix645/groupim-lightgcn-ablation/issues/12) | Weeplaces. MLP maxpool and meanpool × 3 seeds. LightGCN K∈{1,2,3} × {maxpool, meanpool, attention} × 3 seeds, except K=3 attention seed 1111. | 32 | 64 |
| [#13](https://github.com/DarkPhoenix645/groupim-lightgcn-ablation/issues/13) | Weeplaces, attention, seed 1111. MLP and the best LightGCN K from #12. Six rows besides the full model: `L_G`, `L_G+λL_U`, `L_G+L_MI`, uniform weights, cosine-similarity weights, no pre-training. The full-model row is the #12 cell. | 12 | 24 |
| [#14](https://github.com/DarkPhoenix645/groupim-lightgcn-ablation/issues/14) | Weeplaces, attention, seed 1111, best K. Dim 32, dim 128, tied item embeddings off. Dim 64, pretrain on, and pretrain off are the #12 full model and the #13 no-pre-training row. | 3 | 6 |
| [#15](https://github.com/DarkPhoenix645/groupim-lightgcn-ablation/issues/15) | Leftover only, after the 94 hours. Weeplaces, seed 1111, MLP and best LightGCN, λ = 2^-4 … 2^6. λ = 1 is the full model. | 20 | 40 |
| [#16](https://github.com/DarkPhoenix645/groupim-lightgcn-ablation/issues/16) | Not queued. | 0 | 0 |

#13 and #14 start after #12 has picked the best LightGCN K on Weeplaces validation N@100. #13 also waits on the ablation switches ([#10](https://github.com/DarkPhoenix645/groupim-lightgcn-ablation/issues/10)).

The 144-run four-dataset main grid, the 168-run ablation, three seeds on #13, and seeds 5–10 are the cuts this card forces. [#21](https://github.com/DarkPhoenix645/groupim-lightgcn-ablation/issues/21) records those cuts.

Weeplaces MLP attention baseline (seeds 1111–1113, mean ± sample std): N@20 0.509 ± 0.002, N@50 0.532 ± 0.002, R@20 0.693 ± 0.005, R@50 0.783 ± 0.004. Paper Table 2 targets: 0.521, 0.546, 0.716, 0.813. The LightGCN timing seed is one run, K=3 attention: N@20 0.564, N@50 0.588, R@20 0.747, R@50 0.836. Detail: [dataset-notes/weeplaces.md](dataset-notes/weeplaces.md).
