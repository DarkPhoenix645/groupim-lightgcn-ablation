# 1-week full-scope ablation

Canonical plan. GitHub Issues are the tracker (ADR-0006). The HTML in `.scratch/` is a local copy of this list, not committed.

**Hardware:** one RTX 4080. **Budget:** ~450 runs, 80–120 GPU hours; 2–3 concurrent jobs ≈ 30–40h wall-clock. Queue the grid by day 3; leave it overnight.

**Critical path:** Yelp, Gowalla, Douban in the [CSV contract](data-contract.md). Gowalla and Douban CSVs are in `data/`. The Yelp converter is in place; the public dump has no Los Angeles restaurants, so `data/yelp/` is empty. Counts and gaps: [table1-comparison.md](table1-comparison.md).

**Gates**

- End of day 2: Table 1 mismatch → rebuild via §5.1, document, do not drop.
- End of day 3: whole grid queued.
- A small or null LightGCN gain is still a finding.

Epic: [#1](https://github.com/DarkPhoenix645/groupim-lightgcn-ablation/issues/1).

## Tickets

| # | Title | HTML | Blocked by |
| --- | --- | --- | --- |
| [2](https://github.com/DarkPhoenix645/groupim-lightgcn-ablation/issues/2) | Port CrowdDynamicsLab/GroupIM onto uv + Torch 2.x CUDA | b1 | — |
| [3](https://github.com/DarkPhoenix645/groupim-lightgcn-ablation/issues/3) | Audit Weeplaces and reproduce MLP GroupIM baseline | a1, b2 | 2 |
| [4](https://github.com/DarkPhoenix645/groupim-lightgcn-ablation/issues/4) | Experiment launcher: config in, one CSV row out, artifacts saved | c1, c2 | 2 |
| [5](https://github.com/DarkPhoenix645/groupim-lightgcn-ablation/issues/5) | Yelp LA restaurants → author CSVs + Table 1 check | a2, a5, a6 | — |
| [6](https://github.com/DarkPhoenix645/groupim-lightgcn-ablation/issues/6) | Gowalla restaurants → 15-min friend groups + Table 1 check | a3, a5, a6 | — |
| [7](https://github.com/DarkPhoenix645/groupim-lightgcn-ablation/issues/7) | Douban events → co-attendance groups + Table 1 check | a4, a5, a6 | — |
| [8](https://github.com/DarkPhoenix645/groupim-lightgcn-ablation/issues/8) | LightGCN user encoder + Weeplaces smoke | b3, b6 | 3 |
| [9](https://github.com/DarkPhoenix645/groupim-lightgcn-ablation/issues/9) | LightGCN user-loss pretraining analogue | b4 | 8 |
| [10](https://github.com/DarkPhoenix645/groupim-lightgcn-ablation/issues/10) | Ablation switches for every Table-3-style row | b5 | 8, 9 |
| [11](https://github.com/DarkPhoenix645/groupim-lightgcn-ablation/issues/11) | MLP + LightGCN smoke on all four datasets; freeze the grid schedule | i1 | 3–8 |
| [12](https://github.com/DarkPhoenix645/groupim-lightgcn-ablation/issues/12) | Main grid: MLP vs LightGCN K=1,2,3 × pooling × 3 seeds × 4 datasets | e1 | 11 |
| [13](https://github.com/DarkPhoenix645/groupim-lightgcn-ablation/issues/13) | Table-3-style ablation for MLP and best LightGCN | e2 | 10, 11 |
| [14](https://github.com/DarkPhoenix645/groupim-lightgcn-ablation/issues/14) | LightGCN extras: pretrain on/off, dim {32,64,128}, tied item embeddings | e3 | 9, 11 |
| [15](https://github.com/DarkPhoenix645/groupim-lightgcn-ablation/issues/15) | λ sweep 2^-4..2^6 for MLP and best LightGCN | e4 | 10, 11 |
| [16](https://github.com/DarkPhoenix645/groupim-lightgcn-ablation/issues/16) | Extra seeds (5–10) for headline comparisons, leftover GPU | e5 | 12 |
| [17](https://github.com/DarkPhoenix645/groupim-lightgcn-ablation/issues/17) | Per-dataset results tables: mean±std and paired LightGCN vs MLP | d1 | 12–15 |
| [18](https://github.com/DarkPhoenix645/groupim-lightgcn-ablation/issues/18) | Group-size, coherence, and diversity quartile analysis | d2 | 4, 12 |
| [19](https://github.com/DarkPhoenix645/groupim-lightgcn-ablation/issues/19) | MI-variation and short-history hypothesis checks | d3, d4 | 4, 12, 13 |
| [20](https://github.com/DarkPhoenix645/groupim-lightgcn-ablation/issues/20) | λ sensitivity plots and final figures | d5 | 15, 17 |
| [21](https://github.com/DarkPhoenix645/groupim-lightgcn-ablation/issues/21) | Report: method, datasets, results, ablations, limitations | w1 | 5–7, 17–20 |
| [22](https://github.com/DarkPhoenix645/groupim-lightgcn-ablation/issues/22) | Slides or demo if the course requires them | w2 | 21 |
| [23](https://github.com/DarkPhoenix645/groupim-lightgcn-ablation/issues/23) | Clean repo: README, exact commands, results CSVs | w3 | 17 |

`w4` (buffer) is not a ticket.

## Experiment grid (once #11 is green)

- **Main:** MLP vs LightGCN K∈{1,2,3} × {maxpool, meanpool, attention} × 3 seeds × 4 datasets (144).
- **Ablation:** 7 Table-3-style rows × MLP + best LightGCN × attention × 3 seeds × 4 datasets (168; reuse overlap with main).
- **LightGCN extras:** pretrain on/off, dim {32,64,128}, tied item embeddings.
- **λ:** 2^-4 … 2^6 for MLP and best LightGCN.
- **Extra seeds:** 5–10 on headline pairs, smallest datasets first, leftover GPU only.

Headline Weeplaces MLP attention numbers from the paper (replace with our run if close): N@20 0.521, R@20 0.716, N@50 0.546, R@50 0.813.
