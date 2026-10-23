# Experiment protocol

How a run is identified and what it must leave on disk. ADR-0007.

## Run config (minimum)

encoder (`mlp` | `lightgcn`), `K` (LightGCN layers; ignored for MLP), aggregator (`maxpool` | `meanpool` | `attention`), losses (`L_G` / `+λL_U` / `+L_MI`), λ, embedding dim, pretrain_user, pretrain_mi, tied item embeddings, dataset, seed.

One config → exactly one results-CSV row. Same config+seed must be detectably identical in the CSV.

## Concurrency

2–3 jobs on the single RTX 3080 (10 GB). Outputs must not clobber.

## Artifacts every run writes

- Results CSV row: headline N@20 / R@20 / N@50 / R@50 (group), wall-clock, config, seed.
- Per-group test metrics for every test group.
- Group size, coherence, aggregate diversity.
- Per-member discriminator scores (needed for MI-variation).

Analysis tickets (#17–#20) consume these files. They do not launch training.

`uv run python -m groupim_lightgcn_ablation.launch` is the command that writes them. The config and seed hash to a run id. Artifacts for that id land in `results/runs/<run_id>/`: `train.log`, `metrics.csv`, `test_groups.csv`, `member_scores.csv`, `model_user.pt`, `model_group.pt`, and `figures/*.svg`. There is no separate output directory. The ledger of one row per run id is `results/runs.csv`. A repeat of the same config and seed keeps that id and replaces the row. Plots from launcher runs are appended to [figures/catalog.md](figures/catalog.md).

Coherence on a test group is the mean pairwise Pearson correlation of its members' item vectors. Aggregate diversity is the count of distinct items across those members. Discriminator scores are the bilinear logits of present members.

`encoder=lightgcn` swaps only the user preference encoder. K is the number of propagation layers. The readout is the mean of the initial ID embeddings and the K propagated layers. The adjacency is the symmetric normalized bipartite graph of `train_ui` plus `val_ui_tr` and `test_ui_tr`. Held-out user-item files are not edges. Propagation runs once per forward from the current embeddings.
