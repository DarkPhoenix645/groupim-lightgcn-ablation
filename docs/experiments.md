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
