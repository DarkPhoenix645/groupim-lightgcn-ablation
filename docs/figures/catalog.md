# Figure catalog

Every file in this directory is listed here. A new figure gets a row in the index and a block under its run. Do not add a plot without both.

Run ids are `{dataset}-{encoder}-{aggregator}-{seeds}`. One run id can own several figures. Conditions live on the run, not copied onto each file.

## Index

| Figure | Run | Plotted | Y axis |
| --- | --- | --- | --- |
| [weeplaces-val-r20.svg](weeplaces-val-r20.svg) | `weeplaces-mlp-attention-1111-1113` | Validation Recall@20 every 5 group epochs, three seeds. Dashed line is the paper test score 0.716. | Recall@20 |
| [weeplaces-val-r50.svg](weeplaces-val-r50.svg) | `weeplaces-mlp-attention-1111-1113` | Validation Recall@50 every 5 group epochs, three seeds. Dashed line is the paper test score 0.813. | Recall@50 |
| [weeplaces-val-n100.svg](weeplaces-val-n100.svg) | `weeplaces-mlp-attention-1111-1113` | Validation NDCG@100 every 5 group epochs, three seeds. This is the checkpoint metric. The paper does not report it. | NDCG@100 |
| [weeplaces-train-loss.svg](weeplaces-train-loss.svg) | `weeplaces-mlp-attention-1111-1113` | Mean group-training loss after each epoch, three seeds. | Mean train loss |

## `weeplaces-mlp-attention-1111-1113`

MLP GroupIM, attention aggregator, author Weeplaces files. Reproduction baseline in [dataset-notes/weeplaces.md](../dataset-notes/weeplaces.md).

| Field | Value |
| --- | --- |
| Dataset | `weeplaces` (author release, not rebuilt) |
| Encoder | `mlp` |
| K | ignored |
| Aggregator | `attention` |
| Losses | `L_G + λ L_U + L_MI` |
| λ | 1.0 |
| Embedding dim | 64 |
| Dropout | 0.4 |
| Batch size | 256 |
| Negatives per group | 5 |
| User pretrain | on, 100 epochs, Adam lr 0.01 |
| MI pretrain | on, 10 epochs (hardcoded) |
| Group epochs | 200 |
| Group optimizer | Adam, lr 5e-3, weight decay 0 |
| Eval | validation every 5 epochs |
| Checkpoint | best validation NDCG@100, then test |
| Item embeddings | group predictor copied from the pretrained user predictor once, then trained |
| Seeds | 1111, 1112, 1113 |
| Workers | 0 |
| Hardware | one RTX 3080 (10 GB), WSL |
| Command | `uv run python train.py --dataset weeplaces --aggregator attention --pretrain_user --pretrain_mi --cuda --seed <seed> --save out/weeplaces/seed-<seed>/model_user.pt --save_group out/weeplaces/seed-<seed>/model_group.pt` |

| Seed | Log | Metrics | Wall clock |
| --- | --- | --- | --- |
| 1111 | [seed-1111.log](../../results/weeplaces/seed-1111.log) | [metrics.csv](../../out/weeplaces/seed-1111/metrics.csv) | 7,124 s |
| 1112 | [seed-1112.log](../../results/weeplaces/seed-1112.log) | [metrics.csv](../../out/weeplaces/seed-1112/metrics.csv) | 7,090 s |
| 1113 | [seed-1113.log](../../results/weeplaces/seed-1113.log) | [metrics.csv](../../out/weeplaces/seed-1113/metrics.csv) | 7,216 s |

Seed 1111's log starts at group epoch 91. The curves for that seed start there. Seeds 1112 and 1113 include epoch 1.

Test metrics from the saved checkpoints, mean ± sample std: N@20 0.509 ± 0.002, N@50 0.532 ± 0.002, R@20 0.693 ± 0.005, R@50 0.783 ± 0.004.
