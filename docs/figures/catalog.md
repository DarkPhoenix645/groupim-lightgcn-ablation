# Figure catalog

Figures for a run live in `results/runs/<run_id>/figures/`. The block below the marker is rewritten by the launcher from those directories. Do not add a plot without a run directory.

The Weeplaces MLP attention baseline is three launcher records, not a separate figure set:

| Seed | Run |
| --- | --- |
| 1111 | `8c545e92cb67730b` |
| 1112 | `6e836a908470f5f0` |
| 1113 | `6e2bf78cbe14734e` |

Seed 1111's log starts at group epoch 91. Seeds 1112 and 1113 include epoch 1. Test mean ± sample std: N@20 0.509 ± 0.002, N@50 0.532 ± 0.002, R@20 0.693 ± 0.005, R@50 0.783 ± 0.004. Detail: [dataset-notes/weeplaces.md](../dataset-notes/weeplaces.md).

<!-- launcher-runs:start -->

## Launcher runs

| Figure | Run | Plotted | Y axis |
| --- | --- | --- | --- |
| [val-r20.svg](../../results/runs/613d5f01c0af8e14/figures/val-r20.svg) | `613d5f01c0af8e14` | Validation Recall@20 every eval epoch. Seed 1111. | Recall@20 |
| [val-r50.svg](../../results/runs/613d5f01c0af8e14/figures/val-r50.svg) | `613d5f01c0af8e14` | Validation Recall@50 every eval epoch. Seed 1111. | Recall@50 |
| [val-n100.svg](../../results/runs/613d5f01c0af8e14/figures/val-n100.svg) | `613d5f01c0af8e14` | Validation NDCG@100 every eval epoch. Seed 1111. | NDCG@100 |
| [train-loss.svg](../../results/runs/613d5f01c0af8e14/figures/train-loss.svg) | `613d5f01c0af8e14` | Mean group-training loss each epoch. Seed 1111. | Mean train loss |
| [val-r20.svg](../../results/runs/6e2bf78cbe14734e/figures/val-r20.svg) | `6e2bf78cbe14734e` | Validation Recall@20 every eval epoch. Seed 1113. | Recall@20 |
| [val-r50.svg](../../results/runs/6e2bf78cbe14734e/figures/val-r50.svg) | `6e2bf78cbe14734e` | Validation Recall@50 every eval epoch. Seed 1113. | Recall@50 |
| [val-n100.svg](../../results/runs/6e2bf78cbe14734e/figures/val-n100.svg) | `6e2bf78cbe14734e` | Validation NDCG@100 every eval epoch. Seed 1113. | NDCG@100 |
| [train-loss.svg](../../results/runs/6e2bf78cbe14734e/figures/train-loss.svg) | `6e2bf78cbe14734e` | Mean group-training loss each epoch. Seed 1113. | Mean train loss |
| [val-r20.svg](../../results/runs/6e836a908470f5f0/figures/val-r20.svg) | `6e836a908470f5f0` | Validation Recall@20 every eval epoch. Seed 1112. | Recall@20 |
| [val-r50.svg](../../results/runs/6e836a908470f5f0/figures/val-r50.svg) | `6e836a908470f5f0` | Validation Recall@50 every eval epoch. Seed 1112. | Recall@50 |
| [val-n100.svg](../../results/runs/6e836a908470f5f0/figures/val-n100.svg) | `6e836a908470f5f0` | Validation NDCG@100 every eval epoch. Seed 1112. | NDCG@100 |
| [train-loss.svg](../../results/runs/6e836a908470f5f0/figures/train-loss.svg) | `6e836a908470f5f0` | Mean group-training loss each epoch. Seed 1112. | Mean train loss |
| [val-r20.svg](../../results/runs/7d138136942c6110/figures/val-r20.svg) | `7d138136942c6110` | Validation Recall@20 every eval epoch. Seed 1111. | Recall@20 |
| [val-r50.svg](../../results/runs/7d138136942c6110/figures/val-r50.svg) | `7d138136942c6110` | Validation Recall@50 every eval epoch. Seed 1111. | Recall@50 |
| [val-n100.svg](../../results/runs/7d138136942c6110/figures/val-n100.svg) | `7d138136942c6110` | Validation NDCG@100 every eval epoch. Seed 1111. | NDCG@100 |
| [train-loss.svg](../../results/runs/7d138136942c6110/figures/train-loss.svg) | `7d138136942c6110` | Mean group-training loss each epoch. Seed 1111. | Mean train loss |
| [val-r20.svg](../../results/runs/8c545e92cb67730b/figures/val-r20.svg) | `8c545e92cb67730b` | Validation Recall@20 every eval epoch. Seed 1111. | Recall@20 |
| [val-r50.svg](../../results/runs/8c545e92cb67730b/figures/val-r50.svg) | `8c545e92cb67730b` | Validation Recall@50 every eval epoch. Seed 1111. | Recall@50 |
| [val-n100.svg](../../results/runs/8c545e92cb67730b/figures/val-n100.svg) | `8c545e92cb67730b` | Validation NDCG@100 every eval epoch. Seed 1111. | NDCG@100 |
| [train-loss.svg](../../results/runs/8c545e92cb67730b/figures/train-loss.svg) | `8c545e92cb67730b` | Mean group-training loss each epoch. Seed 1111. | Mean train loss |

## `613d5f01c0af8e14`

Encoder `lightgcn`, aggregator `attention`, dataset `weeplaces`, seed 1111.

| Field | Value |
| --- | --- |
| Losses | `full` |
| λ | 1.0 |
| K | 3 |
| Embedding dim | 64 |
| Epochs | 200 group, 100 user pretrain, 10 MI |
| Pretrain | user True, MI True |
| Tied item embeddings | True |
| CUDA | True |
| Log | [train.log](../../results/runs/613d5f01c0af8e14/train.log) |
| Metrics | [metrics.csv](../../results/runs/613d5f01c0af8e14/metrics.csv) |

Test N@20 0.5640, N@50 0.5875, R@20 0.7466, R@50 0.8360. Wall clock 7197.9 s.

## `6e2bf78cbe14734e`

Encoder `mlp`, aggregator `attention`, dataset `weeplaces`, seed 1113.

| Field | Value |
| --- | --- |
| Losses | `full` |
| λ | 1.0 |
| K | 0 |
| Embedding dim | 64 |
| Epochs | 200 group, 100 user pretrain, 10 MI |
| Pretrain | user True, MI True |
| Tied item embeddings | True |
| CUDA | True |
| Log | [train.log](../../results/runs/6e2bf78cbe14734e/train.log) |
| Metrics | [metrics.csv](../../results/runs/6e2bf78cbe14734e/metrics.csv) |

Test N@20 0.5094, N@50 0.5327, R@20 0.6979, R@50 0.7878. Wall clock 7216.1 s.

## `6e836a908470f5f0`

Encoder `mlp`, aggregator `attention`, dataset `weeplaces`, seed 1112.

| Field | Value |
| --- | --- |
| Losses | `full` |
| λ | 1.0 |
| K | 0 |
| Embedding dim | 64 |
| Epochs | 200 group, 100 user pretrain, 10 MI |
| Pretrain | user True, MI True |
| Tied item embeddings | True |
| CUDA | True |
| Log | [train.log](../../results/runs/6e836a908470f5f0/train.log) |
| Metrics | [metrics.csv](../../results/runs/6e836a908470f5f0/metrics.csv) |

Test N@20 0.5067, N@50 0.5302, R@20 0.6887, R@50 0.7793. Wall clock 7090.3 s.

## `7d138136942c6110`

Encoder `lightgcn`, aggregator `attention`, dataset `weeplaces`, seed 1111.

| Field | Value |
| --- | --- |
| Losses | `full` |
| λ | 1.0 |
| K | 3 |
| Embedding dim | 64 |
| Epochs | 1 group, 1 user pretrain, 1 MI |
| Pretrain | user True, MI True |
| Tied item embeddings | True |
| CUDA | True |
| Log | [train.log](../../results/runs/7d138136942c6110/train.log) |
| Metrics | [metrics.csv](../../results/runs/7d138136942c6110/metrics.csv) |

Test N@20 0.1567, N@50 0.1820, R@20 0.2887, R@50 0.4016. Wall clock 88.3 s.

## `8c545e92cb67730b`

Encoder `mlp`, aggregator `attention`, dataset `weeplaces`, seed 1111.

| Field | Value |
| --- | --- |
| Losses | `full` |
| λ | 1.0 |
| K | 0 |
| Embedding dim | 64 |
| Epochs | 200 group, 100 user pretrain, 10 MI |
| Pretrain | user True, MI True |
| Tied item embeddings | True |
| CUDA | True |
| Log | [train.log](../../results/runs/8c545e92cb67730b/train.log) |
| Metrics | [metrics.csv](../../results/runs/8c545e92cb67730b/metrics.csv) |

Test N@20 0.5098, N@50 0.5333, R@20 0.6924, R@50 0.7830. Wall clock 7123.7 s.
<!-- launcher-runs:end -->
