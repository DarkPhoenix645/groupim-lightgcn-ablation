# Weeplaces

Author-released files in `data/weeplaces/`. Not rebuilt. This is the reproduction baseline (ADR-0008).

The printed Table 1 interaction totals are not the files. 8,643 × 58.83 = 508,468 and 22,733 × 2.95 = 67,062. The CSVs are 508,485 user-item rows and 67,130 group-item rows. Users, items, groups, and both averages match the table. The files are the training input.

## Split

Measured on the released CSVs. Users and groups are disjoint 70/10/20 blocks, contiguous ids, train then validation then test.

| Split | Users | User ids | Groups | Group ids |
| --- | ---: | --- | ---: | --- |
| train | 6,050 | 0–6049 | 15,913 | 0–15912 |
| val | 864 | 6050–6913 | 2,273 | 15913–18185 |
| test | 1,729 | 6914–8642 | 4,547 | 18186–22732 |

`train_ui` is only the training users. Validation and test users are held out of that file. Fold-in is `val_ui_tr` / `test_ui_tr`. Held-out user items are `val_ui_te` / `test_ui_te`. The group loader still trains on `train_ui` plus the two fold-in files.

`ceil(0.4 n)` held-out items matches 807/864 validation users and 1,609/1,729 test users. The other 57 and 120 users each hold out one fewer item than that rule. The files were not edited to force the ceiling.

## Membership and leaks

Every group member has at least one user-item row. 8,550 distinct members, 8,643 users.

One validation member-set also appears in training: `(5824, 6708)`. Five test member-sets also appear in training: `(899, 5345)`, `(1040, 1942)`, `(4338, 6204)`, `(6148, 8509)`, `(6351, 7765)`. Left as released.

Of 13,846 test group-item rows, 12,755 are also in some member's `train_ui`, 718 are only in test fold-in, 276 are only in the held-out user file, and 97 are in no individual file.

## MLP attention baseline

Author setup: attention aggregator, user pretrain (100 epochs), MI pretrain (10 epochs), 200 group epochs, embedding size 64. The checkpoint is the best validation NDCG@100. Test metrics below are that checkpoint. Seeds 1111, 1112, 1113 are run ids `8c545e92cb67730b`, `6e836a908470f5f0`, and `6e2bf78cbe14734e` in `results/runs/`.

These are the MLP baseline for later tables. The gap to Table 2 is small and stable across seeds, so it is recorded rather than treated as a broken port. Sample standard deviation, n−1.

| Metric | Seed 1111 | Seed 1112 | Seed 1113 | Mean ± std | Paper |
| --- | ---: | ---: | ---: | ---: | ---: |
| N@20 | 0.510 | 0.507 | 0.509 | 0.509 ± 0.002 | 0.521 |
| N@50 | 0.533 | 0.530 | 0.533 | 0.532 ± 0.002 | 0.546 |
| R@20 | 0.692 | 0.689 | 0.698 | 0.693 ± 0.005 | 0.716 |
| R@50 | 0.783 | 0.779 | 0.788 | 0.783 ± 0.004 | 0.813 |

Wall clock was 7,124 s, 7,090 s, and 7,216 s.

Validation curves and the group train loss are in [figures/catalog.md](../figures/catalog.md). Seed 1111's text log starts at group epoch 91. Seeds 1112 and 1113 are complete.

## LightGCN K=3 attention timing run

Same author budget as the MLP baseline. Seed 1111 is run `613d5f01c0af8e14`. Wall clock 7,198 s. That is the LightGCN time-per-run in the locked schedule ([scope.md](../scope.md)).

Test metrics for this seed: N@20 0.564, N@50 0.588, R@20 0.747, R@50 0.836.

## #12 Weeplaces grid

Author budget, three seeds, `losses=full`. MLP maxpool and meanpool plus LightGCN K∈{1,2,3} × {attention, meanpool, maxpool}. The four earlier full runs are reused. Best LightGCN K is **1**: mean of each attention seed's max validation group N@100 is K=1 0.6007, K=3 0.5999, K=2 0.5966. Test N@20 ranks K=2 first (0.565) then K=3 (0.563) then K=1 (0.561); that is not the selection rule.

Test mean ± sample std (n=3):

| Encoder | Agg | N@20 | N@50 | R@20 | R@50 |
| --- | --- | ---: | ---: | ---: | ---: |
| MLP | attention | 0.5086 ± 0.0017 | 0.5321 ± 0.0017 | 0.6930 ± 0.0046 | 0.7834 ± 0.0042 |
| MLP | meanpool | 0.5090 ± 0.0023 | 0.5329 ± 0.0021 | 0.6918 ± 0.0043 | 0.7844 ± 0.0049 |
| MLP | maxpool | 0.4636 ± 0.0043 | 0.4889 ± 0.0047 | 0.6448 ± 0.0027 | 0.7450 ± 0.0033 |
| LightGCN K=1 | attention | 0.5610 ± 0.0031 | 0.5838 ± 0.0027 | 0.7493 ± 0.0027 | 0.8352 ± 0.0036 |
| LightGCN K=1 | meanpool | 0.5599 ± 0.0011 | 0.5833 ± 0.0004 | 0.7434 ± 0.0031 | 0.8322 ± 0.0030 |
| LightGCN K=1 | maxpool | 0.5198 ± 0.0036 | 0.5446 ± 0.0030 | 0.7031 ± 0.0057 | 0.7982 ± 0.0084 |
| LightGCN K=2 | attention | 0.5651 ± 0.0011 | 0.5881 ± 0.0011 | 0.7489 ± 0.0046 | 0.8365 ± 0.0040 |
| LightGCN K=2 | meanpool | 0.5632 ± 0.0014 | 0.5855 ± 0.0014 | 0.7477 ± 0.0013 | 0.8308 ± 0.0022 |
| LightGCN K=2 | maxpool | 0.5260 ± 0.0024 | 0.5502 ± 0.0024 | 0.7087 ± 0.0015 | 0.8014 ± 0.0036 |
| LightGCN K=3 | attention | 0.5635 ± 0.0007 | 0.5864 ± 0.0011 | 0.7477 ± 0.0012 | 0.8346 ± 0.0016 |
| LightGCN K=3 | meanpool | 0.5613 ± 0.0028 | 0.5846 ± 0.0026 | 0.7447 ± 0.0039 | 0.8323 ± 0.0030 |
| LightGCN K=3 | maxpool | 0.5254 ± 0.0022 | 0.5501 ± 0.0016 | 0.7032 ± 0.0008 | 0.7985 ± 0.0052 |

Summary figures: [issue12-n20-grid.svg](../figures/issue12-n20-grid.svg), [issue12-k-select.svg](../figures/issue12-k-select.svg), [issue12-encoder.svg](../figures/issue12-encoder.svg), [issue12-aggregator.svg](../figures/issue12-aggregator.svg).
