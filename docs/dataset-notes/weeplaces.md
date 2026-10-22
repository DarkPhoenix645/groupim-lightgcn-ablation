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

Author setup: attention aggregator, user pretrain (100 epochs), MI pretrain (10 epochs), 200 group epochs, embedding size 64. The checkpoint is the best validation NDCG@100. Test metrics below are that checkpoint. Seeds 1111, 1112, 1113. Per-seed rows are `out/weeplaces/seed-*/metrics.csv`.

These are the MLP baseline for later tables. The gap to Table 2 is small and stable across seeds, so it is recorded rather than treated as a broken port. Sample standard deviation, n−1.

| Metric | Seed 1111 | Seed 1112 | Seed 1113 | Mean ± std | Paper |
| --- | ---: | ---: | ---: | ---: | ---: |
| N@20 | 0.510 | 0.507 | 0.509 | 0.509 ± 0.002 | 0.521 |
| N@50 | 0.533 | 0.530 | 0.533 | 0.532 ± 0.002 | 0.546 |
| R@20 | 0.692 | 0.689 | 0.698 | 0.693 ± 0.005 | 0.716 |
| R@50 | 0.783 | 0.779 | 0.788 | 0.783 ± 0.004 | 0.813 |

Wall clock was 7,124 s, 7,090 s, and 7,216 s.

Validation curves and the group train loss are in [figures/catalog.md](../figures/catalog.md) under `weeplaces-mlp-attention-1111-1113`. Seed 1111's text log starts at group epoch 91. Seeds 1112 and 1113 are complete.
