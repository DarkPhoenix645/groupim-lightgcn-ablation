# Dataset CSV contract

On-disk format inherited from [CrowdDynamicsLab/GroupIM](https://github.com/CrowdDynamicsLab/GroupIM). One directory per dataset, same filenames. Loaders must not grow a second schema. ADR-0003.

Each line of an interaction file is a `(id, id)` pair. `group_users.csv` is `(group, user)`.

| File | Role |
| --- | --- |
| `train_ui.csv` | Every interaction of the training users (70% of users). A training user who is also a member of a validation or test group keeps those interactions here. |
| `val_ui_tr.csv` | Fold-in user-item interactions for validation users. |
| `val_ui_te.csv` | Held-out user-item interactions for validation (user-level eval). |
| `test_ui_tr.csv` | Fold-in user-item interactions for test users. |
| `test_ui_te.csv` | Held-out user-item interactions for test (user-level eval). |
| `train_gi.csv` | Training group-item interactions. |
| `val_gi.csv` | Validation group-item interactions. |
| `test_gi.csv` | Test group-item interactions. |
| `group_users.csv` | Membership `(group, user)` for every group in train/val/test. |

## Splits

Measured on `data/weeplaces/` and reproduced by `groupim_lightgcn_ablation.csv_contract.emit_author_csvs` (seed 1111):

- Partition **groups** 70 / 10 / 20. Sizes are `round_half_up(0.7 n)`, `round_half_up(0.1 n)`, and the remainder. Group ids are one contiguous range: train, then validation, then test.
- Validation and test groups are **ephemeral**: that member combination is absent from training. Groups that share a member combination stay in the same split. Larger combinations are placed first so the whole set still fits one side of the 70/10/20 cut. The released Weeplaces files leak a few combinations; new datasets must not.
- Partition **users** the same 70 / 10 / 20 way, into disjoint contiguous id blocks in the same order. This is the author file layout. `train_ui` is not a copy of the validation and test histories.
- Each validation or test user holds out `ceil(0.4 n)` items (`n = 1` holds out the only item). The rest are fold-in. Held-out pairs are not written into `train_ui`. On Weeplaces this matches about 93% of evaluation users; the other long-history users differ by one item.
- The group loader trains on `train_ui` plus the two fold-in files. That is the file layout behind the paper's "individual interactions of all users are used for training." Held-out user items stay ranking targets.

## Dataset construction (paper §5.1)

| Dataset | Construction | On this branch |
| --- | --- | --- |
| Weeplaces | Author files under `data/weeplaces/`. Format reference. Not rebuilt. | Nine CSVs present. |
| Yelp | Los Angeles restaurants. Group = a user and friends at that restaurant in a 15-minute window, or the same calendar day when the review has no clock. Keep users/restaurants with 5+ distinct partners. | Converter is in place. The public dump has no Los Angeles restaurants, so `data/yelp/` is not written. [yelp.md](dataset-notes/yelp.md). |
| Gowalla | Restaurant check-ins through June 2011 plus the friend graph. Group = one user and friends at the same place within 15 minutes. A later outing is another group. Keep users/places with 5+ distinct partners. | Nine CSVs in `data/gowalla/`. [gowalla.md](dataset-notes/gowalla.md). |
| Douban | Co-attendance groups. Keep users and items with 10+ distinct partners. | Nine CSVs in `data/douban/`. The public benchmark's item is the event, not a venue. [douban.md](dataset-notes/douban.md). |

Counts that miss Table 1 are recorded in [table1-comparison.md](table1-comparison.md). A dataset is not dropped (ADR-0005).
