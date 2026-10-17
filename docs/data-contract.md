# Dataset CSV contract

On-disk format inherited from [CrowdDynamicsLab/GroupIM](https://github.com/CrowdDynamicsLab/GroupIM). One directory per dataset, same filenames. Loaders must not grow a second schema. ADR-0003.

Each line of an interaction file is a `(id, id)` pair. `group_users.csv` is `(group, user)`.

| File | Role |
| --- | --- |
| `train_ui.csv` | Training user-item interactions. **All** individual interactions go here, including for users who appear in val/test groups. |
| `val_ui_tr.csv` | Fold-in user-item interactions for validation users. |
| `val_ui_te.csv` | Held-out user-item interactions for validation (user-level eval). |
| `test_ui_tr.csv` | Fold-in user-item interactions for test users. |
| `test_ui_te.csv` | Held-out user-item interactions for test (user-level eval). |
| `train_gi.csv` | Training group-item interactions. |
| `val_gi.csv` | Validation group-item interactions. |
| `test_gi.csv` | Test group-item interactions. |
| `group_users.csv` | Membership `(group, user)` for every group in train/val/test. |

## Splits

- Partition **groups** 70 / 10 / 20 train / val / test.
- Val and test groups are **ephemeral**: that member combination is absent from training.
- User-item interactions of all users train the user encoder; they are not held out with the group split.

## Dataset construction (paper §5.1)

| Dataset | Construction |
| --- | --- |
| Weeplaces | Author files under `data/weeplaces/` in CrowdDynamicsLab/GroupIM. Format reference. |
| Yelp | Los Angeles restaurants. Convert to the files above. |
| Gowalla | Restaurant check-ins to June 2011 + friend graph. Group = same POI within 15 minutes, user + friends. Keep users/items with 5+ check-ins. |
| Douban | Event-venue. Keep users/venues with 10+ interactions. Group = users attending an event together. |

If counts diverge from paper Table 1, rebuild with §5.1 and write the deviation down. Do not drop the dataset (ADR-0005).
