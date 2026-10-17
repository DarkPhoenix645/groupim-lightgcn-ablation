# Author CSV contract is the dataset interface

Every dataset (Weeplaces, Yelp, Gowalla, Douban) must emit the files the author loaders already understand: `train_ui`, `val_ui_tr` / `val_ui_te`, `test_ui_tr` / `test_ui_te`, `train_gi` / `val_gi` / `test_gi`, `group_users`. Splits are 70/10/20 over **groups**, strict ephemeral membership, all user-item interactions in training. See `docs/data-contract.md`.

**Status:** accepted

**Consequences:** new datasets are converters, not new loader code. If Table 1 counts are off, rebuild with paper §5.1 and document the gap; do not invent a second on-disk format.
