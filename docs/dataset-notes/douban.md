## Douban

Source: Douban-Event benchmark, Yin et al. ICDE 2019, from Hongzhi Yin's dataset page.
The release stores event ids. It has no venue id, so the item is the event.
A group is a co-attendance set from that benchmark, kept when at least two
members and one shared event survive the 10+ filter.
Benchmark before the filter: 70,743 users, 60,028 events, 109,538 groups (110,597 lines; repeated ids have the same members).
Distinct attendance after the filter: 3,210,994.

| Field | Paper | Computed | Delta |
| --- | ---: | ---: | ---: |
| users | 23,032 | 42,053 | 19,021 |
| items | 10,039 | 30,336 | 20,297 |
| groups | 58,983 | 91,901 | 32,918 |
| user_item | 1,731,429 | 3,210,994 | 1,479,565 |
| group_item | 93,635 | 143,008 | 49,373 |
| items_per_user | 75.1700 | 76.3559 | 1.1859 |
| items_per_group | 1.5900 | 1.5561 | -0.0339 |
| group_size | 4.2000 | 5.0775 | 0.8775 |
