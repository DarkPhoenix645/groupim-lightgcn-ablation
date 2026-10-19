## Yelp

Source: Yelp-LA benchmark, Yin et al. ICDE 2019, from Hongzhi Yin's dataset page.
The Yelp Open Dataset has no Los Angeles restaurants, so it is not the source.
Groups in this release are already built, and the files have no timestamps,
so the 15-minute window was not applied again.
Users and businesses with fewer than 5 distinct partners are dropped.
Benchmark before the filter: 34,504 users, 22,611 businesses, 24,103 groups.
Distinct visits after the filter: 424,774.
These metrics are on the Yin-LA variant, not the paper's Yelp benchmark.

| Field | Paper | Computed | Delta |
| --- | ---: | ---: | ---: |
| users | 7,037 | 18,911 | 11,874 |
| items | 7,671 | 11,126 | 3,455 |
| groups | 10,214 | 14,396 | 4,182 |
| user_item | 220,091 | 424,774 | 204,683 |
| group_item | 11,160 | 16,433 | 5,273 |
| items_per_user | 31.3000 | 22.4617 | -8.8383 |
| items_per_group | 1.0900 | 1.1415 | 0.0515 |
| group_size | 6.8000 | 5.7919 | -1.0081 |
