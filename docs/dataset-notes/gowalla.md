## Gowalla

Source: figshare 10.6084/m9.figshare.22126586 (Gowalla API dump).
Restaurants are spots whose category is under Food.
Places are limited to a latitude/longitude box for the continental US, Alaska, and Hawaii.
Check-ins at or after 2011-07-01 are dropped.
Window: 15 minutes.
US Food spots: 321,444.
Check-ins before the 5+ filter: 4,727,020.
Distinct visits after the filter: 2,572,585.
Each group is one 15-minute outing at one restaurant, so items per group is 1. The same people on another outing are another group. Those 225,657 groups are 43,265 member sets, with 4.28 restaurants per set and mean size 2.52. Table 1 has 44,565 groups, 3.47 items per group, and group size 2.8.

| Field | Paper | Computed | Delta |
| --- | ---: | ---: | ---: |
| users | 25,405 | 64,002 | 38,597 |
| items | 38,306 | 160,171 | 121,865 |
| groups | 44,565 | 225,657 | 181,092 |
| user_item | 1,025,500 | 2,572,585 | 1,547,085 |
| group_item | 154,526 | 225,657 | 71,131 |
| items_per_user | 40.3700 | 40.1954 | -0.1746 |
| items_per_group | 3.4700 | 1 | -2.4700 |
| group_size | 2.8000 | 2.1493 | -0.6507 |
