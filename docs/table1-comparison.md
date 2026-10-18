# Table 1 comparison

Paper counts are GroupIM Table 1. Computed counts read the nine author CSVs:
distinct users, distinct items, distinct groups, interaction rows, and the three averages.
A gap is recorded here. A dataset is not dropped because a count misses the paper (ADR-0005).

Weeplaces is the format reference. Its user, item, and group counts match Table 1.
The printed Weeplaces user-item total (1,358,458) and group-item total (180,229) do not
match the printed averages or the released CSVs. The averages do: about 58.83 items per
user and 2.95 items per group. Weeplaces is not rebuilt.

Gowalla and Douban are read from their nine CSVs when those files exist.
Yelp stays unconverted while the public dump has no Los Angeles restaurants.
Per-dataset notes live in docs/dataset-notes/.

| Dataset | Field | Paper | Computed | Delta |
| --- | --- | ---: | ---: | ---: |
| weeplaces | users | 8,643 | 8,643 | 0 |
| weeplaces | items | 25,081 | 25,081 | 0 |
| weeplaces | groups | 22,733 | 22,733 | 0 |
| weeplaces | user_item | 1,358,458 | 508,485 | -849,973 |
| weeplaces | group_item | 180,229 | 67,130 | -113,099 |
| weeplaces | items_per_user | 58.8300 | 58.8320 | 0.0020 |
| weeplaces | items_per_group | 2.9500 | 2.9530 | 0.0030 |
| weeplaces | group_size | 2.9000 | 2.8887 | -0.0113 |
| yelp | status |  | not converted |  |
| gowalla | users | 25,405 | 64,002 | 38,597 |
| gowalla | items | 38,306 | 160,171 | 121,865 |
| gowalla | groups | 44,565 | 225,657 | 181,092 |
| gowalla | user_item | 1,025,500 | 2,572,585 | 1,547,085 |
| gowalla | group_item | 154,526 | 225,657 | 71,131 |
| gowalla | items_per_user | 40.3700 | 40.1954 | -0.1746 |
| gowalla | items_per_group | 3.4700 | 1 | -2.4700 |
| gowalla | group_size | 2.8000 | 2.1493 | -0.6507 |
| douban | users | 23,032 | 42,053 | 19,021 |
| douban | items | 10,039 | 30,336 | 20,297 |
| douban | groups | 58,983 | 91,901 | 32,918 |
| douban | user_item | 1,731,429 | 3,210,994 | 1,479,565 |
| douban | group_item | 93,635 | 143,008 | 49,373 |
| douban | items_per_user | 75.1700 | 76.3559 | 1.1859 |
| douban | items_per_group | 1.5900 | 1.5561 | -0.0339 |
| douban | group_size | 4.2000 | 5.0775 | 0.8775 |

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

## Yelp

The converter is `python -m groupim_lightgcn_ablation.yelp`. It keeps California businesses whose city is Los Angeles and whose category token is exactly `Restaurants`, then builds groups from a user and friends at that restaurant. A review timestamp with a clock uses a 15-minute window. A date with no clock uses the calendar day. Users and restaurants with fewer than 5 distinct partners are dropped.

The Yelp Open Dataset in `data/raw/yelp/` returns 0 Los Angeles restaurants, so `data/yelp/` has no author CSVs. Yelp stays in the study. Another city in that dump is not a substitute.
