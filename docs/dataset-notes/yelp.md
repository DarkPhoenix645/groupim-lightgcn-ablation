## Yelp

The converter is `python -m groupim_lightgcn_ablation.yelp`. It keeps California businesses whose city is Los Angeles and whose category token is exactly `Restaurants`, then builds groups from a user and friends at that restaurant. A review timestamp with a clock uses a 15-minute window. A date with no clock uses the calendar day. Users and restaurants with fewer than 5 distinct partners are dropped.

The Yelp Open Dataset in `data/raw/yelp/` returns 0 Los Angeles restaurants, so `data/yelp/` has no author CSVs. Yelp stays in the study. Another city in that dump is not a substitute.
