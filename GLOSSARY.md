# Group recommendation

Neural group recommendation: predict items a **group** of **users** will like, using individual histories plus a group objective with mutual information.

## People and groups

| Term | Definition | Avoid |
| --- | --- | --- |
| **User** | A person who interacts with items alone and as a member of groups. | Client, account, member (use member only for the role inside a group) |
| **Item** | The recommendable entity. Dataset-specific kinds are restaurants, POIs, or venues; the model still sees an item. | POI, venue, restaurant (except when stating a dataset's raw source) |
| **Group** | A set of users treated as one recommendation target for an item. | Party, session, cohort |
| **Member** | A user that belongs to a given group. | Participant, attendee (except when describing Douban event construction) |
| **Ephemeral group** | A specific combination of members that does not occur in the training split. Validation and test groups are ephemeral. | Unseen group, cold group, new group |
| **Group size** | The number of members in a group. | |

## Interactions

| Term | Definition | Avoid |
| --- | --- | --- |
| **User-item interaction** | An observed pair `(user, item)` from an individual's history. All of these go to training, including for users who also appear in val/test groups. | |
| **Group-item interaction** | An observed pair `(group, item)`: the group consumed or visited that item together. | |
| **Fold-in interaction** | A user-item edge shown to the model at validation or test time so the user encoder has that user's recent history. It is not a ranking target. | Warm-up, context edge, side interaction |
| **Held-out interaction** | A user-item or group-item edge used only as a ranking target at validation or test time. | Test edge (ambiguous with fold-in) |

## Model

| Term | Definition | Avoid |
| --- | --- | --- |
| **User preference encoder** | Maps a user's item history to a user embedding. The baseline is an MLP; the treatment is LightGCN. | |
| **MLP encoder** | The author GroupIM user encoder: a feed-forward net over the user's item vector. | Original encoder, vanilla encoder |
| **LightGCN encoder** | User encoder of ID embeddings propagated over the user-item graph for K layers, then the mean of the layer outputs. | GCN, NGCF, graph encoder (too generic) |
| **Aggregator** | How member embeddings become one group embedding. Canonical choices: `maxpool`, `meanpool`, `attention`. | Pooling, combiner, group encoder |
| **Uniform weights** | Aggregator that gives every member the same weight. | |
| **Cosine-similarity weights** | Aggregator that weights members by cosine similarity rather than learned attention. | |
| **Discriminator** | Network that scores agreement between a group embedding and a member embedding for mutual information. | Critic, MI head |
| **User pretraining** | Optimising the user preference encoder on user-item interactions (`L_U`) before group training. | Warm start, pretrain (unqualified) |
| **MI pretraining** | A short run that trains only the discriminator before full group training. | Pretrain (unqualified) |
| **Group loss (`L_G`)** | The group ranking / reconstruction objective on group-item interactions. | |
| **User loss (`L_U`)** | The individual user-item objective, scaled by λ when added to group training. | |
| **MI loss (`L_MI`)** | The mutual-information term between group embeddings and member embeddings. | |
| **Full model** | GroupIM with `L_G + λL_U + L_MI`, learned attention, and both pretraining stages. | |

## Evaluation

| Term | Definition | Avoid |
| --- | --- | --- |
| **NDCG@K (`N@K`)** | Normalised discounted cumulative gain at cutoff K. Headline cutoffs are 20 and 50. | |
| **Recall@K (`R@K`)** | Recall at cutoff K. Headline cutoffs are 20 and 50. | |
| **Coherence** | How similar the members of a group are in their individual preferences. | |
| **Aggregate diversity** | How spread-out the members' individual item histories are. | |
| **MI-variation** | Standard deviation of discriminator scores across members of one group. | |
| **User history length** | How many user-item interactions a user has. Used to test whether LightGCN gains concentrate on short-history users. | |

## Datasets

| Term | Definition | Avoid |
| --- | --- | --- |
| **Weeplaces** | POI check-in dataset shipped with the author GroupIM repo. The only reproduction baseline (ADR-0008). | |
| **Yelp** | Reconstructed variant, not the paper's benchmark. Yin et al. ICDE 2019 Yelp-LA: prebuilt groups, no timestamps, users and businesses with at least 5 distinct partners. | Calling it the paper's Yelp |
| **Gowalla** | Reconstructed variant. figshare US Food check-ins through June 2011. A group is one 15-minute outing, so each group has one item. Users and places with at least 5 distinct partners. | Calling an outing the paper's group |
| **Douban** | Reconstructed variant. Yin et al. ICDE 2019 Douban-Event. The item is an event, not a venue. Users and events with at least 10 distinct partners. | Calling an event a venue |
| **Table 1** | The GroupIM paper's dataset statistics. Weeplaces entity counts and averages match it. The other three datasets are compared against it and are not expected to match. | |
