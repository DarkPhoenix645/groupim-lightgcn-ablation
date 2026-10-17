# LightGCN replaces the user preference encoder only

The treatment is LightGCN as the user preference encoder (ID embeddings, K layers, mean of layer outputs, graph from train plus fold-in edges, one propagate per step). Aggregators (`maxpool` / `meanpool` / `attention`), the discriminator, `L_G` / `L_U` / `L_MI`, and both pretraining stages stay. That is the only way the ablation is "GroupIM + LightGCN" rather than a new model.

**Status:** accepted

**Considered options:** replace the whole GroupIM stack with a LightGCN group recommender; add LightGCN as a second parallel codebase. Rejected: neither yields a controlled encoder ablation.
