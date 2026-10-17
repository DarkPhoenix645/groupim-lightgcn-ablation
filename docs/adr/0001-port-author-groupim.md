# Port CrowdDynamicsLab/GroupIM rather than rewrite

The SIGIR 2020 authors already shipped a working stack (MLP encoder, aggregators, discriminator, loaders, train loop). We port [CrowdDynamicsLab/GroupIM](https://github.com/CrowdDynamicsLab/GroupIM) to this repo instead of reimplementing from the paper, because a from-scratch port would silently drift on negatives, padding, MI, and eval, and then we could not claim a fair LightGCN ablation against their MLP.

**Status:** accepted

**Considered options:** rewrite from the paper; vendor the repo as a submodule and wrap it. Rejected: a rewrite is unverifiable against their numbers; a permanent submodule keeps Python 3.6 / torch 1.1 in the critical path.
