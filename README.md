# groupim-lightgnc-ablation

GroupIM with a LightGCN user encoder: full ablation across Weeplaces, Yelp, Gowalla, and Douban.

Baseline code is the authors' [CrowdDynamicsLab/GroupIM](https://github.com/CrowdDynamicsLab/GroupIM) (SIGIR 2020; Python 3.6 / torch 1.1). This repo ports that stack to Python 3.12 + Torch 2.x via uv, then swaps the MLP encoder for LightGCN.

Papers: `papers/groupim.pdf`, `papers/lightgcn.pdf`.

Plan, CSV contract, experiment protocol, ADRs: [`docs/`](docs/README.md). Language: [`GLOSSARY.md`](GLOSSARY.md). Tickets: [GitHub Issues](https://github.com/DarkPhoenix645/groupim-lightgnc-ablation/issues).

## Setup

```bash
uv sync
uv run groupim-lightgnc-ablation
```

GPU: one RTX 4080. Use `uv add` / `uv run`; do not pip-install into a random venv.
