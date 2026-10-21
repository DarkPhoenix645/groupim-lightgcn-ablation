# Agent instructions

GroupIM + LightGCN ablation. Python via **uv** only (`uv add`, `uv sync`, `uv run`). Never pip/conda.

Read before exploring: `GLOSSARY.md`, `docs/adr/` for the area you are touching, `docs/data-contract.md`, `docs/experiments.md`. The week plan and issue map are `docs/scope.md`. `.scratch/` is gitignored local scratch — not source of truth, not for anything a later session needs.

Baseline to port: [CrowdDynamicsLab/GroupIM](https://github.com/CrowdDynamicsLab/GroupIM) (ADR-0001). LightGCN is an encoder swap only (ADR-0004).

Target hardware: one RTX 3080 (10 GB). Paper PDFs live in `papers/`. Work tracker: GitHub Issues via `gh`.

## Agent skills

### Issue tracker

Issues live in this repo's GitHub Issues. Use the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Domain docs

Single-context: `GLOSSARY.md` plus `docs/adr/` at the repo root. See `docs/agents/domain.md`.
