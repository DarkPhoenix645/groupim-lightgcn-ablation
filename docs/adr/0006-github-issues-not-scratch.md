# GitHub Issues are the work tracker; `.scratch/` is local only

Work lives in this repo's GitHub Issues (`docs/agents/issue-tracker.md`). `.scratch/` is gitignored local scratch (the original HTML task list, one-off scripts). Agents must not treat `.scratch/` as source of truth or put anything there that a later session needs.

**Status:** accepted

**Considered options:** local markdown tickets under `.scratch/` (Matt Pocock local-tracker template). Rejected: we already publish with `gh`, and `.scratch/` will not be in commits.
