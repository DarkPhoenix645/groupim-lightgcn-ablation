# GitHub Issues are the work tracker; `.scratch/` is local only

Work lives in this repo's GitHub Issues (`docs/agents/issue-tracker.md`): the epic, its sub-issues, and native blocked-by links. `.scratch/` is gitignored local scratch. An HTML task list used while the tickets were first written is not a tracker and is not cited from issues or from `docs/scope.md`. Agents must not treat `.scratch/` as source of truth or put anything there that a later session needs.

**Status:** accepted

**Considered options:** local markdown tickets under `.scratch/` (Matt Pocock local-tracker template). Rejected: we already publish with `gh`, and `.scratch/` will not be in commits.
