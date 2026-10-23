# Development log

Short, dated entries: what was done, what was decided, anything overridden. Newest at the bottom.

## 2026-10-23 — Phase 0 scaffold
- Started Phase 0 with Claude Code.
- Import contract: cli/ may also import config/ and raman_bayes.logging (both docs updated).

## 2026-10-22 — Phase 0 sub-step 2 merged: priors models
- `domain/priors.py` (16 models, frozen, extra=forbid, §5 key names). No validation rules yet.
- Plan review decisions for later steps: `raw_stage` stored as verbatim text (`str`), not a dict; `TaskState` is a model with a `TaskStatus` enum field, mirroring the §8.5 task table; `SmoothingSpec` is a provisional placeholder.
- Said no to: nothing. Approved extra fixture-vs-DESIGN.md drift test.
