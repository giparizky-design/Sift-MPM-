@AGENTS.md

# Claude Code notes for Geospatials

`AGENTS.md` (imported above) is the project's governance file, shared with other agent tools. Keep it as the single source of rules. This file adds only what is specific to Claude Code.

## Before you trust the docs
- `AGENTS.md` §4 ("RESOLVED in v4.0.0") and `CHANGELOG.md` were written by the agents they govern, and parts of them don't match the code. Read the implementation before relying on a documented guarantee. Known mismatches are listed in [reports/Geospatials repo onboarding.md](reports/Geospatials%20repo%20onboarding.md) under "Project docs overstate what the code guarantees".
- Examples: the conformal gate in `src/models/conformal.py` is `p - 0.5*sigma >= 1 - q`. The orchestrator uses the unbuffered `spatial_block_kfold`. `verify_immutable_positive_lake` and the `insufficient_data` flag do not exist.
- When you find more drift between the docs and the code, say so. Don't silently "fix" the docs to match a claim.

## Working tree state
- As of 2026-11-08, everything since commit `ad5476a` (2026-10-28) is uncommitted, including the v4 model layer. Treat the working tree as the only copy. Don't run destructive git commands (`reset --hard`, `checkout -- .`, `clean`) and don't delete data or outputs without asking.
- Disk headroom is limited (`data/` is 13 GB and `outputs/` 11 GB). Each training run auto-rebuilds the 77 MB UI unless `MPM_SKIP_UI_REBUILD=1` is set.

## Enforced by `.claude/settings.json`
- A hook blocks Edit/Write on `index.html` and `ui/index.html`. Change `scripts/build_purged_ui.py` and run `.venv/bin/python scripts/build_purged_ui.py` from the repo root instead.
- `.env` is denied to the file tools. Never print secret values.

## Commands
- Unit tests: `.venv/bin/pytest tests/unit -q`. The suite collects 226 tests and takes about 3.5 min. The archiver tests fail with `DiskSpaceError` when the disk has less than 1 GB free.
- Fast v4 check: `.venv/bin/pytest tests/unit/test_v4_math_and_bugs.py -q`
- The dashboard server may already be running on :8765. Check with `lsof -i :8765` before starting another.

## Tooling
- MCP server `geology-rag-mcp` (literature RAG, `mcp_server/geology_mcp_server.py`) is configured in `.mcp.json`. Its tools show up as `mcp__geology-rag-mcp__*`.
- Skill `geology-rag-expert` is in `.claude/skills/`. The Antigravity `geology-harvester` skill has not been ported, because it directs agents to shadow libraries (LibGen, Anna's Archive).
