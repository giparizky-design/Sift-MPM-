# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

@AGENTS.md

# Claude Code notes for Geospatials

`AGENTS.md` (imported above) is the project's governance file, shared with other agent tools. Keep it as the single source of rules. This file adds only what is specific to Claude Code.

## Before you trust the docs
- `AGENTS.md` §4 ("RESOLVED in v4.0.0") and `CHANGELOG.md` were written by the agents they govern, and parts of them don't match the code. Read the implementation before relying on a documented guarantee. Known mismatches are listed in [reports/Geospatials repo onboarding.md](reports/Geospatials%20repo%20onboarding.md) under "Project docs overstate what the code guarantees".
- Examples: the conformal gate in `src/models/conformal.py` is `p - 0.5*sigma >= 1 - q`. The orchestrator uses the unbuffered `spatial_block_kfold`. `verify_immutable_positive_lake` and the `insufficient_data` flag do not exist.
- When you find more drift between the docs and the code, say so. Don't silently "fix" the docs to match a claim.
- `README.md` is stale: it still describes a "Pre-data" Yilgarn orogenic-gold pilot. The active work is Kalimantan K=4 metals and K=3 coal. `HANDOFF.md` is v2.0 from 2026-10-28 and predates the v4 model layer. For current state, read `audits/INDEX.md` and the newest audits.
- Test counts disagree across docs (`AGENTS.md` §5 says 184; the v4.13 changelog entry reports 285 passed). Report the count you actually get; don't edit one doc to match another.

## Architecture flow (read before changing the pipeline)
- Feature cube build (`scripts/fetch_real_layers.py` + `scripts/build_real_feature_cubes.py`) → `FeatureStack` `(C, H, W)` → flattened DataFrame → labels and PU negatives (`src/labels/`) → K=4 metal / K=3 coal ensembles (`src/models/metals.py`, `coal.py`) → calibration → conformal bounds → target polygons and site cards (`src/outputs/`) → UI bundle.
- `scripts/run_seven_pass_training.py` orchestrates all of it. It has two inference paths (in-memory, and chunked for >150k rows) that compute the lower bound differently. Change both or merge them, and test parity.
- `src/models/pu_pipeline.py` is only a re-export facade. Edit the underlying modules.

## Which data to trust
- Use the real-data-only cubes `data/kalimantan_real/processed` (island) and `data/east_kalimantan_real/processed` (EK window). Each has a `layer_provenance.json`, and the training and CV scripts call `assert_feature_provenance`.
- The older `data/kalimantan/processed` and the old EK cube contain seeded-random or heuristic layers. They pass only with `--allow-unverified-layers`; don't use them for new results.
- Never fill missing data with constants or random values; use NaN. Never add a layer without a provenance record.
- Audit v4.13 lists open label and leakage problems in the real cube (porphyry labels are mostly non-copper records, `dilation_tendency_Td` is a fault-Voronoi map, forest loss leaks into mine and propensity layers). Read it before any retrain.

## Working tree state
- As of 2026-11-08, everything since commit `ad5476a` (2026-10-28) is uncommitted, including the v4 model layer. Treat the working tree as the only copy. Don't run destructive git commands (`reset --hard`, `checkout -- .`, `clean`) and don't delete data or outputs without asking.
- Disk headroom is limited (`data/` is 13 GB and `outputs/` 11 GB). Each training run auto-rebuilds the 77 MB UI unless `MPM_SKIP_UI_REBUILD=1` is set.

## Enforced by `.claude/settings.json`
- A hook blocks Edit/Write on `index.html` and `ui/index.html`. Change `scripts/build_purged_ui.py` and run `.venv/bin/python scripts/build_purged_ui.py` from the repo root instead.
- `.env` is denied to the file tools. Never print secret values.

## Commands
- Unit tests: `.venv/bin/pytest tests/unit -q`. The suite takes about 3.5–6 min. The archiver tests fail with `DiskSpaceError` when the disk has less than 1 GB free.
- Fast v4 check: `.venv/bin/pytest tests/unit/test_v4_math_and_bugs.py -q`
- Single test: `.venv/bin/pytest tests/unit/<file>.py::<test_name> -q` (or `-k <pattern>`).
- `tests/integration/` (challenger tests) is not part of the unit command. Run it separately with `.venv/bin/pytest tests/integration -q`.
- The dashboard server may already be running on :8765. Check with `lsof -i :8765` before starting another.

## Tooling
- MCP server `geology-rag-mcp` (literature RAG, `mcp_server/geology_mcp_server.py`) is configured in `.mcp.json`. Its tools show up as `mcp__geology-rag-mcp__*`.
- Skill `geology-rag-expert` is in `.claude/skills/`. The Antigravity `geology-harvester` skill has not been ported, because it directs agents to shadow libraries (LibGen, Anna's Archive).
