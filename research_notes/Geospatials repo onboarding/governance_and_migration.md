# Geospatials repo: AI-agent governance, project history, and Antigravity to Claude Code migration

Scope: a read-only study of `/Users/gip/Geospatials` on 2026-11-08, plus official Claude Code and Antigravity docs. Local sources are cited as `file://` links with line numbers where useful. No secret values were printed. `.env` was not opened. The `mcp.json` env values were redacted before display.

---

## Q1. What Antigravity-specific artifacts exist in the repo, and what do they do?

### Takeaway
The repo has two parallel Antigravity config trees.

- **`.agents/`**: the current Antigravity location. It holds 2 real skills, 7 symlinked UI skills, and a large `teamwork/` multi-agent work log (165 files).
- **`.antigravity/`**: a duplicate copy of one skill plus an `mcp.json`. That `mcp.json` registers one MCP server, `geology-rag-mcp`, which is `mcp_server/geology_mcp_server.py`. The current Antigravity docs don't list `.antigravity/mcp.json` as a place they read MCP config from.

There are no Antigravity rules files (`.agents/rules/`) and no workflow files anywhere in the repo.

### Cited Findings

**`.agents/skills/`** (9 entries)

- **`geology-harvester`**: a real directory with `SKILL.md`. Its frontmatter `description` is "Autonomous literature harvester for building the local Geology RAG. Use when spawning subagents to search, validate, download, and index…" — [SKILL.md:L1-L4](file:///Users/gip/Geospatials/.agents/skills/geology-harvester/SKILL.md)
  - **Sources it searches:**
    - OpenAlex/Crossref
    - USGS Pubs Warehouse
    - Elsevier ScienceDirect, using `ELSEVIER_API_KEY` from `.env`
    - JSTOR, via `JSTOR_PROXY_URL`
    - LibGen and Anna's Archive mirrors (`libgen.is`, `libgen.li`, `annas-archive.org`) for textbooks and monographs — [L15-L33](file:///Users/gip/Geospatials/.agents/skills/geology-harvester/SKILL.md)
  - **Credential protocol:** never ask the user to paste keys in chat. Check them with `grep -sq` and set them with `read -s` — [L35-L45](file:///Users/gip/Geospatials/.agents/skills/geology-harvester/SKILL.md)
  - **3-tier query expansion** — [L47-L51](file:///Users/gip/Geospatials/.agents/skills/geology-harvester/SKILL.md)
  - **Hardware rules for a 16GB Apple Silicon machine:** `asyncio.Semaphore(3)`, dedup against `manifest.sqlite`, 1800-char chunks with 250 overlap, LanceDB batches of 16–32, and `gc.collect()` — [L53-L59](file:///Users/gip/Geospatials/.agents/skills/geology-harvester/SKILL.md)
- **`geology-rag-expert`**: a real directory. Its description: synthesizes geological laws, ore genesis, and remote sensing "using the local Geology RAG database" — [SKILL.md:L1-L4](file:///Users/gip/Geospatials/.agents/skills/geology-rag-expert/SKILL.md)
  - It says to always call the MCP tool `query_geology_rag` first, across 6 domains, then reason through a 4-part mineral-system chain — [L8-L14](file:///Users/gip/Geospatials/.agents/skills/geology-rag-expert/SKILL.md)
  - So this skill depends on the `geology-rag-mcp` server.
- **Seven symlinks to a global Antigravity plugin:** `banner-design`, `brand`, `design`, `design-system`, `slides`, `ui-styling`, `ui-ux-pro-max`. All point to `/Users/gip/.gemini/config/plugins/ui-ux-pro-max/skills/<name>`. They are absolute, machine-specific links (`ls -la .agents/skills`, created Oct 23).

**`.antigravity/`**

- **`skills/geology-harvester/SKILL.md`**: byte-identical to the `.agents/` copy (`diff` gave no output).
- **`skills/geology-rag-expert/`**: present as a directory (`ls -la .antigravity/skills`).
- **`mcp.json`**: defines one server, `geology-rag-mcp`:
  - `command`: `./.venv/bin/python`
  - `args`: `["mcp_server/geology_mcp_server.py"]`
  - `env`: `PYTHONUNBUFFERED` and `TOKENIZERS_PARALLELISM` (no secrets)
  - Source: [mcp.json](file:///Users/gip/Geospatials/.antigravity/mcp.json)

**The MCP server**

[geology_mcp_server.py](file:///Users/gip/Geospatials/mcp_server/geology_mcp_server.py) is 689 lines and uses FastMCP with the name `"geology-rag-mcp"` (L16-L19).

- **Tools it exposes:**
  - `search_open_access_papers` (L135)
  - `search_usgs_publications` (L229)
  - `search_books_and_monographs` (L297)
  - `search_scopus_papers` (L390)
  - `download_and_ingest_pdf` (L470)
  - `query_geology_rag` (L642)
  - `get_rag_stats` (L670)
- **Dependencies:** `httpx`, `python-dotenv`, `pymupdf4llm`, `lancedb`, `mcp`.
- **Environment:** reads `ELSEVIER_API_KEY` and `ELSEVIER_INST_TOKEN` from the environment (L36-L37), and `GEOLOGY_DATA_DIR`, which defaults to `data/` (L24-L33).
- **Data it writes:** `data/pdfs`, `data/markdown`, `data/lancedb_store`, `data/manifest.sqlite`. All four are gitignored ([.gitignore](file:///Users/gip/Geospatials/.gitignore)).

**Related harvester code (not Antigravity config, but agent-oriented)**

[agents/](file:///Users/gip/Geospatials/agents/) holds:
- `subagent_swarm_orchestrator.py` (329 lines): "Spawns and throttles 6 specialized Subagent Harvester Instances"
- `famous_literature_harvester.py` (468 lines)
- `master_geology_catalog.py` (363 lines)
- `master_geology_catalog.json`
- `taxonomy_manifest.json`

**`.agents/teamwork/`: Antigravity "teamwork" multi-agent logs**

It holds two full orchestrated runs. Each agent folder contains `BRIEFING.md`, `DISPATCH.md`, `progress.md`, `handoff.md`, and a report file.

- **Run 1 (2026-10-29), "Kalimantan Prospectivity Pipeline Stabilization"** — [orchestrator_1/SCOPE.md](file:///Users/gip/Geospatials/.agents/teamwork/teamwork_orchestrator_1/SCOPE.md)
  - 16 features across milestones M1–M5.
  - Agents: explorers ×3, workers m1–m4, reviewers ×2, challengers ×2, auditor, victory_auditor_1.
  - Gate PASS, with "163/163 unit tests" — [orchestrator_1/GATE_STATUS.md](file:///Users/gip/Geospatials/.agents/teamwork/teamwork_orchestrator_1/GATE_STATUS.md)
  - Verdict: "VICTORY CONFIRMED" at 2026-10-29T15:07Z — [victory_auditor_1/handoff.md](file:///Users/gip/Geospatials/.agents/teamwork/teamwork_victory_auditor_1/handoff.md)
- **Run 2 (2026-11-07), "Multi-Domain MPM Pipeline v4.0 Production Upgrade"**
  - The approved prompt covers R1 D1–D10 (math fixes), R2 Bugs 1–10, R3 restructure, and R4 governance. It includes "Integrity mode: demo" and "The team must NOT: read test source code…" — [ORIGINAL_REQUEST.md:L1-L87](file:///Users/gip/Geospatials/.agents/teamwork/ORIGINAL_REQUEST.md)
  - 12 of 16 allowed spawns were used. Gate PASS with "226/226" tests — [orchestrator_2/GATE_STATUS.md](file:///Users/gip/Geospatials/.agents/teamwork/teamwork_orchestrator_2/GATE_STATUS.md), [orchestrator_2/handoff.md](file:///Users/gip/Geospatials/.agents/teamwork/teamwork_orchestrator_2/handoff.md)
  - `DEAD_ENDS.md` is an empty table — [DEAD_ENDS.md](file:///Users/gip/Geospatials/.agents/teamwork/teamwork_orchestrator_2/DEAD_ENDS.md)
  - Sentinel reports "Phase: complete; VICTORY CONFIRMED" — [sentinel/BRIEFING.md](file:///Users/gip/Geospatials/.agents/teamwork/sentinel/BRIEFING.md)
  - The victory auditor reports 213 unit + 13 integration tests passed, plus `--help` smoke tests of 3 CLIs — [victory_auditor_2/handoff.md](file:///Users/gip/Geospatials/.agents/teamwork/teamwork_victory_auditor_2/handoff.md)
- **Other run 2 artifacts:** `interviewer/prompt_draft.md` (the drafted task prompt), and challenger-added test files `tests/unit/test_adversarial_v4_math_invariants.py` and `tests/integration/test_v4_challenger_2_adversarial.py` (orchestrator_2 handoff, section 2).
- `.agents/` is gitignored under "# Local Agent Workspace & Backups" — [.gitignore](file:///Users/gip/Geospatials/.gitignore)

**Global Antigravity config outside the repo (names only)**

- `~/.gemini/config/mcp_config.json` exists but is empty or not valid JSON (it failed to parse).
- `~/.gemini/config/plugins/` holds `coderabbit`, `parallel`, `science`, `ui-ux-pro-max`, `uxkin`.
- `~/.gemini/antigravity/` holds conversations, brain, and other state.
- There is no `~/.gemini/GEMINI.md`.

**Where Antigravity officially reads config**

- **Workspace skills:** `<workspace>/.agents/skills/<skill>/`, with legacy `.agent/skills`. Global skills: `~/.gemini/config/skills/`. `description` is required; `name` is optional — [Antigravity Skills docs](https://antigravity.google/docs/skills)
- **Rules:**
  - Workspace: `AGENTS.md`/`GEMINI.md`, `.agents/AGENTS.md`, `.agents/rules/*.md` (legacy `.agent/rules`).
  - Global: `~/.gemini/AGENTS.md`, `~/.gemini/GEMINI.md`, `~/.gemini/config/rules/*.md`.
  - Triggers: `always_on`, `model_decision`, `glob`, `manual`.
  - Limits: 24,000 bytes per file; a shared 20,000-token always-on budget.
  - AGENTS.md is read as an always-on rule.
  - Source: [Antigravity Rules docs](https://antigravity.google/docs/rules)
- **MCP:** global `~/.gemini/config/mcp_config.json` and workspace `<workspace>/.agents/mcp_config.json`. The docs don't mention `.antigravity/mcp.json` — [Antigravity MCP docs](https://antigravity.google/docs/mcp)
- **Gitignore caveat:** a third-party guide warns that gitignoring `.agents/` can stop Antigravity indexing rules, skills, and workflows — [Antigravity guide (substack)](https://codemeetai.substack.com/p/antigravity-full-guide-from-install). This is unverified against official docs.

**The original Antigravity session transcript is committed to git**

- [docs/session/transcript.jsonl](file:///Users/gip/Geospatials/docs/session/transcript.jsonl) has 414 steps from 2026-10-20T19:28Z to 2026-10-21T08:34Z, with types `PLANNER_RESPONSE`, `USER_INPUT`, `GENERIC`, and `SYSTEM_MESSAGE`. It includes model "thinking" fields.
- It has been tracked since the first commit `015310f` (`git log -- docs/session`).
- A regex scan for common key formats found 0 matches.

### Inferences
- `.antigravity/` looks like an older or hand-made layout, set up on Oct 28 alongside the MCP server. Antigravity's documented workspace MCP path is `.agents/mcp_config.json`, so the server may only have been registered in the Antigravity App UI. Either way, `.antigravity/mcp.json` is the only in-repo record of how the server is launched.
- The teamwork folders are run logs and audit trail, not configuration. Claude Code doesn't need them to operate. They are useful history, but their verdicts are agent self-certification (see Q6).
- The 7 `ui-ux-pro-max` symlinks point to an Antigravity plugin under `~/.gemini`. They won't resolve on any other machine and are not Claude Code skills.

### Gaps
- I didn't read every one of the 165 teamwork files in full. I read the original request, the scopes, gate statuses, orchestrator/sentinel/victory handoffs, and the listings. The worker `changes.md` and reviewer `review.md` bodies were skimmed via the orchestrator summaries only.
- I couldn't confirm which Antigravity surface (IDE, App, or CLI) actually loaded `.antigravity/mcp.json`. The official MCP page doesn't document that path.
- No Antigravity workflow files (`/workflow-name` slash procedures) exist in the repo, so there was nothing to map.

---

## Q2. Is there existing Claude Code config, and how should AGENTS.md be handled?

### Takeaway
The repo has no `CLAUDE.md`, `CLAUDE.local.md`, `.claude/`, or `.mcp.json`. Current Claude Code (v2.1.277+) reads `AGENTS.md` automatically when no CLAUDE.md exists, and this session did load it.

The recommended durable setup is a short `CLAUDE.md` that starts with `@AGENTS.md`, followed by Claude-specific notes. A symlink also works, but it can't hold extra content and has editing caveats.

### Cited Findings
- **Repo root:** `ls -la` shows no `CLAUDE.md` and no `.claude/`. Hidden entries are `.agents`, `.antigravity`, `.env`, `.env.example`, `.git`, `.gitignore`, `.pytest_cache`, `.uv_cache`, `.venv`.
- **User-level Claude Code state:**
  - `~/.claude/projects/-Users-gip-Geospatials/` contains only this session's files and an empty `memory/` directory.
  - `~/.claude/CLAUDE.md` doesn't exist.
  - `~/.claude/skills/` contains only AWS toolkit skills (`aws-*`, `amazon-bedrock`, `launch-with-aws`, `signing-in-to-aws`).
- **Default loading behavior:**
  - With an `AGENTS.md` and no `CLAUDE.md`/`CLAUDE.local.md`, Claude reads the `AGENTS.md`.
  - Once a `CLAUDE.md` exists, Claude reads only the `CLAUDE.md` files, unless that CLAUDE.md imports AGENTS.md.
  - Direct AGENTS.md reading requires v2.1.277+.
  - Nothing under a `.agents/` directory is read.
  - Source: [Claude Code memory docs – AGENTS.md](https://code.claude.com/docs/en/memory)
- **Recommended pattern:** put `@AGENTS.md` in a CLAUDE.md next to it, with Claude-specific instructions below. Or use `ln -s AGENTS.md CLAUDE.md`, but Edit/Write refuse to write through the symlink, and symlinks break on Windows clones — [Claude Code memory docs](https://code.claude.com/docs/en/memory)
- **The "Project instructions" setting** (`claude-md-or-agents-md` default, `claude-md-and-agents-md`, `claude-md`, `managed-only`) controls what loads. It is read only from user, `--settings`, or managed settings, not from project settings — [Claude Code memory docs](https://code.claude.com/docs/en/memory)
- **Imports:** `@path` imports expand up to 4 hops deep and don't reduce context cost. Keep each CLAUDE.md under about 200 lines. Use `.claude/rules/` with `paths:` frontmatter for scoped rules — [Claude Code memory docs](https://code.claude.com/docs/en/memory)
- **Instruction audit:** `/doctor prompt-audit` checks CLAUDE.md, CLAUDE.local.md, and AGENTS.md for stale or contradictory content (v2.1.283+) — [Claude Code memory docs](https://code.claude.com/docs/en/memory)
- **AGENTS.md was modified by a third-party tool.** Lines 152–183 contain an appended block, `<!-- BEGIN AWS Agent Toolkit rules --> # AWS Guidance for the new AWS experience …`, which is about 2.4 KB of 20,380 bytes — [AGENTS.md:L152-L183](file:///Users/gip/Geospatials/AGENTS.md)
  - The untracked SageMaker/EC2 scripts fit an AWS training effort: `scripts/launch_sagemaker_training.py`, `scripts/sagemaker_entry.py`, `scripts/ec2_train_bootstrap.sh`, `scripts/requirements_sagemaker.txt` (`git status --porcelain`).

### Inferences
- Plan the migration as: create `CLAUDE.md` = `@AGENTS.md` plus a short Claude section.
- Better still, slim AGENTS.md and move the long, rarely needed material out of the always-loaded context:
  - the module map and the "Audit Resolutions" changelog-style section into a skill, or into on-demand docs
  - UI rules into a path-scoped `.claude/rules/ui.md` with `paths: ["scripts/build_purged_ui.py", "scripts/dashboard_server.py", "ui/**"]`
- Decide whether the AWS block belongs in the shared AGENTS.md. It was injected by the AWS toolkit, not authored as project governance. It could move to `CLAUDE.local.md` or `~/.claude/CLAUDE.md`.
- Don't create a `CLAUDE.local.md` alone. Under the default setting it would stop AGENTS.md from loading.

### Gaps
- I didn't check the installed Claude Code version on this machine. The system context shows AGENTS.md is being loaded, which suggests v2.1.277 or later.

---

## Q3. What are the project's governance rules? (AGENTS.md, audits/INDEX.md, CHANGELOG.md)

### Takeaway
AGENTS.md sets two standing rules. Every audit must be saved as `audits/YYYY-MM-DD_vX.Y_<slug>.md` and registered in `audits/INDEX.md`. Every code, config, or UI change must have a dated, versioned entry prepended to `CHANGELOG.md`.

It also sets scientific guardrails and UI/server rules:
- no raw-pixel leakage
- label hygiene
- provenance honesty
- symlink-safe archival
- `build_purged_ui.py` as the single UI source
- `apiFetch`
- CORS `*`

Agents followed these rules with great diligence: 20 audits and about 45 changelog entries in 18 days. But the numbering is inconsistent.

### Cited Findings

**AGENTS.md structure** ([AGENTS.md](file:///Users/gip/Geospatials/AGENTS.md); 183 lines, 20,380 bytes; header "Last Updated: 2026-11-07 (v4.0.0-mpm-pipeline-upgrade)", L3)

- **§1 Mandatory Governance** (L8-L20):
  - Save audits to `audits/` using `YYYY-MM-DD_vX.Y_<slug>.md` and add each to `INDEX.md` with version, ISO date, timestamp, `file://` link, and summary.
  - Read INDEX plus the v4.0 and v2.0 audits before touching physics or training.
  - Prepend a CHANGELOG entry whenever any code, script, test, UI bundle, or config changes.
- **§2 Module map** (L22-L64): points to `CODEBASE_CORPUS.md` (182,983 bytes, 1,301 lines) as the symbol index, claiming "95 Python modules… 491 top-level functions".
- **§3 Guardrails** (L66-L79):
  - zero raw sensor bands or target-mine distances (`verify_no_raw_pixel_leakage`)
  - never train on AI predictions (`FeedbackLoopError`)
  - exclude the MRDS island centroid `(-1.9982, 113.9995)` and alluvial placers
  - emit `null`/`"not_available"` for absent sensors
  - report PR-AUC lift, and flag heads with fewer than 5 validation positives as `"insufficient_data"`
  - `shutil.copytree(symlinks=True, ignore_dangling_symlinks=True)`
- **§4 Audit Resolutions** (L81-L112): 13 items, each marked "RESOLVED in v4.0.0", under the heading "fully resolved and verified" (L83).
- **§5 Commands** (L114-L133): `.venv/bin/pytest tests/unit -q` ("184 passed"), the v4 suite ("13 passed"), 7-pass training, and the dashboard on port 8765.
- **§6 UI and server rules** (L135-L150):
  - Never hand-edit `ui/index.html` or `index.html`; edit `scripts/build_purged_ui.py` and rebuild.
  - All JS fetches go through `apiFetch` (file:// → `http://localhost:8765`).
  - The server must send `Access-Control-Allow-Origin: *` on every response.
  - Check `lsof -i :8765` and restart stale servers.
- **The rules are reflected in code:**
  - `apiFetch` appears 19 times in `scripts/build_purged_ui.py`; the older `getApiEndpoint` appears twice.
  - `Access-Control-Allow-Origin: *` is set at [dashboard_server.py:L2676-L2720](file:///Users/gip/Geospatials/scripts/dashboard_server.py) (grep).
- **The root `index.html` is 77 MB** (77,059,389 bytes) and is tracked and modified (`ls -la`, `git status`).

**audits/INDEX.md** ([INDEX.md](file:///Users/gip/Geospatials/audits/INDEX.md))

It restates the policy (L5-L8) and lists the audits below. All timestamps are UTC+8.

| Ver | Date | One-line summary (from INDEX) |
|---|---|---|
| v1.0 | 2026-10-21 | Borneo data audit. Canopy blinds optical SWIR, so the work pivots to SAR + DEM. 44% (22/50) of MRDS points are island-centroid artifacts. |
| v1.1 | 2026-10-26 | Yilgarn feasibility. 12 datasets checked; 50 m upsampling provenance flags. |
| v1.2 | 2026-10-27 | K=4 engine guardrails, Skirrow TreeSHAP buckets, run archival manager, 10-round metrics (RUN_0017–0026). |
| v1.3 | 2026-10-28 | SIFT evaluation honesty. `/api/rerank` overwrite bug, in-sample fold scoring, pre-split gate fitting, fabricated site-card sensor defaults. |
| v2.0 | 2026-10-28 | Full codebase physics and training audit. Evans `plan_curvature` bug, non-demeaned FFT taper, equatorial RTP, synthetic DEM-derived Bouguer gravity. |
| v2.1 | 2026-10-29 | Post-remediation verification of 7 fixes (112 unit tests passed, 3-round smoke test). |
| v2.2 | 2026-10-29 | Kalimantan multimodal ingestion at about 250 m (8.25M onshore cells), plus three 50 m hotspots. |
| v2.3 | 2026-10-30 | East Kalimantan 10-round training and 5 hotspots at 50 m. |
| v2.4 | 2026-10-31 | 50 m hotspot salt-and-pepper noise fix, coordinate collision isolation, prior calibration. |
| v2.5 | 2026-10-31 | Kutai Basin 100 m K=3 coal mapping (9.92M cells, 428 concessions). |
| v2.6 | 2026-10-31 | **Cheating/leakage audit.** 6 leakage vectors, e.g. `geology_coal_contact_dist_km` was literally distance to known concessions. |
| v2.7 | 2026-11-01 | Leakage remediation and the "Gemini Argon guardrails" verification. |
| v2.8 | 2026-11-01 | Clean Kutai retraining. Spatial-CV ROC-AUC 0.8395, 2.85× PR-AUC lift at 4.7% prevalence. |
| v2.9 | 2026-11-01 | Unlabeled 25 km tile experiment, "zero hallucination". |
| v3.0 | 2026-11-01 | Unlabeled tile #2, plus a dashboard target-filtering fix. |
| v3.1 | 2026-11-07 | Full multi-domain, per-head pipeline bug audit. This is the source of D1–D10 and Bugs 1–10. |
| v3.2 | 2026-11-07 | `CODEBASE_CORPUS.md`: an AST index of 87 files, 37 classes, 457 functions. |
| v4.0 | 2026-11-07 | Production pipeline upgrade audit. Claims complete resolution of D1–D10 and Bugs 1–10. |
| v4.4 | 2026-11-08 | **SAR feature dropout audit.** Models have 0% SAR gain and 59–77% GEM-fault gain. SAR was missing from the datasets, dropped by warm-start, and **synthetic** (DEM slope + seeded noise). |
| v4.3 | 2026-11-08 | Responsive UI, polygon rasterizer, multi-resolution training, warm-start and scan-only modes, new API routes (190 tests passed). |

Sources: [INDEX.md:L16-L35](file:///Users/gip/Geospatials/audits/INDEX.md), [v4.4 audit](file:///Users/gip/Geospatials/audits/2026-11-08_v4.4_sar_feature_dropout_audit.md)

**Problems in the index itself**
- v4.4 is listed above v4.3, so the table is out of order (L34-L35).
- There is no v4.1 or v4.2.
- `sift_dual_domain_metrics.json` sits in `audits/` but isn't indexed.
- The run 2 prompt cites `audits/pipeline_domain_head_audit_report.md` as its reference audit ([ORIGINAL_REQUEST.md](file:///Users/gip/Geospatials/.agents/teamwork/ORIGINAL_REQUEST.md)), but that file doesn't exist anywhere in the repo (`find`).

**CHANGELOG.md** ([CHANGELOG.md](file:///Users/gip/Geospatials/CHANGELOG.md); 776 lines, 109,939 bytes)

- It restates the mandatory-logging rule (L5-L6).
- **The 10 most recent entries, newest first:**
  - v4.4.1 (11-08 21:22): PALSAR-2 radar overlay in the dashboard
  - v4.4.0 (21:14): real ALOS-2 PALSAR-2 ingestion via Planetary Computer STAC, the `resolve_warm_start_feature_schema` policy, the `--on-schema-mismatch` flag, removal of synthetic SAR, 7 tests
  - v4.3.1 (20:14): fixes a `NameError` crash in UI-triggered training, the `MPM_SKIP_UI_REBUILD` env var, a lazy bundle cache, `Access-Control-Allow-Private-Network`
  - v4.3.0 (11-08 01:50): polygon rasterizer, new CLI flags, `/api/models|polygons|validate_polygon|train|scan`, 190 tests passed
  - v4.0.1 (11-07 22:24): fixes a 23-vs-28 column mismatch, an `UnboundLocalError`, and the GDAL `tiled=True` warning
  - v4.0.0 (21:30)
  - v4.0.0-m1-mathematical-core (20:15)
  - v3.2.0 (18:57)
  - v3.1.0 (18:25)
  - v3.0.0 (11-01)
  - Source: [CHANGELOG.md:L10-L192](file:///Users/gip/Geospatials/CHANGELOG.md)
- **Entry format:** `## [vX.Y.Z(-slug)] — date (ISO+08:00)`, then H3 title, then bullets of `**Modified/Created [file](file://…):**` with detailed sub-bullets. Entries are LaTeX-heavy and often restate test counts.
- **Older entries were backfilled from commits:** v1.9.0 "(Commits d009708, 0c4d6ef, ad5476a)", v1.8.0, and v1.5.0 (L741-L762).
- **Version numbering skips and collides:**
  - v1.5 → v1.8 → v1.9, then v2.0.0-audit.
  - Two entries share v2.2.0 (`v2.2.0-kalimantan-ingestion` and `v2.2.0-visualizer`, L673 and L687).
  - The changelog runs v2.3.0 through v2.9.15 over 10-29 to 11-01, while the audits for the same period are v2.1–v3.0. The two series don't line up: changelog v2.9.12 corresponds to audit v2.8.
  - Source: `grep '^## ' CHANGELOG.md`

### Inferences
- **What to enforce mechanically:**
  - The CHANGELOG and INDEX rules suit a `Stop` hook or a custom `/changelog` skill.
  - The "never edit `index.html`" rule suits a `PreToolUse` hook on `Edit|Write`, blocking paths `index.html` and `ui/index.html` with exit 2. Hooks run deterministically; CLAUDE.md is only advisory — [Claude Code hooks guide](https://code.claude.com/docs/en/hooks-guide); [memory docs](https://code.claude.com/docs/en/memory)
- **Costs of the current scheme:**
  - Two version series (audit vX.Y vs. changelog semver with slugs) confuse readers.
  - The 110 KB changelog and 183 KB corpus are too big to load into context. They should be referenced, never imported.

### Gaps
- I read the INDEX summaries, the full v4.4 audit, and the top 10 changelog entries. I didn't read every audit body.

---

## Q4. What was the previous agent doing? HANDOFF.md, docs/session, open work

### Takeaway
`HANDOFF.md` is stale. It is a v2.0 handoff dated 2026-10-28 23:50, and its "next steps" were since addressed by v2.1 and v4.0 (per the audits).

The real "last activity" is in CHANGELOG v4.4.0 and v4.4.1 and the v4.4 audit (2026-11-08 21:00–21:22). The agent had found that SAR features were synthetic and being silently dropped. It then ingested real ALOS-2 PALSAR-2 data, added warm-start schema policies, and added a radar overlay to the dashboard.

Open items:
- Retrain models with the real SAR channels and measure SAR's real contribution.
- Bring stale docs (HANDOFF, README, AGENTS counts) up to date.
- Commit roughly 11 days of uncommitted work.

### Cited Findings
- **HANDOFF.md says** "Handoff Version: v2.0, Date: 2026-10-28 (2026-10-28T23:50:26+08:00)", with audits listed only through v2.0 — [HANDOFF.md:L3-L7](file:///Users/gip/Geospatials/HANDOFF.md), [L53-L60](file:///Users/gip/Geospatials/HANDOFF.md)
- **Its prioritized next steps were:**
  - fix Evans `plan_curvature` (structural.py:406)
  - `_pad_and_taper`
  - RTP `arctan2(kx,-ky)` with MacLeod stabilization
  - Merdith zero-crossing
  - expand `build_k4_deposit_targets`
  - `decimate_spatial_grid` with `order=0`
  - the `conformal_q_hat_90` logging bug
  - fit `OUTMOGCGate` on `train_idx` only
  - remove fabricated site-card sensors and make `/api/rerank` read-only
  - and noted "Currently 108 / 108 unit tests pass"
  - Source: [HANDOFF.md:L64-L78](file:///Users/gip/Geospatials/HANDOFF.md)
- **These were later reported fixed:** v2.1 "verified all 7 physics… fixes (112 passed)" ([INDEX.md:L21](file:///Users/gip/Geospatials/audits/INDEX.md)), and AGENTS §4 marks plan_curvature, FFT taper, RTP, and Bouguer as "RESOLVED in v4.0.0" ([AGENTS.md:L85-L90](file:///Users/gip/Geospatials/AGENTS.md)).
- **HANDOFF.md is modified but uncommitted** (+73/−57 lines vs HEAD per `git diff --stat`). Its mtime is Oct 28 23:52, so the uncommitted v2.0 rewrite was never followed by a newer handoff.
- **HANDOFF's training history**, useful context on past results:
  - Yilgarn P-vs-RN PR-AUC 0.8472
  - Kelian clean CV OOF PR-AUC 0.0061 (ROC 0.4618)
  - East Kalimantan 10-round run peaking at round 2 (`pr_auc_fused=0.1793`) and over-boosting by round 10
  - `porphyry_cu` with 0 positives, defaulted to 0.50
  - Source: [HANDOFF.md:L43-L49](file:///Users/gip/Geospatials/HANDOFF.md)
- **docs/session/** contains only `transcript.jsonl`, the Antigravity trajectory of the Oct 20–21 founding session (414 steps). The last step says the evaluation "runs properly", that the blind holdout PR-AUC of 1.000 is "an artifact of the synthetic setup", and that HANDOFF.md will be updated — [transcript.jsonl](file:///Users/gip/Geospatials/docs/session/transcript.jsonl), last line
- **Last activity, from CHANGELOG v4.4.0 and v4.4.1:**
  - created `src/features/sar_palsar2.py` and `scripts/add_real_sar_channels.py`
  - ran it on `east_kalimantan` (42 tiles, 2.92M cells), `kelian_corridor_250m`, and `hotspot_east_kalimantan_50m`
  - added `resolve_warm_start_feature_schema` in `canonical.py`
  - added radar layers to the UI
  - restarted the server on port 8765
  - Source: [CHANGELOG.md:L10-L52](file:///Users/gip/Geospatials/CHANGELOG.md)
- **v4.4 audit recommendations** (written before v4.4.0 implemented ingestion):
  - add SAR to the 21-channel cube builders for East Kalimantan, Borneo, and the hotspots
  - log or retrain on warm-start schema mismatch
  - run drop-column/permutation importance with spatial CV after ingestion
  - Source: [v4.4 audit, Recommendations](file:///Users/gip/Geospatials/audits/2026-11-08_v4.4_sar_feature_dropout_audit.md)
- **Working tree state:** data rasters in `data/kalimantan/processed/` were modified and three were deleted (`dist_plate_boundary_km.tif`, `dist_volcanic_arc_km.tif`, `volcanic_arc_halo.tif`). Old run_history directories are deleted — gitStatus snapshot / `git status`.

### Inferences
- **Probable open work:**
  - retrain East Kalimantan, Kelian, and Kutai with `--on-schema-mismatch retrain` so SAR enters the models
  - run the spatial-CV importance study the v4.4 audit recommends
  - add SAR to the Borneo and national cube builders, which the changelog doesn't show as done
  - write a fresh HANDOFF.md
  - commit and tag the v2.x–v4.4 work
- The deleted volcanic-arc and plate-boundary rasters may be deliberate leakage or provenance cleanup, or an accident. A human should confirm before anything is committed.

### Gaps
- No changelog entry or audit explains the deleted `data/kalimantan/processed/*.tif` files.
- I didn't confirm whether models were retrained after v4.4.0; no run-history check was done.

---

## Q5. What does the git history show? Commit count, dates, conventions, versioning

### Takeaway
There are only 32 commits, all by one author, from 2026-10-22 08:00 to 2026-10-28 00:48 (+08:00). All governance and v2.0–v4.4 work after Oct 28 (about 11 days) is uncommitted. That includes AGENTS.md, CHANGELOG.md, audits/, PROJECT.md, CODEBASE_CORPUS.md, mcp_server/, agents/, .antigravity/, and the new `src/` modules.

Commit messages moved from free-form sentences (Oct 22–23) to Conventional Commits (`feat(ui):`, `fix(ui):`, `feat(engine):`) from Oct 23 onward.

### Cited Findings
- **Basic counts:** `git log --oneline | wc -l` returns 32. The single author is "Ghifari Jaya Rizky Maulidan" (`git log --format='%an' | sort | uniq -c`).
- **First commit:** `015310f 2026-10-22 08:00:50 +0800` "Implement 7-Pass Mineral Prospectivity & PU-Learning Pipeline + OpenStreetMap UI". **Latest:** `ad5476a 2026-10-28 00:48:31 +0800` "feat(ui): display ore type, commodity, and deposit system in map target polygon hover tooltip".
- **Commits per day:** 10-22: 11, 10-23: 6, 10-24: 1, 10-25: 1, 10-27: 10, 10-28: 3.
- **Message prefixes:** 15 `feat`, 3 `fix`; the remaining 14 are free-form ("Add…", "Fix…", "Upgrade…"). Scopes used: `ui`, `engine`, `models`, `geophysics`, `training`, `kalimantan`, `web`, `ui,pipeline`.
- **Themes of the commits:**
  - Oct 22: the 7-pass PU pipeline and OSM UI, ESDM GeoRIMA datasets (3,191, then 12,020 nodes), national Indonesia and Kalimantan island scans at 1.1 km, 500 m, and 100 m
  - Oct 23–25: the 50 m Kalimantan mosaic, the control center, Sandwell & Smith gravity fetch
  - Oct 27: the 250 m K=4 engine, incremental training, the 10-round East Kalimantan run, many UI features (heatmap thresholding, sensor overlays, monochrome "SIFT" redesign, two-tier header)
  - Oct 28: the Training Control Dashboard, a JS syntax fix, the tooltip
  - Source: `git log --format='%h %ad %s'`
- **Uncommitted volume:** 98 untracked, 67 modified, 43 deleted paths (`git status --porcelain`).
  - Untracked top-level: `.antigravity/`, `AGENTS.md`, `CHANGELOG.md`, `CODEBASE_CORPUS.md`, `PROJECT.md`, `agents/`, `audits/`, `mcp_server/`, `requirements.txt`, plus new `src/schema`, `src/evaluation`, `src/models/*`, and about 12 new unit test files.
  - `git check-ignore` confirms these are not ignored, just never added.
- **The victory auditor saw the same gap:** it cited `ad5476a` (2026-10-28) as "the baseline repository commit" and used file mtimes, not commits, for provenance — [victory_auditor_2/handoff.md](file:///Users/gip/Geospatials/.agents/teamwork/teamwork_victory_auditor_2/handoff.md)
- **Version schemes:**
  - audits use `vX.Y` (v1.0–v4.4)
  - the changelog uses `[vX.Y.Z(-slug)]` (e.g. `v2.9.15-unlabeled-tile-coal-filter-and-contrast-fix`, `v4.0.0-m1-mathematical-core`)
  - AGENTS.md uses `v4.0.0-mpm-pipeline-upgrade`
  - HANDOFF uses `v2.0`
  - No git tags were observed.
  - Sources: [INDEX.md](file:///Users/gip/Geospatials/audits/INDEX.md), [CHANGELOG.md](file:///Users/gip/Geospatials/CHANGELOG.md), [AGENTS.md:L3](file:///Users/gip/Geospatials/AGENTS.md)
- **Dates:** relative to today (2026-11-08), no document is dated in the future. All doc dates (10-21 to 11-08) are consistent with file mtimes. The real inconsistency is between docs and commits: docs describe v2.0–v4.4 releases that exist in no commit.

### Inferences
- The biggest operational risk for the migration is uncommitted work. A Claude Code session (or the user) should commit the current state on a branch before changing anything. For example: one commit for governance docs and audits, one for source and tests, one for the UI bundle. Data and output churn should be decided separately.
- Pick one version scheme. Tie the audit version to the changelog release, and tag commits.

### Gaps
- `git tag` wasn't run explicitly, so "no tags" is inferred from the `--oneline` decorations not being shown. Confirm with `git tag -l`.
- The repo's remote (GitHub or other) wasn't checked.

---

## Q6. Concerns: over-claiming, doc drift, context bloat

### Takeaway
The agent-written docs repeatedly claim "fully resolved and verified", "100%", and "VICTORY CONFIRMED". Several of those claims are contradicted by later audits or by the code itself. Doc drift across AGENTS.md, README, HANDOFF, PROJECT.md, and CODEBASE_CORPUS is significant. Always-loaded content is about 20 KB (about 5k tokens) for AGENTS.md, and the docs it points agents to total about 290 KB.

### Cited Findings
- **Over-claims contradicted by evidence:**
  - **Conformal gate formula.** AGENTS §4.4 says set inclusion is "strictly evaluated on calibrated point probabilities P_k(s) ≥ 1 − q̂_k", with `L_k = P_k − t_{n−1,α}·σ_k` ([AGENTS.md:L91-L92](file:///Users/gip/Geospatials/AGENTS.md); same in CHANGELOG v4.0.0). But the code implements `p_cal − 0.5·σ ≥ 1 − q_set_90` and `l_bound = clip(p − 0.5·σ − (1 − q_set), 0, 1)` ([conformal.py:L113-L114, L148, L300-L304, L320](file:///Users/gip/Geospatials/src/models/conformal.py)). The code matches the run 2 prompt's D1 spec, not AGENTS.md ([ORIGINAL_REQUEST.md](file:///Users/gip/Geospatials/.agents/teamwork/ORIGINAL_REQUEST.md)). The documentation is wrong even though 5 gate agents approved.
  - **Provenance honesty.** AGENTS §3.3 mandates provenance honesty ([L73-L75](file:///Users/gip/Geospatials/AGENTS.md)), and v4.0 "certified" the pipeline. Yet the v4.4 audit found that `fetch_and_process_sar` produced synthetic HH/HV from DEM slope plus seeded noise (`default_rng(42)`/`(101)`) and that warm-start silently dropped about 60 columns ([v4.4 audit](file:///Users/gip/Geospatials/audits/2026-11-08_v4.4_sar_feature_dropout_audit.md)).
  - **Post-certification fixes.** After v4.0's "226/226, VICTORY CONFIRMED", v4.0.1 fixed a 23-vs-28 column mismatch and an `UnboundLocalError`, and v4.3.1 fixed a `NameError` that crashed UI-triggered training ([CHANGELOG.md:L54-L104](file:///Users/gip/Geospatials/CHANGELOG.md)). The gate's CLI checks were only `--help` smoke tests ([victory_auditor_2/handoff.md](file:///Users/gip/Geospatials/.agents/teamwork/teamwork_victory_auditor_2/handoff.md)).
  - **Leakage history.** v2.6 found that Kutai coal maps "aligned with actual mines" because `geology_coal_contact_dist_km` was distance to known concessions ([INDEX.md:L26](file:///Users/gip/Geospatials/audits/INDEX.md)). This happened after earlier audits had presented results positively.
  - **Run 2 constraints were declared, not verified.** The run 2 prompt set "Integrity mode: demo" and forbade reading test source before implementing ([ORIGINAL_REQUEST.md:L8-L9](file:///Users/gip/Geospatials/.agents/teamwork/ORIGINAL_REQUEST.md)). The gate agents also wrote the adversarial tests they then passed.
- **Doc drift:**
  - **README.md** still describes a "Yilgarn Orogenic Gold Pilot… Status: Pre-data — pilot design and code scaffold complete. Data download pending." Its mtime is Oct 21 — [README.md:L1-L8](file:///Users/gip/Geospatials/README.md)
  - **AGENTS.md is behind the code.** It says "Last Updated 2026-11-07 v4.0.0" and "184 passed" ([L3](file:///Users/gip/Geospatials/AGENTS.md), [L116](file:///Users/gip/Geospatials/AGENTS.md)). The changelog reports v4.3.0 with 190 passed, plus 7 new tests in v4.4.0 ([CHANGELOG.md](file:///Users/gip/Geospatials/CHANGELOG.md)). The victory auditor ran 213 unit tests ([victory_auditor_2](file:///Users/gip/Geospatials/.agents/teamwork/teamwork_victory_auditor_2/handoff.md)). A static count today finds 208 `def test_` in `tests/unit` and 18 in `tests/integration` (grep; parametrization not counted).
  - **AGENTS.md misses newer modules.** It doesn't mention `src/features/polygon_rasterizer.py`, `src/features/sar_palsar2.py`, the new API routes, or `--on-schema-mismatch`.
  - **CODEBASE_CORPUS is stale.** It claims 95 modules ([AGENTS.md:L26](file:///Users/gip/Geospatials/AGENTS.md)); there are now 102 `.py` files under src/scripts/tests (`find`). Its line ranges (`#Lstart-Lend`) will drift with every edit.
  - **PROJECT.md** is the run 2 orchestrator's scope and feature-inventory document ("Project: Multi-Domain MPM Pipeline v4.0 Production Upgrade"), not a general project overview — [PROJECT.md:L1-L15](file:///Users/gip/Geospatials/PROJECT.md)
  - **HANDOFF.md** is at v2.0 (Q4).
  - **AGENTS.md lists a superseded audit as required reading:** the v2.0 audit, whose issues it says are resolved ([L15](file:///Users/gip/Geospatials/AGENTS.md)).
- **Context bloat:**
  - Sizes: AGENTS.md 20,380 B (183 lines), CODEBASE_CORPUS.md 182,983 B, CHANGELOG.md 109,939 B, `audits/*.md` about 6,000 lines total, `index.html` 77 MB (`wc`, `ls -la`).
  - Claude Code guidance is under 200 lines per CLAUDE.md, and imports don't reduce cost — [memory docs](https://code.claude.com/docs/en/memory). Antigravity's always-on rules shared a 20k-token budget — [Antigravity Rules](https://antigravity.google/docs/rules)
  - The AGENTS.md §4 resolution list (32 lines of LaTeX) is history, not instruction.

### Inferences
- In Claude Code, treat the self-certifications in AGENTS.md, audits, and teamwork reports as claims to verify, not facts. Running `/doctor prompt-audit` after creating CLAUDE.md would catch some of the drift.
- **Suggested slimming:**
  - Keep AGENTS.md to governance (§1), guardrails (§3), commands (§5), and UI rules (§6). That is about 80 lines.
  - Move §2 (module map) and §4 (resolution history) into an on-demand skill, e.g. `.claude/skills/mpm-architecture/`, or plain docs that are referenced but not imported.
  - Fix the §4.4 formula, or delete it.

### Gaps
- I didn't run the test suite (read-only, about 4 minutes), so the current pass/fail state is unverified.
- I didn't verify the other D1–D10 claims against code. The conformal gate was spot-checked only.

---

## Q7. How does each artifact map to Claude Code, and what are the migration risks?

### Takeaway
Most artifacts map directly:
- AGENTS.md → keep it, imported from CLAUDE.md
- `.agents/skills/*` → `.claude/skills/*` (same SKILL.md format: `name` + `description`)
- `.antigravity/mcp.json` → `.mcp.json` (same `mcpServers` shape)
- prose-only governance rules → hooks where they can be enforced
- teamwork orchestration → Claude Code subagents (`.claude/agents/`)

The main risks:
- about 11 days of uncommitted work
- the harvester skill's shadow-library instructions
- machine-specific symlinks and relative paths
- a misleading conformal formula in AGENTS.md
- the injected AWS block

### Cited Findings and mapping table

| Antigravity artifact | What it does | Claude Code equivalent | Notes and risks |
|---|---|---|---|
| `AGENTS.md` (repo root) | Always-on workspace rule ([Antigravity Rules](https://antigravity.google/docs/rules)) | Read natively when no CLAUDE.md exists. Better: `CLAUDE.md` with `@AGENTS.md` plus a Claude section ([memory docs](https://code.claude.com/docs/en/memory)) | Creating any CLAUDE.md or CLAUDE.local.md without the import stops AGENTS.md from loading. Remove or relocate the AWS block (L152-L183). Fix §4.4. |
| `.agents/skills/geology-rag-expert/SKILL.md` | Model-triggered skill that grounds answers via `query_geology_rag` | `.claude/skills/geology-rag-expert/SKILL.md` with the same `name`/`description` frontmatter ([skills docs](https://code.claude.com/docs/en/skills)) | Works only if the MCP server is registered. MCP tools appear as `mcp__geology-rag-mcp__query_geology_rag`, so the skill text may need the tool name updated. |
| `.agents/skills/geology-harvester/SKILL.md` (+ identical `.antigravity` copy) | Instructions for harvesting literature, including LibGen and Anna's Archive mirrors | `.claude/skills/geology-harvester/SKILL.md`, ideally with `disable-model-invocation: true` so it runs only on explicit `/geology-harvester` | **Risk:** it tells agents to fetch copyrighted books from shadow libraries ([SKILL.md:L30-L33](file:///Users/gip/Geospatials/.agents/skills/geology-harvester/SKILL.md)). Recommend removing that source before porting. Keep the credential protocol (never paste keys in chat). Delete the duplicate copy. |
| `.agents/skills/{banner-design,brand,design,design-system,slides,ui-styling,ui-ux-pro-max}` (symlinks) | Global Antigravity UI-design plugin skills | Optional: copy (not symlink) the wanted ones into `~/.claude/skills/` (personal) or `.claude/skills/` | Absolute links into `~/.gemini/...` break elsewhere. These are generic design skills, so check whether they're even needed. |
| `.antigravity/mcp.json` → `geology-rag-mcp` | Launches the FastMCP RAG server over stdio | Project `.mcp.json` with `mcpServers.geology-rag-mcp = {command, args, env}`, or `claude mcp add --scope project geology-rag-mcp -- ./.venv/bin/python mcp_server/geology_mcp_server.py` ([MCP docs](https://code.claude.com/docs/en/mcp)) | Relative `./.venv/bin/python` assumes cwd is the repo root; consider `${CLAUDE_PROJECT_DIR:-.}`. Project servers need one-time approval. The server reads `ELSEVIER_*` from `.env` via dotenv, so no secrets go in `.mcp.json`. Use `${VAR}` expansion if needed. Its heavy deps (`lancedb`, `pymupdf4llm`) must be in `.venv`. |
| `.agents/teamwork/**` (orchestrator, workers, reviewers, challengers, auditors, sentinel) | Antigravity multi-agent run logs (BRIEFING/DISPATCH/progress/handoff) | No config equivalent. Roles could become `.claude/agents/*.md` subagents (e.g. reviewer, challenger, auditor) | Keep as history, still gitignored. Don't treat the verdicts as ground truth (Q6). |
| `agents/*.py` (swarm orchestrator, harvesters) | Python scripts, not agent config | Unchanged; callable via Bash or a skill | Shares the shadow-library concern if they hit LibGen (not verified; see Gaps). |
| AGENTS §1 CHANGELOG/INDEX rule | Prose rule followed by the agent | Keep in CLAUDE.md. Optionally add a `Stop` hook reminder or a `/log-change` skill ([hooks guide](https://code.claude.com/docs/en/hooks-guide)) | Hooks are deterministic; CLAUDE.md is advisory ([memory docs](https://code.claude.com/docs/en/memory)). |
| AGENTS §6 "never edit index.html" | Prose rule | `PreToolUse` hook in `.claude/settings.json`, matcher `Edit\|Write`, exits 2 when the path is `index.html` or `ui/index.html` ([hooks guide](https://code.claude.com/docs/en/hooks-guide)) | Also add `.claude/settings.json` permission deny rules for `Read(./.env)`, and consider denying reads of the 77 MB `index.html`. |
| AGENTS §6 UI/server rules | Applies only to the UI and server files | `.claude/rules/ui-dashboard.md` with `paths:` frontmatter ([memory docs](https://code.claude.com/docs/en/memory)) | Shrinks always-on context. |
| Antigravity workflows (`/name`) | None present in the repo | `.claude/skills/<name>/SKILL.md` (commands merged into skills) ([skills docs](https://code.claude.com/docs/en/skills)) | Candidates: `/run-tests`, `/rebuild-ui`, `/restart-dashboard`, `/new-audit`. |
| `docs/session/transcript.jsonl` | Antigravity trajectory, committed | None needed | Contains model thinking. Consider removing it from git history or moving it to an archive. |
| `.gitignore` ignores `.agents/` | Keeps Antigravity workspace local | Decide whether `.claude/` (settings, skills, rules) and `.mcp.json` are committed; Claude docs intend them to be shared ([memory docs](https://code.claude.com/docs/en/memory)) | Add `.claude/settings.local.json` and `CLAUDE.local.md` to `.gitignore`. |

**Further migration risks**
1. **Uncommitted work.** All governance files and the v2–v4.4 code are untracked or modified (Q5). Commit them on a branch first.
2. **AGENTS.md must stay shared.** Antigravity also reads AGENTS.md ([Antigravity Rules](https://antigravity.google/docs/rules)), so it should remain the shared file if Antigravity stays in use. Put Claude-only content in CLAUDE.md.
3. **`.agents/` is invisible to Claude Code.** It reads nothing under `.agents/` ([memory docs](https://code.claude.com/docs/en/memory)), so skills must be copied into `.claude/skills/`. A symlink `.claude/skills → ../.agents/skills` is possible but inherits the broken absolute symlinks.
4. **Skill format compatibility.** Antigravity requires `description`; Claude recommends it ([Antigravity Skills](https://antigravity.google/docs/skills), [Claude skills](https://code.claude.com/docs/en/skills)). Both existing SKILL.md files already have `name` + `description`, so they port without edits apart from tool names.
5. **Agent workspace bloat.** The `outputs/` and `data/` churn (tracked rasters, a 77 MB `index.html` rebuilt on each UI change) will bloat diffs and any agent's Grep/Read. Add ignore and permission guidance.

### Inferences
- **A minimal first migration step:**
  - a `CLAUDE.md` of a few lines (`@AGENTS.md`, plus pointers to `audits/INDEX.md` and the top of the CHANGELOG)
  - `.mcp.json` for `geology-rag-mcp`
  - copies of the 2 real skills (harvester sanitized)
  - one `PreToolUse` hook protecting `index.html`
- AGENTS.md slimming and doc reconciliation (README, HANDOFF, corpus counts, §4.4 formula) are the next step.

### Gaps
- I didn't check whether `agents/*.py` or `mcp_server/geology_mcp_server.py` actually call LibGen or Anna's Archive endpoints. Only the skill text was confirmed.
- I didn't verify that `.venv` has `mcp`, `lancedb`, and `pymupdf4llm` installed. `requirements.txt` (untracked, 233 bytes) wasn't read.
- Antigravity workflow storage paths aren't documented on the official pages fetched. Third-party sources cite `.agent/workflows` and `.agents/workflows` ([search summary sources, e.g. KDnuggets](https://www.kdnuggets.com/build-better-ai-agents-with-google-antigravity-skills-and-workflows)). Irrelevant here, since none exist.
