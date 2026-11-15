---
name: geology-rag-expert
description: Synthesizes geological laws, structural traps, ore genesis chemistry, crustal architecture, and remote sensing signatures using the local Geology RAG database (geology-rag-mcp). Use for mineral-system reasoning behind prospectivity features or targets.
---

# Geology RAG Domain Expert Skill

Ported from the Antigravity skill at `.agents/skills/geology-rag-expert/`. In Claude Code the RAG tool is `mcp__geology-rag-mcp__query_geology_rag`, served by `.mcp.json`. If that tool is unavailable, say so and do not present ungrounded synthesis as RAG-backed.

## Workflow
1. **Always Ground in Local RAG First**: Call `mcp__geology-rag-mcp__query_geology_rag` across multiple relevant domains (`stratigraphic_laws`, `structural_faults_vaults`, `crustal_architecture`, `ore_genesis`, `remote_sensing`, `ore_prospectivity_ml`) before synthesizing an answer.
2. **Connect the 4-Part Mineral System Chain**:
   - **Stratigraphic & Lithological Architecture**: Apply Steno's Laws (Superposition, Original Horizontality, Lateral Continuity, Cross-Cutting, Inclusions), Faunal Succession, and Walther's Law of Facies to reconstruct 3D basin geometry and reactive host units.
   - **Crustal & Structural Plumbing (Faults & Vaults)**: Trace deep trans-lithospheric fault conduits, Sibson's fault-valve seismic pumping, cross-cutting intrusions/veins, and anticlinal vaults/caprock seals.
   - **Physicochemical Ore Genesis**: Detail fluid ligands (chloride vs. bisulfide complexes), $P$-$T$-$f\text{O}_2$-$\text{pH}$ triggers, boiling/phase separation, and wall-rock sulfidation.
   - **Remote Sensing & ML Prospectivity**: Translate the alteration halo and structural geometry into observable sensor bands (ASTER/Sentinel-2/EnMAP SWIR-TIR, SAR/LiDAR structural lineaments, satellite magnetics/gravity) and ML feature engineering.
3. **Respect project guardrails** (AGENTS.md §3): any feature suggested must be a physically meaningful ratio, gradient or tensor, never raw sensor bands or distances to known mines.
