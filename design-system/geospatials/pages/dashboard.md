# Geospatials // Modern Minimalist Geospatial Analytics Dashboard

> **PAGE OVERRIDE SPECIFICATION**  
> Overrides `design-system/geospatials/MASTER.md` for the primary analytical and geospatial intelligence command workspace.

---

## 1. Visual Philosophy & Design Principles
- **Spaciousness via Geometry & Elevation**: The geospatial map is the primary spatial anchor, occupying 54–100% of workspace width and 100% viewport height. Clutter is eliminated via floating frosted acrylic glass pills (`backdrop-filter: blur(12px)`), clean 1px hairline borders (`#1E293B` / `#E2E8F0`), and multi-mode workspace views.
- **Modern Swiss Precision**: High information-to-ink ratio with clear negative space, 12–18px panel paddings, disciplined alignment, and accessible contrast ratios (minimum 4.5:1 text contrast).
- **Data-Dense Telemetry**: Multi-sensor layer overlays (EMIT SWIR, Mag 1VD, AEM, DEM, Mines, Target Polygons), live coordinate cursor tracking (MGA Zone 51S Easting/Northing), interactive 3D subsurface tomographic scanner, conformal risk prediction intervals, and ensemble PR-AUC curves.

---

## 2. Design Dials
- **Variance**: 7/10 (Balanced / Modern Swiss Studio)
- **Motion**: 5/10 (Standard 150–220ms transitions, respects `prefers-reduced-motion`)
- **Density**: 8/10 (High-acuity analytical density with breathable card geometry)

---

## 3. Strict Monochromatic & Scientific Accent Tokens

| Role | Dark Token | Light Token | Purpose |
|------|------------|-------------|---------|
| `--bg-canvas` | `#070A10` | `#F8FAFC` | Deep Abyssal Base / Technical Alabaster |
| `--bg-surface` | `#0D131F` | `#FFFFFF` | Cards & Structured Panels |
| `--bg-elevated` | `#141D2E` | `#F1F5F9` | Elevated Chips, Badges, Tab Containers |
| `--bg-glass` | `rgba(13, 19, 31, 0.85)` | `rgba(255, 255, 255, 0.88)` | Floating Map Bars & HUD Controls |
| `--border-hairline` | `#1E293B` | `#E2E8F0` | 1px Structural Dividers |
| `--text-primary` | `#F8FAFC` | `#0F172A` | Primary Labels & Metrics |
| `--text-secondary` | `#94A3B8` | `#475569` | Geological Domain & Secondary Data |
| `--text-muted` | `#64748B` | `#94A3B8` | Units, Grid Reference & Hints |
| `--accent-primary` | `#38BDF8` | `#0284C7` | Electric Cyan Prospectivity Vectors |
| `--accent-amber` | `#F59E0B` | `#D97706` | Mineralized Jog Anomalies & Drill Alerts |
| `--accent-emerald` | `#10B981` | `#059669` | High Confidence / Tight Conformal Bounds |
| `--accent-purple` | `#A855F7` | `#7C3AED` | Neural Network U-Net Model Vectors |

---

## 4. 4-Tier Non-Cliché Typography
1. **Display & Headings**: `Schibsted Grotesk` (700/800, `-0.01em` tracking) — Crisp, modern, authoritative.
2. **Editorial Geological Notes**: `Newsreader` (italic, 400/500) — Nuanced geological domain & structural interpretations.
3. **UI & Control Labels**: `Atkinson Hyperlegible` / `Plus Jakarta Sans` — Clean, legible at small scales.
4. **Telemetry & Coordinates**: `IBM Plex Mono` / `Fira Code` (`tabular-nums slashed-zero`) — Precision MGA51 coordinates, conformal intervals, and depth slices.

---

## 5. Architectural Workspace Modes
- **Mode 1: `Split Studio` (Default)**: Full-height interactive map on left pane with floating basemap and filter pills; resizable 1px hairline drag handle; right-hand analytical console with Active Target Dossier, Tabbed Viewports, and Ranked Portfolio Table.
- **Mode 2: `Map Focus`**: Full-bleed 100% width/height geospatial map with floating frosted glass HUD cards.
- **Mode 3: `Analytics Deep-Dive`**: Full-screen statistical charts and ranked telemetry register for portfolio-wide resource audits.
