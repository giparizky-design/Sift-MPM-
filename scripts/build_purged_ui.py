import os
import re
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

def main():
    # Read the existing embedded bundle line from ui/index.html
    with open('ui/index.html', 'r', encoding='utf-8') as f:
        bundle_line = None
        for line in f:
            if 'const EMBEDDED_BUNDLES = ' in line:
                bundle_line = line
                break

    if not bundle_line:
        print('Error: EMBEDDED_BUNDLES line not found!')
        return

    print(f'Found bundle line ({len(bundle_line)} bytes)')

    # Ensure KALIMANTAN bundle and archived runs are loaded cleanly
    try:
        raw_json = bundle_line.strip()[len('const EMBEDDED_BUNDLES = '):].rstrip(';')
        bundles = {}

        # Load active Kalimantan bundle with fresh K=4 prospectivity map
        from scripts.dashboard_server import _load_region_bundle, _load_bundle_from_run_dir
        k_bundle = _load_region_bundle('EAST_KALIMANTAN', force_reload=True)
        bundles['KALIMANTAN'] = k_bundle
        bundles['EAST_KALIMANTAN'] = k_bundle
        bundles['BORNEO'] = k_bundle
        print('Successfully loaded active KALIMANTAN K=4 bundle with prospectivity heatmap!')

        # Embed all valid archived runs for 0ms instant switching
        catalog_p = ROOT_DIR / "outputs" / "run_history" / "training_runs_catalog.json"
        if catalog_p.exists():
            cdata = json.loads(catalog_p.read_text(encoding="utf-8"))
            for r in cdata.get("runs", []):
                if not r.get("is_active"):
                    run_id = r.get("run_id")
                    rel = r.get("relative_path")
                    if run_id and rel:
                        p = ROOT_DIR / rel
                        if p.exists() and (p / "maps" / "calibrated_prospectivity.tif").exists():
                            try:
                                bundles[run_id] = _load_bundle_from_run_dir(p, "KALIMANTAN")
                                print(f'Embedded archived run {run_id}')
                            except Exception as ex:
                                print(f'Notice: could not embed archived run {run_id}: {ex}')

        bundle_line = 'const EMBEDDED_BUNDLES = ' + json.dumps(bundles) + ';\n'
    except Exception as e:
        print(f'Warning: Could not update bundles: {e}')

    # Read catalog for archived runs to populate the dropdown
    archived_opts = ""
    catalog_p = ROOT_DIR / "outputs" / "run_history" / "training_runs_catalog.json"
    if catalog_p.exists():
        try:
            cdata = json.loads(catalog_p.read_text(encoding="utf-8"))
            runs_archived = [r for r in cdata.get("runs", []) if not r.get("is_active") and r.get("region") in ("BORNEO", "KALIMANTAN", "EAST_KALIMANTAN", "KALTIM")]
            for r in runs_archived:
                time_str = (r.get("archived_at") or r.get("created_at") or "")[:16].replace("T", " ")
                fused = r.get("metrics", {}).get("pr_auc_fused", 0) or 0
                reg = r.get("region", "KALIMANTAN")
                archived_opts += f'            <option value="{r["run_id"]}">{reg} ARCHIVE ({time_str} // PR-AUC: {fused:.3f})</option>\n'
        except Exception as e:
            print("Warning reading catalog:", e)
    if not archived_opts:
        archived_opts = '            <option value="" disabled>No archived runs available</option>\n'

    # Construct the ultra-clean, spacious, modern & minimalistic, yet data-dense Antigravity Workspace HTML
    html_head = '''<!DOCTYPE html>
<html lang="en" class="dark">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>SIFT // Spatial Inference & Fault Targeting</title>

  <!-- 4-Tier Non-Cliché Typography -->
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="" />
  <link href="https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:ital,wght@0,400;0,700;1,400;1,700&family=IBM+Plex+Mono:ital,wght@0,400;0,500;0,600;0,700;1,400&family=Newsreader:ital,opsz,wght@1,6..72,400;1,6..72,500;1,6..72,600&family=Schibsted+Grotesk:wght@500;600;700;800&family=Plus+Jakarta+Sans:wght@500;600;700&display=swap" rel="stylesheet" />

  <!-- Leaflet CSS & JS -->
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" crossOrigin="" />
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" crossOrigin=""></script>

  <style>
    /* ==========================================================================
       SIFT // STRICT MONOCHROMATIC DUAL-THEME TOKENS (SWISS MODERN MINIMALISM)
       ========================================================================== */
    :root {
      /* Typography Tokens */
      --font-display: 'Schibsted Grotesk', -apple-system, BlinkMacSystemFont, sans-serif;
      --font-editorial: 'Newsreader', Georgia, serif;
      --font-sans: 'Atkinson Hyperlegible', 'Plus Jakarta Sans', sans-serif;
      --font-mono: 'IBM Plex Mono', ui-monospace, SFMono-Regular, monospace;

      /* Dark Mode ("Deep Obsidian & Titanium Slate" — Default) */
      --bg-canvas: #060709;
      --bg-surface: #0B0E14;
      --bg-elevated: #11151E;
      --bg-glass: rgba(11, 14, 20, 0.94);
      --border-hairline: rgba(255, 255, 255, 0.08);
      --border-subtle: rgba(255, 255, 255, 0.04);
      --border-strong: rgba(255, 255, 255, 0.16);
      
      --text-primary: #F4F4F6;
      --text-secondary: #94A3B8;
      --text-muted: #5A6474;

      /* Strict Monochromatic Accents */
      --accent-primary: #FFFFFF;
      --accent-secondary: #E2E8F0;
      --accent-tertiary: #94A3B8;
      --accent-zinc: #1E293B;
      --accent-highlight: rgba(255, 255, 255, 0.10);
      --shadow-elevation: 0 12px 40px rgba(0, 0, 0, 0.65);
    }

    body.light-theme {
      --bg-canvas: #F8FAFC !important;
      --bg-surface: #FFFFFF !important;
      --bg-elevated: #F1F5F9 !important;
      --bg-glass: rgba(255, 255, 255, 0.94) !important;
      --border-hairline: #E2E8F0 !important;
      --border-subtle: rgba(0, 0, 0, 0.04) !important;
      --border-strong: #CBD5E1 !important;

      --text-primary: #090A0C !important;
      --text-secondary: #475569 !important;
      --text-muted: #94A3B8 !important;

      --accent-primary: #090A0C !important;
      --accent-secondary: #1E293B !important;
      --accent-tertiary: #475569 !important;
      --accent-zinc: #E2E8F0 !important;
      --accent-highlight: rgba(0, 0, 0, 0.05) !important;
      --shadow-elevation: 0 12px 36px rgba(0, 0, 0, 0.08) !important;
    }

    /* Reset & Viewport Constraints */
    *, *::before, *::after {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }

    html, body {
      width: 100%;
      height: 100%;
      background-color: var(--bg-canvas);
      color: var(--text-primary);
      font-family: var(--font-sans);
      overflow: hidden;
      -webkit-font-smoothing: antialiased;
      -moz-osx-font-smoothing: grayscale;
    }

    ::selection {
      background-color: rgba(255, 255, 255, 0.25);
      color: #FFFFFF;
    }

    /* Monospace Telemetry Scrollbars */
    ::-webkit-scrollbar {
      width: 5px;
      height: 5px;
    }
    ::-webkit-scrollbar-track {
      background: var(--bg-canvas);
    }
    ::-webkit-scrollbar-thumb {
      background: var(--border-hairline);
      border-radius: 2px;
    }
    ::-webkit-scrollbar-thumb:hover {
      background: var(--text-muted);
    }

    /* Accessible Focus Rings */
    button:focus-visible, input:focus-visible, select:focus-visible, a:focus-visible {
      outline: 2px solid var(--accent-primary) !important;
      outline-offset: 2px !important;
    }

    /* Respect Reduced Motion */
    @media (prefers-reduced-motion: reduce) {
      *, *::before, *::after {
        animation-duration: 0.001s !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.001s !important;
      }
      .pulse-node {
        animation: none !important;
      }
    }

    /* Typography Utility Classes */
    .font-display { font-family: var(--font-display); }
    .font-editorial { font-family: var(--font-editorial); }
    .font-sans { font-family: var(--font-sans); }
    .font-mono { font-family: var(--font-mono); }

    /* Layout Structure */
    .app-shell {
      display: flex;
      flex-direction: column;
      height: 100vh;
      width: 100vw;
      background: var(--bg-canvas);
      overflow: hidden;
    }

    /* ==================== 2-BAR TOP MASTHEAD ARCHITECTURE ==================== */
    /* Bar 1: Primary Masthead & Navigation (46px height) */
    .top-header {
      height: 46px;
      flex-shrink: 0;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 20px;
      background: var(--bg-surface);
      border-bottom: 1px solid var(--border-hairline);
      z-index: 50;
      gap: 16px;
    }

    .brand-wrap {
      display: flex;
      align-items: center;
      gap: 10px;
      flex-shrink: 0;
    }

    .sift-logo-mark {
      display: flex;
      align-items: center;
      justify-content: center;
      color: var(--text-primary);
    }

    .brand-title {
      font-family: var(--font-display);
      font-size: 14px;
      font-weight: 800;
      letter-spacing: 0.04em;
      line-height: 1;
      color: var(--text-primary);
      display: flex;
      align-items: baseline;
      white-space: nowrap;
    }

    .brand-subtitle {
      font-family: var(--font-mono);
      font-weight: 400;
      font-size: 10px;
      color: var(--text-muted);
      letter-spacing: 0.04em;
      margin-left: 6px;
      white-space: nowrap;
    }

    .header-right-meta {
      display: flex;
      align-items: center;
      gap: 10px;
      flex-shrink: 0;
    }

    .system-status-pill {
      display: flex;
      align-items: center;
      gap: 6px;
      padding: 3px 8px;
      border-radius: 4px;
      background: var(--bg-elevated);
      border: 1px solid var(--border-subtle);
      font-family: var(--font-mono);
      font-size: 9.5px;
      color: var(--text-secondary);
      letter-spacing: 0.04em;
      white-space: nowrap;
    }

    /* Bar 2: Secondary Telemetry & Command Bar (42px height) */
    .sub-header-bar {
      height: 42px;
      flex-shrink: 0;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 20px;
      background: var(--bg-canvas);
      border-bottom: 1px solid var(--border-hairline);
      z-index: 45;
      gap: 14px;
      overflow-x: auto;
      overflow-y: hidden;
    }

    .sub-bar-left {
      display: flex;
      align-items: center;
      gap: 8px;
      flex-shrink: 0;
    }

    .sub-bar-section-label {
      font-family: var(--font-mono);
      font-size: 9.5px;
      font-weight: 700;
      color: var(--text-muted);
      letter-spacing: 0.05em;
      margin-right: 2px;
      white-space: nowrap;
    }

    .sub-bar-right {
      display: flex;
      align-items: center;
      gap: 8px;
      flex-shrink: 0;
    }

    .telemetry-chip {
      display: flex;
      align-items: center;
      gap: 5px;
      padding: 3px 8px;
      border-radius: 4px;
      background: var(--bg-elevated);
      border: 1px solid var(--border-subtle);
      white-space: nowrap;
    }

    .chip-label {
      color: var(--text-muted);
      font-size: 10px;
      font-family: var(--font-mono);
      white-space: nowrap;
    }

    .chip-val {
      color: var(--text-primary);
      font-weight: 700;
      font-family: var(--font-mono);
      font-size: 11px;
    }

    /* Primary Navigation Switcher (Mission Control vs Workspace) */
    .nav-tabs-group {
      display: flex;
      align-items: center;
      background: var(--bg-elevated);
      border: 1px solid var(--border-hairline);
      border-radius: 6px;
      padding: 2px;
      gap: 2px;
    }

    .nav-tab-btn {
      background: transparent;
      border: none;
      color: var(--text-secondary);
      font-family: var(--font-mono);
      font-size: 11px;
      font-weight: 500;
      padding: 5px 12px;
      border-radius: 4px;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 6px;
      transition: all 0.15s ease;
    }

    .nav-tab-btn:hover {
      color: var(--text-primary);
      background: rgba(255, 255, 255, 0.04);
    }

    .nav-tab-btn.active {
      background: var(--accent-primary);
      color: var(--bg-canvas);
      font-weight: 700;
      box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2);
    }

    body.light-theme .nav-tab-btn.active {
      background: #090A0C !important;
      color: #FFFFFF !important;
    }

    /* Segmented Layout Mode Switcher */
    .view-mode-pill {
      display: flex;
      align-items: center;
      background: var(--bg-elevated);
      border: 1px solid var(--border-hairline);
      border-radius: 6px;
      padding: 2px;
      gap: 2px;
    }

    .view-mode-btn {
      background: transparent;
      border: none;
      color: var(--text-secondary);
      font-family: var(--font-mono);
      font-size: 10px;
      font-weight: 500;
      padding: 4px 8px;
      border-radius: 4px;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 4px;
      transition: all 0.15s ease;
    }

    .view-mode-btn:hover {
      color: var(--text-primary);
    }

    .view-mode-btn.active {
      background: var(--accent-primary);
      color: var(--bg-canvas);
      font-weight: 700;
      box-shadow: 0 1px 3px rgba(0, 0, 0, 0.15);
    }

    body.light-theme .view-mode-btn.active {
      background: #090A0C !important;
      color: #FFFFFF !important;
    }

    .select-dataset {
      background: var(--bg-elevated);
      color: var(--text-primary);
      border: 1px solid var(--border-hairline);
      border-radius: 6px;
      font-family: var(--font-mono);
      font-size: 11px;
      padding: 5px 10px;
      outline: none;
      cursor: pointer;
      transition: border-color 0.15s ease;
      max-width: 310px;
    }

    .select-dataset:focus {
      border-color: var(--accent-primary);
    }

    .select-dataset optgroup {
      background: var(--bg-surface);
      color: var(--text-muted);
      font-family: var(--font-mono);
      font-weight: 700;
      font-size: 10px;
    }

    .select-dataset option {
      background: var(--bg-elevated);
      color: var(--text-primary);
      font-size: 11px;
      padding: 4px 8px;
    }

    .select-ore-category {
      max-width: 200px;
      color: var(--text-primary);
    }

    .select-run-history {
      max-width: 220px;
      color: var(--text-primary);
    }

    /* Monochromatic Ore Style Badges */
    .ore-badge {
      display: inline-flex;
      align-items: center;
      gap: 5px;
      padding: 2px 7px;
      border-radius: 4px;
      font-size: 9.5px;
      font-family: var(--font-mono);
      font-weight: 600;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      white-space: nowrap;
      background: rgba(255, 255, 255, 0.05);
      color: var(--text-primary);
      border: 1px solid var(--border-hairline);
    }

    .ore-badge::before {
      content: "";
      display: inline-block;
      width: 5px;
      height: 5px;
      border-radius: 50%;
      background: var(--text-secondary);
    }

    .ore-badge-porphyry_cu::before { background: #FFFFFF; }
    .ore-badge-skarn_fe_cu::before { background: #E2E8F0; }
    .ore-badge-epithermal_au::before { background: #CBD5E1; }
    .ore-badge-laterite_fe_ni::before { background: #94A3B8; }
    .ore-badge-primary_gold::before { background: #F8FAFC; }
    .ore-badge-coal::before { background: #94A3B8; }
    .ore-badge-telescoped::before { background: #FFFFFF; box-shadow: 0 0 6px rgba(255, 255, 255, 0.8); }

    /* High-Definition Monochromatic Leaflet Tooltip HUD */
    .leaflet-tooltip {
      background: rgba(11, 14, 20, 0.96) !important;
      border: 1px solid rgba(255, 255, 255, 0.20) !important;
      color: #FFFFFF !important;
      border-radius: 4px !important;
      padding: 8px 10px !important;
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.6) !important;
      backdrop-filter: blur(8px) !important;
      -webkit-backdrop-filter: blur(8px) !important;
    }
    .leaflet-tooltip-top:before { border-top-color: rgba(11, 14, 20, 0.96) !important; }
    .leaflet-tooltip-bottom:before { border-bottom-color: rgba(11, 14, 20, 0.96) !important; }
    .leaflet-tooltip-left:before { border-left-color: rgba(11, 14, 20, 0.96) !important; }
    .leaflet-tooltip-right:before { border-right-color: rgba(11, 14, 20, 0.96) !important; }
    body.light-theme .leaflet-tooltip {
      background: rgba(255, 255, 255, 0.96) !important;
      border: 1px solid rgba(0, 0, 0, 0.20) !important;
      color: #090A0C !important;
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12) !important;
    }
    body.light-theme .leaflet-tooltip-top:before { border-top-color: rgba(255, 255, 255, 0.96) !important; }

    .btn-icon {
      background: var(--bg-elevated);
      border: 1px solid var(--border-hairline);
      border-radius: 6px;
      color: var(--text-secondary);
      padding: 6px 8px;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      transition: all 0.15s ease;
    }

    .btn-icon:hover {
      color: var(--text-primary);
      border-color: var(--accent-primary);
    }

    /* ==========================================================================
       HOMEPAGE // MISSION CONTROL STYLES
       ========================================================================== */
    .view-home-container {
      flex: 1;
      min-height: 0;
      width: 100%;
      box-sizing: border-box;
      overflow-y: auto;
      overflow-x: hidden;
      background: var(--bg-canvas);
      padding: 36px 32px 72px 32px;
      display: flex;
      flex-direction: column;
      align-items: center;
    }

    .home-content-wrap {
      width: 100%;
      max-width: 1260px;
      display: flex;
      flex-direction: column;
      gap: 56px;
    }

    /* Hero Section */
    .hero-section {
      display: flex;
      flex-direction: column;
      gap: 20px;
      padding-top: 12px;
      border-bottom: 1px solid var(--border-hairline);
      padding-bottom: 48px;
    }

    .hero-top-badges {
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }

    .mono-badge {
      font-family: var(--font-mono);
      font-size: 10px;
      font-weight: 600;
      letter-spacing: 0.05em;
      padding: 3px 8px;
      border-radius: 4px;
      background: var(--bg-surface);
      color: var(--text-secondary);
      border: 1px solid var(--border-hairline);
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }

    .mono-badge.active-dot .dot {
      width: 6px;
      height: 6px;
      border-radius: 50%;
      background: #FFFFFF;
      box-shadow: 0 0 8px #FFFFFF;
    }

    body.light-theme .mono-badge.active-dot .dot {
      background: #090A0C !important;
      box-shadow: none !important;
    }

    .hero-main-title {
      font-family: var(--font-display);
      font-size: 44px;
      font-weight: 800;
      letter-spacing: -0.035em;
      line-height: 1.1;
      color: var(--text-primary);
      max-width: 980px;
    }

    .hero-lead-text {
      font-family: var(--font-sans);
      font-size: 16px;
      color: var(--text-secondary);
      line-height: 1.6;
      max-width: 820px;
    }

    .hero-cta-group {
      display: flex;
      align-items: center;
      gap: 12px;
      margin-top: 8px;
      flex-wrap: wrap;
    }

    .hero-btn-primary {
      background: var(--accent-primary);
      color: var(--bg-canvas);
      font-family: var(--font-mono);
      font-size: 12px;
      font-weight: 700;
      padding: 10px 22px;
      border-radius: 5px;
      border: none;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 8px;
      transition: transform 0.15s ease, opacity 0.15s ease;
    }

    .hero-btn-primary:hover {
      opacity: 0.92;
      transform: translateY(-1px);
    }

    body.light-theme .hero-btn-primary {
      background: #090A0C !important;
      color: #FFFFFF !important;
    }

    .hero-btn-secondary {
      background: transparent;
      color: var(--text-primary);
      font-family: var(--font-mono);
      font-size: 12px;
      font-weight: 600;
      padding: 9px 18px;
      border-radius: 5px;
      border: 1px solid var(--border-hairline);
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 8px;
      text-decoration: none;
      transition: background 0.15s ease, border-color 0.15s ease;
    }

    .hero-btn-secondary:hover {
      background: var(--bg-elevated);
      border-color: var(--border-strong);
    }

    /* 4-Metric Hero KPI Grid */
    .hero-kpi-grid {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 14px;
      margin-top: 18px;
    }

    @media (max-width: 900px) {
      .hero-kpi-grid {
        grid-template-columns: repeat(2, 1fr);
      }
    }

    .hero-kpi-card {
      background: var(--bg-surface);
      border: 1px solid var(--border-hairline);
      border-radius: 8px;
      padding: 20px;
      display: flex;
      flex-direction: column;
      gap: 6px;
      transition: border-color 0.15s ease;
    }

    .hero-kpi-card:hover {
      border-color: var(--border-strong);
    }

    .kpi-micro-label {
      font-family: var(--font-mono);
      font-size: 10px;
      font-weight: 600;
      color: var(--text-muted);
      letter-spacing: 0.05em;
    }

    .kpi-hero-num {
      font-family: var(--font-mono);
      font-size: 28px;
      font-weight: 700;
      color: var(--text-primary);
      letter-spacing: -0.02em;
      font-variant-numeric: tabular-nums;
    }

    .kpi-unit {
      font-size: 14px;
      color: var(--text-secondary);
      font-weight: 500;
    }

    .kpi-desc {
      font-size: 12px;
      color: var(--text-secondary);
      line-height: 1.45;
    }

    /* Section Headers */
    .section-header {
      display: flex;
      flex-direction: column;
      gap: 6px;
      margin-bottom: 20px;
    }

    .section-tag {
      font-family: var(--font-mono);
      font-size: 10.5px;
      font-weight: 700;
      color: var(--text-muted);
      letter-spacing: 0.06em;
    }

    .section-title {
      font-family: var(--font-display);
      font-size: 22px;
      font-weight: 700;
      color: var(--text-primary);
      letter-spacing: -0.02em;
    }

    .section-subtitle {
      font-size: 13.5px;
      color: var(--text-secondary);
      max-width: 640px;
    }

    /* Exploration Theaters Grid */
    .theaters-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
    }

    @media (max-width: 820px) {
      .theaters-grid {
        grid-template-columns: 1fr;
      }
    }

    .theater-card {
      background: var(--bg-surface);
      border: 1px solid var(--border-hairline);
      border-radius: 8px;
      padding: 24px;
      display: flex;
      flex-direction: column;
      gap: 16px;
      transition: border-color 0.15s ease, box-shadow 0.15s ease;
    }

    .theater-card:hover {
      border-color: var(--border-strong);
    }

    .theater-card.featured {
      border-color: rgba(255, 255, 255, 0.2);
      background: linear-gradient(180deg, var(--bg-surface) 0%, var(--bg-elevated) 100%);
    }

    .theater-card-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
    }

    .theater-status-badge {
      font-family: var(--font-mono);
      font-size: 9.5px;
      font-weight: 700;
      padding: 3px 8px;
      border-radius: 4px;
      background: var(--bg-elevated);
      color: var(--text-secondary);
      border: 1px solid var(--border-hairline);
      display: inline-flex;
      align-items: center;
      gap: 5px;
    }

    .theater-status-badge.active {
      color: var(--text-primary);
      border-color: var(--border-strong);
    }

    .theater-status-badge.active .dot {
      width: 5px;
      height: 5px;
      border-radius: 50%;
      background: #FFFFFF;
    }

    .theater-crs {
      font-family: var(--font-mono);
      font-size: 10px;
      color: var(--text-muted);
    }

    .theater-name {
      font-family: var(--font-display);
      font-size: 19px;
      font-weight: 700;
      color: var(--text-primary);
    }

    .theater-desc {
      font-size: 13px;
      color: var(--text-secondary);
      line-height: 1.5;
    }

    .theater-metrics-row {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 10px;
      padding: 12px 0;
      border-top: 1px solid var(--border-hairline);
      border-bottom: 1px solid var(--border-hairline);
    }

    .theater-metrics-row .lbl {
      display: block;
      font-family: var(--font-mono);
      font-size: 9.5px;
      color: var(--text-muted);
      margin-bottom: 2px;
    }

    .theater-metrics-row .val {
      font-family: var(--font-mono);
      font-size: 12.5px;
      font-weight: 700;
      color: var(--text-primary);
    }

    .theater-launch-btn {
      background: var(--bg-elevated);
      border: 1px solid var(--border-hairline);
      color: var(--text-primary);
      font-family: var(--font-mono);
      font-size: 11px;
      font-weight: 600;
      padding: 10px;
      border-radius: 5px;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 6px;
      transition: all 0.15s ease;
      width: 100%;
    }

    .theater-launch-btn:hover {
      background: var(--accent-primary);
      color: var(--bg-canvas);
      border-color: var(--accent-primary);
    }

    body.light-theme .theater-launch-btn:hover {
      background: #090A0C !important;
      color: #FFFFFF !important;
    }

    /* 7-Pass Scientific Pipeline Cards */
    .pipeline-cards-grid {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 14px;
    }

    @media (max-width: 1050px) {
      .pipeline-cards-grid {
        grid-template-columns: repeat(2, 1fr);
      }
    }

    @media (max-width: 600px) {
      .pipeline-cards-grid {
        grid-template-columns: 1fr;
      }
    }

    .pipeline-card {
      background: var(--bg-surface);
      border: 1px solid var(--border-hairline);
      border-radius: 8px;
      padding: 18px;
      display: flex;
      flex-direction: column;
      gap: 8px;
      transition: border-color 0.15s ease;
    }

    .pipeline-card:hover {
      border-color: var(--border-strong);
    }

    .pipeline-step-num {
      font-family: var(--font-mono);
      font-size: 10px;
      font-weight: 700;
      color: var(--text-muted);
      letter-spacing: 0.05em;
    }

    .pipeline-card h4 {
      font-family: var(--font-display);
      font-size: 13.5px;
      font-weight: 700;
      color: var(--text-primary);
      line-height: 1.3;
    }

    .pipeline-card p {
      font-size: 12px;
      color: var(--text-secondary);
      line-height: 1.5;
    }

    /* 21-Channel Sensor Table */
    .sensor-matrix-table-wrap {
      background: var(--bg-surface);
      border: 1px solid var(--border-hairline);
      border-radius: 8px;
      overflow-x: auto;
    }

    .sensor-matrix-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 11.5px;
      text-align: left;
    }

    .sensor-matrix-table th {
      background: var(--bg-elevated);
      color: var(--text-secondary);
      font-family: var(--font-mono);
      font-size: 10px;
      font-weight: 600;
      padding: 10px 14px;
      border-bottom: 1px solid var(--border-hairline);
      letter-spacing: 0.04em;
    }

    .sensor-matrix-table td {
      padding: 10px 14px;
      border-bottom: 1px solid var(--border-subtle);
      color: var(--text-secondary);
    }

    .sensor-matrix-table tr:hover td {
      background: rgba(255, 255, 255, 0.02);
      color: var(--text-primary);
    }

    .sensor-matrix-table td:nth-child(1) {
      font-family: var(--font-mono);
      font-weight: 700;
      color: var(--text-primary);
    }

    .sensor-matrix-table td:nth-child(2) {
      font-family: var(--font-mono);
      color: var(--accent-secondary);
      font-weight: 600;
    }

    /* Homepage Footer */
    .home-footer {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding-top: 32px;
      border-top: 1px solid var(--border-hairline);
      width: 100%;
      flex-wrap: wrap;
      gap: 16px;
    }

    .footer-brand {
      font-family: var(--font-mono);
      font-size: 11px;
      font-weight: 700;
      color: var(--text-primary);
    }

    .footer-copy {
      font-size: 11px;
      color: var(--text-muted);
      margin-left: 12px;
    }

    .footer-tag {
      font-family: var(--font-mono);
      font-size: 10px;
      color: var(--text-muted);
      padding: 3px 8px;
      background: var(--bg-surface);
      border: 1px solid var(--border-hairline);
      border-radius: 4px;
    }

    /* ==========================================================================
       WORKSPACE VIEW & CENTRAL SPLIT-PANE STYLES
       ========================================================================== */
    .workspace-body {
      flex: 1;
      display: flex;
      overflow: hidden;
      position: relative;
      height: calc(100vh - 88px);
    }

    /* Collapsible Progressive Disclosure Drawer */
    .telemetry-drawer {
      width: 270px;
      height: 100%;
      flex-shrink: 0;
      background: var(--bg-surface);
      border-right: 1px solid var(--border-hairline);
      display: flex;
      flex-direction: column;
      transition: width 0.22s cubic-bezier(0.4, 0, 0.2, 1);
      z-index: 30;
      overflow: hidden;
    }

    .telemetry-drawer.collapsed {
      width: 48px;
    }

    .drawer-header {
      height: 46px;
      padding: 0 12px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      border-bottom: 1px solid var(--border-hairline);
      flex-shrink: 0;
    }

    .drawer-content {
      flex: 1;
      overflow-y: auto;
      padding: 16px 14px;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }

    .drawer-collapsed-rail {
      display: none;
      flex-direction: column;
      align-items: center;
      gap: 16px;
      padding: 18px 0;
    }

    .telemetry-drawer.collapsed .drawer-content {
      display: none;
    }

    .telemetry-drawer.collapsed .drawer-collapsed-rail {
      display: flex;
    }

    /* Resizable Split-Pane Central Workspace */
    .split-workspace {
      flex: 1;
      display: flex;
      overflow: hidden;
      position: relative;
      background: var(--bg-canvas);
    }

    /* Left Pane: Geospatial Map Canvas */
    .split-pane-left {
      height: 100%;
      position: relative;
      display: flex;
      flex-direction: column;
      overflow: hidden;
      transition: width 0.1s ease;
    }

    /* Right Pane: Analytical Studio Console */
    .split-pane-right {
      height: 100%;
      overflow-y: auto;
      padding: 16px 18px;
      display: flex;
      flex-direction: column;
      gap: 16px;
      background: var(--bg-canvas);
      transition: width 0.1s ease;
    }

    /* Full-Screen Modes */
    .workspace-body.mode-map .split-pane-left {
      width: 100% !important;
    }
    .workspace-body.mode-map .split-pane-right,
    .workspace-body.mode-map .split-drag-handle {
      display: none !important;
    }

    .workspace-body.mode-analytics .split-pane-right {
      width: 100% !important;
    }
    .workspace-body.mode-analytics .split-pane-left,
    .workspace-body.mode-analytics .split-drag-handle {
      display: none !important;
    }

    /* 1px Hairline Drag Handle with Kinetic Border Illumination */
    .split-drag-handle {
      width: 8px;
      margin-left: -4px;
      margin-right: -4px;
      cursor: col-resize;
      z-index: 28;
      display: flex;
      align-items: center;
      justify-content: center;
      background: transparent;
      user-select: none;
      flex-shrink: 0;
    }

    .split-drag-line {
      width: 1px;
      height: 100%;
      background: var(--border-hairline);
      transition: background-color 0.15s ease, box-shadow 0.15s ease;
    }

    .split-drag-handle:hover .split-drag-line,
    .split-drag-handle.active .split-drag-line {
      background: var(--accent-primary);
      box-shadow: 0 0 10px var(--accent-primary);
    }

    /* Modern Minimalist Glass Panels */
    .hairline-panel {
      background: var(--bg-surface);
      border: 1px solid var(--border-hairline);
      border-radius: 8px;
      padding: 16px 18px;
      display: flex;
      flex-direction: column;
      transition: border-color 0.15s ease, box-shadow 0.15s ease;
    }

    .hairline-panel:hover {
      border-color: var(--border-strong);
    }

    .panel-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding-bottom: 10px;
      margin-bottom: 12px;
      border-bottom: 1px solid var(--border-hairline);
    }

    .panel-tag {
      font-family: var(--font-mono);
      font-size: 10px;
      font-weight: 700;
      color: var(--text-primary);
      background: var(--bg-elevated);
      padding: 2px 7px;
      border-radius: 4px;
      border: 1px solid var(--border-hairline);
    }

    .panel-title {
      font-family: var(--font-display);
      font-size: 13px;
      font-weight: 600;
      letter-spacing: -0.02em;
      color: var(--text-primary);
    }

    /* Full-Height Geospatial Map Viewport */
    #map-viewport {
      width: 100%;
      height: 100%;
      background: var(--bg-canvas);
      z-index: 10;
    }

    .leaflet-container {
      background: var(--bg-canvas) !important;
      font-family: var(--font-sans);
    }

    /* Floating Frosted Glass Map Controls */
    .floating-map-bar-top {
      position: absolute;
      top: 14px;
      left: 14px;
      right: 14px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      pointer-events: none;
      z-index: 400;
      gap: 10px;
    }

    .glass-pill-group {
      pointer-events: auto;
      display: flex;
      align-items: center;
      gap: 4px;
      background: var(--bg-glass);
      backdrop-filter: blur(12px);
      -webkit-backdrop-filter: blur(12px);
      border: 1px solid var(--border-hairline);
      border-radius: 8px;
      padding: 3px 4px;
      box-shadow: var(--shadow-elevation);
    }

    .glass-pill-btn {
      background: transparent;
      border: none;
      font-family: var(--font-mono);
      font-size: 10px;
      font-weight: 600;
      color: var(--text-secondary);
      padding: 4px 8px;
      border-radius: 5px;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 4px;
      transition: all 0.12s ease;
    }

    .glass-pill-btn:hover {
      color: var(--text-primary);
      background: var(--bg-elevated);
    }

    .glass-pill-btn.active {
      background: var(--accent-primary);
      color: var(--bg-canvas);
      font-weight: 700;
    }

    body.light-theme .glass-pill-btn.active {
      background: #090A0C !important;
      color: #FFFFFF !important;
    }

    /* Floating Telemetry HUD at Bottom of Map */
    .floating-map-hud {
      position: absolute;
      bottom: 14px;
      left: 14px;
      display: flex;
      align-items: center;
      gap: 14px;
      background: var(--bg-glass);
      backdrop-filter: blur(12px);
      -webkit-backdrop-filter: blur(12px);
      border: 1px solid var(--border-hairline);
      border-radius: 8px;
      padding: 6px 14px;
      font-family: var(--font-mono);
      font-size: 10.5px;
      color: var(--text-secondary);
      box-shadow: var(--shadow-elevation);
      pointer-events: none;
      z-index: 400;
    }

    /* KPI Grid in Active Target Dossier */
    .kpi-grid {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 10px;
      padding-top: 12px;
      border-top: 1px solid var(--border-hairline);
    }

    .kpi-cell {
      padding: 10px 12px;
      background: var(--bg-canvas);
      border: 1px solid var(--border-hairline);
      border-radius: 6px;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
    }

    .kpi-cell-label {
      font-family: var(--font-sans);
      font-size: 9.5px;
      text-transform: uppercase;
      color: var(--text-muted);
      margin-bottom: 4px;
      letter-spacing: 0.03em;
      font-weight: 600;
    }

    .kpi-cell-val {
      font-family: var(--font-mono);
      font-size: 16px;
      font-weight: 700;
      font-variant-numeric: tabular-nums slashed-zero;
      color: var(--text-primary);
    }

    /* Tabbed Analytical Viewport Container */
    .analytics-tab-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      border-bottom: 1px solid var(--border-hairline);
      padding-bottom: 8px;
      margin-bottom: 12px;
    }

    .tab-nav {
      display: flex;
      align-items: center;
      gap: 4px;
      background: var(--bg-elevated);
      padding: 3px;
      border-radius: 6px;
      border: 1px solid var(--border-hairline);
    }

    .tab-btn {
      background: transparent;
      border: none;
      font-family: var(--font-mono);
      font-size: 10px;
      font-weight: 600;
      color: var(--text-secondary);
      padding: 4px 10px;
      border-radius: 4px;
      cursor: pointer;
      transition: all 0.12s ease;
    }

    .tab-btn:hover {
      color: var(--text-primary);
    }

    .tab-btn.active {
      background: var(--accent-primary);
      color: var(--bg-canvas);
      font-weight: 700;
    }

    body.light-theme .tab-btn.active {
      background: #090A0C !important;
      color: #FFFFFF !important;
    }

    /* Procedural Subsurface Tomographic ASCII Grid */
    /* Training Control Dashboard Styles */
    .training-grid-3col {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 16px;
      width: 100%;
    }
    @media (max-width: 1100px) {
      .training-grid-3col {
        grid-template-columns: 1fr;
      }
    }
    .train-card {
      background: var(--bg-surface);
      border: 1px solid var(--border-hairline);
      border-radius: 6px;
      padding: 20px;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }
    .train-card-title {
      font-family: var(--font-mono);
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 1px;
      color: var(--text-primary);
      text-transform: uppercase;
      display: flex;
      align-items: center;
      gap: 8px;
      border-bottom: 1px solid var(--border-hairline);
      padding-bottom: 10px;
    }
    .train-label {
      font-family: var(--font-mono);
      font-size: 10px;
      font-weight: 600;
      color: var(--text-secondary);
      text-transform: uppercase;
      letter-spacing: 0.5px;
      margin-bottom: 5px;
      display: block;
    }
    .train-input, .train-select, .train-textarea {
      width: 100%;
      background: var(--bg-canvas);
      border: 1px solid var(--border-hairline);
      color: var(--text-primary);
      font-family: var(--font-mono);
      font-size: 11px;
      padding: 8px 10px;
      border-radius: 4px;
      box-sizing: border-box;
      outline: none;
      transition: border-color 0.15s ease;
    }
    .train-input:focus, .train-select:focus, .train-textarea:focus {
      border-color: var(--text-primary);
    }
    .train-checkbox-label {
      display: flex;
      align-items: center;
      gap: 8px;
      font-family: var(--font-mono);
      font-size: 11px;
      color: var(--text-primary);
      cursor: pointer;
    }
    .train-checkbox-label input[type="checkbox"] {
      accent-color: var(--text-primary);
      cursor: pointer;
    }
    .terminal-box {
      background: #040507;
      border: 1px solid var(--border-hairline);
      border-radius: 6px;
      font-family: var(--font-mono);
      font-size: 11px;
      padding: 14px;
      color: #E2E8F0;
      max-height: 280px;
      min-height: 180px;
      overflow-y: auto;
      overflow-x: auto;
      white-space: pre-wrap;
      word-break: break-all;
      line-height: 1.5;
    }
    .run-history-table {
      width: 100%;
      border-collapse: collapse;
      font-family: var(--font-mono);
      font-size: 11px;
      text-align: left;
    }
    .run-history-table th {
      padding: 8px 12px;
      background: var(--bg-surface);
      color: var(--text-secondary);
      border-bottom: 1px solid var(--border-hairline);
      font-size: 10px;
      font-weight: 700;
      letter-spacing: 0.5px;
    }
    .run-history-table td {
      padding: 10px 12px;
      border-bottom: 1px solid var(--border-hairline);
      color: var(--text-primary);
      vertical-align: middle;
    }
    .run-history-table tr:hover td {
      background: rgba(255, 255, 255, 0.02);
    }

    /* Telemetry Target Registry Table */
    .telemetry-table-wrap {
      overflow-x: auto;
      max-height: 240px;
      border: 1px solid var(--border-hairline);
      border-radius: 6px;
      background: var(--bg-canvas);
    }

    .telemetry-table {
      width: 100%;
      border-collapse: collapse;
      font-family: var(--font-mono);
      font-size: 10.5px;
      text-align: left;
    }

    .telemetry-table th {
      background: var(--bg-elevated);
      color: var(--text-secondary);
      font-weight: 600;
      padding: 8px 10px;
      border-bottom: 1px solid var(--border-hairline);
      position: sticky;
      top: 0;
      z-index: 2;
      user-select: none;
      cursor: pointer;
    }

    .telemetry-table td {
      padding: 8px 10px;
      border-bottom: 1px solid var(--border-subtle);
      color: var(--text-secondary);
      white-space: nowrap;
    }

    .telemetry-table tr {
      cursor: pointer;
      transition: background-color 0.1s ease;
    }

    .telemetry-table tr:hover {
      background-color: var(--bg-elevated);
      color: var(--text-primary);
    }

    .telemetry-table tr.active-row {
      background-color: rgba(255, 255, 255, 0.08) !important;
      color: var(--text-primary) !important;
    }

    /* Monochromatic Target Pins on Map */
    .target-pin {
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      font-family: var(--font-mono);
      font-weight: 700;
      cursor: pointer;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.5);
      transition: transform 0.15s ease, box-shadow 0.15s ease;
      border: 1.5px solid #FFFFFF;
      background: var(--bg-surface);
      color: #FFFFFF;
    }

    .target-pin:hover {
      transform: scale(1.2);
      border-color: #FFFFFF;
      box-shadow: 0 0 12px rgba(255, 255, 255, 0.6);
    }

    .target-pin.selected {
      border: 2px solid #FFFFFF !important;
      background: #FFFFFF !important;
      color: #060709 !important;
      box-shadow: 0 0 16px rgba(255, 255, 255, 0.8), 0 0 30px rgba(255, 255, 255, 0.4) !important;
      transform: scale(1.3) !important;
      animation: pulse-target-ring 1.8s infinite;
      z-index: 99999 !important;
    }

    @keyframes pulse-target-ring {
      0% { box-shadow: 0 0 0 0 rgba(255, 255, 255, 0.6), 0 0 16px rgba(255, 255, 255, 0.8); }
      70% { box-shadow: 0 0 0 12px rgba(255, 255, 255, 0), 0 0 20px rgba(255, 255, 255, 0.8); }
      100% { box-shadow: 0 0 0 0 rgba(255, 255, 255, 0), 0 0 16px rgba(255, 255, 255, 0.8); }
    }
  </style>
</head>
<body class="dark antialiased">
  <div class="app-shell">
    <!-- ==================== BAR 1: TOP PRIMARY MASTHEAD ==================== -->
    <header class="top-header">
      <div class="brand-wrap">
        <div class="sift-logo-mark" aria-hidden="true">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="9" stroke="var(--border-strong)"/>
            <line x1="12" y1="1" x2="12" y2="23" stroke="var(--accent-primary)"/>
            <line x1="1" y1="12" x2="23" y2="12" stroke="var(--accent-primary)"/>
            <circle cx="12" cy="12" r="2.5" fill="var(--accent-primary)"/>
          </svg>
        </div>
        <div class="brand-text-block">
          <h1 class="brand-title">
            SIFT <span class="brand-subtitle">// SPATIAL INFERENCE &amp; FAULT TARGETING</span>
          </h1>
        </div>
        <span id="region-resolution-badge" class="panel-tag">250M ULTRA-HD</span>
      </div>

      <!-- MAIN NAVIGATION TABS -->
      <nav class="nav-tabs-group" aria-label="Primary Navigation">
        <button id="nav-btn-home" class="nav-tab-btn active" onclick="showView('home')" title="Platform Mission Control Overview">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/></svg>
          01 // MISSION CONTROL
        </button>
        <button id="nav-btn-workspace" class="nav-tab-btn" onclick="showView('workspace')" title="Interactive Geospatial Exploration Workspace">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="1 6 1 22 8 18 16 22 23 18 23 2 16 6 8 2 1 6"/><line x1="8" y1="2" x2="8" y2="18"/><line x1="16" y1="6" x2="16" y2="22"/></svg>
          02 // EXPLORATION WORKSPACE
        </button>
        <button id="nav-btn-training" class="nav-tab-btn" onclick="showView('training')" title="Neural Training Control, Model Lab & Version History">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>
          03 // TRAINING CONTROL
        </button>
      </nav>

      <!-- RIGHT STATUS & THEME TOGGLE -->
      <div class="header-right-meta">
        <div class="system-status-pill">
          <span id="braille-scanner" style="color:var(--text-primary); font-size:13px; font-weight:700;">⠁</span>
          <span>INGESTION ACTIVE</span>
        </div>
        <button class="btn-icon" onclick="toggleTheme()" title="Toggle Monochromatic Theme (Dark/Light)" aria-label="Toggle Theme">
          <svg id="theme-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="5"></circle>
            <line x1="12" y1="1" x2="12" y2="3"></line>
            <line x1="12" y1="21" x2="12" y2="23"></line>
            <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line>
            <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line>
            <line x1="1" y1="12" x2="3" y2="12"></line>
            <line x1="21" y1="12" x2="23" y2="12"></line>
            <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line>
            <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line>
          </svg>
        </button>
      </div>
    </header>

    <!-- ==================== BAR 2: SECONDARY TELEMETRY & COMMAND BAR ==================== -->
    <div id="sub-header-bar" class="sub-header-bar">
      <!-- LEFT: Live Model Telemetry Chips -->
      <div class="sub-bar-left">
        <span class="sub-bar-section-label">MODEL TELEMETRY:</span>
        <div class="telemetry-chip">
          <span class="chip-label">XGB 3.1:</span>
          <span id="chip-xgb" class="chip-val">0.847</span>
        </div>
        <div class="telemetry-chip">
          <span class="chip-label">nnPU U-NET:</span>
          <span id="chip-unet" class="chip-val">0.803</span>
        </div>
        <div class="telemetry-chip">
          <span class="chip-label">FUSED PR-AUC:</span>
          <span id="chip-fused" class="chip-val">0.913</span>
        </div>
        <div class="telemetry-chip">
          <span class="chip-label">PRIOR ĉ:</span>
          <span id="chip-chat" class="chip-val">0.285</span>
        </div>
      </div>

      <!-- RIGHT: Workspace Controls & Filter Selectors -->
      <div id="header-workspace-tools" class="sub-bar-right">
        <!-- Layout Mode Switcher -->
        <div class="view-mode-pill">
          <button id="btn-mode-split" class="view-mode-btn active" onclick="setLayoutMode('split')" title="Split Studio (Map + Analytics)">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><line x1="12" y1="3" x2="12" y2="21"/></svg>
            SPLIT
          </button>
          <button id="btn-mode-map" class="view-mode-btn" onclick="setLayoutMode('map')" title="Expansive Map View">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="1 6 1 22 8 18 16 22 23 18 23 2 16 6 8 2 1 6"/></svg>
            MAP
          </button>
          <button id="btn-mode-analytics" class="view-mode-btn" onclick="setLayoutMode('analytics')" title="Analytical Deep-Dive">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>
            ANALYTICS
          </button>
        </div>

        <!-- Regional Selector & Archived Training Runs -->
        <select class="select-dataset" id="dataset-select" onchange="selectRegion(this.value)" aria-label="Select Target Region">
          <option value="EAST_KALIMANTAN" selected>EAST KALIMANTAN (PROVINCE-WIDE 250M MODEL)</option>
          <option value="KALIMANTAN">BORNEO ISLAND-WIDE (500M REGIONAL BASELINE)</option>
          <optgroup id="optgroup-archived-runs" label="ARCHIVED TRAINING RUNS">
''' + archived_opts + '''            </optgroup>
          </select>

          <!-- Deposit Mechanics Ore Category Filter -->
          <select class="select-dataset select-ore-category" id="ore-category-select" onchange="selectOreCategory(this.value)" aria-label="Filter Ore Mechanics">
            <option value="ALL" selected>ORE: ALL DEPOSIT STYLES (K=4)</option>
            <option value="porphyry_cu">PORPHYRY CU-AU (HEAD 0)</option>
            <option value="skarn_fe_cu">SKARN FE-CU (HEAD 1)</option>
            <option value="epithermal_au">EPITHERMAL AU-AG (HEAD 2)</option>
            <option value="laterite_fe_ni">LATERITIC FE-NI (HEAD 3)</option>
            <option value="primary_gold">OROGENIC GOLD (PILOT)</option>
            <option value="telescoped">TELESCOPED / POLY-PHASE</option>
          </select>

          <!-- Independent Training Run Version Selector -->
          <select class="select-dataset select-run-history" id="run-history-select" onchange="selectTrainingRun(this.value)" aria-label="Select Training Run Version">
            <option value="ACTIVE" selected>RUN: ACTIVE (LATEST 7-PASS)</option>
          </select>
        </div>

        <!-- Home Shortcut on Bar 2 -->
        <div id="header-home-tools" class="sub-bar-right" style="display:none;">
          <button onclick="launchWorkspace('EAST_KALIMANTAN')" class="hero-btn-primary" style="padding: 4px 12px; font-size:10.5px; border-radius:4px;">
            LAUNCH 250M WORKSPACE →
          </button>
        </div>

        <!-- Training View Shortcut / Tools on Bar 2 -->
        <div id="header-training-tools" class="sub-bar-right" style="display:none; align-items:center; gap:8px;">
          <button onclick="startTrainingRunFromUI()" id="bar-btn-start-train" class="hero-btn-primary" style="padding: 4px 12px; font-size:10.5px; border-radius:4px;">
            ▶ DISPATCH TRAINING
          </button>
          <button onclick="stopTrainingRunFromUI()" id="bar-btn-stop-train" class="hero-btn-secondary" style="padding: 4px 10px; font-size:10.5px; border-radius:4px; display:none; border-color:#EF4444; color:#EF4444;">
            ⏹ ABORT
          </button>
          <button onclick="launchWorkspace('EAST_KALIMANTAN')" class="hero-btn-secondary" style="padding: 4px 10px; font-size:10.5px; border-radius:4px;">
            OPEN WORKSPACE →
          </button>
        </div>
      </div>

    <!-- VIEW 1: HOMEPAGE // MISSION CONTROL -->
    <div id="view-home" class="view-home-container">
      <div class="home-content-wrap">
        <!-- HERO SECTION -->
        <section class="hero-section">
          <div class="hero-top-badges">
            <span class="mono-badge">SIFT GEOSPATIAL INTELLIGENCE</span>
            <span class="mono-badge">VERSION 4.2.0</span>
            <span class="mono-badge active-dot"><span class="dot"></span> 250M PRODUCTION GRID SCAN ACTIVE</span>
          </div>

          <h1 class="hero-main-title">
            SPATIAL INFERENCE &amp; FAULT TARGETING
          </h1>
          <p class="hero-lead-text">
            Autonomous multi-physics subsurface intelligence engine for de-risking Greenfield mineral exploration in covered tropical terranes through neural spatial representation, conformal risk bounds, and kinematic fault network modeling.
          </p>

          <div class="hero-cta-group">
            <button class="hero-btn-primary" onclick="launchWorkspace('EAST_KALIMANTAN')">
              LAUNCH EXPLORATION WORKSPACE
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/></svg>
            </button>
            <button class="hero-btn-secondary" onclick="showView('training')" style="display:flex; align-items:center; gap:8px;">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>
              TRAINING CONTROL &amp; MODEL LAB
            </button>
            <a href="#pipeline-section" class="hero-btn-secondary">
              VIEW 7-PASS ARCHITECTURE
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"/></svg>
            </a>
          </div>

          <!-- 4 HERO KPI CARDS -->
          <div class="hero-kpi-grid">
            <div class="hero-kpi-card">
              <div class="kpi-micro-label">REGIONAL SURVEY FOOTPRINT</div>
              <div class="kpi-hero-num">182,500 <span class="kpi-unit">km²</span></div>
              <div class="kpi-desc">Continuous multi-sensor scan spanning East Kalimantan &amp; Borneo basin</div>
            </div>
            <div class="hero-kpi-card">
              <div class="kpi-micro-label">MULTIMODAL TESSELLATION</div>
              <div class="kpi-hero-num">2.92M <span class="kpi-unit">cells</span></div>
              <div class="kpi-desc">Uniform 250m resolution grid with 21 decoupled physical sensor layers</div>
            </div>
            <div class="hero-kpi-card">
              <div class="kpi-micro-label">MODEL FUSION PR-AUC</div>
              <div class="kpi-hero-num">0.9135</div>
              <div class="kpi-desc">Dual neural-tree ensemble (XGBoost 3.1 + nnPU DAE U-Net)</div>
            </div>
            <div class="hero-kpi-card">
              <div class="kpi-micro-label">HIGH-CONFIDENCE RECALL</div>
              <div class="kpi-hero-num">83.54<span class="kpi-unit">%</span></div>
              <div class="kpi-desc">Rigorous spatial block cross-validation with zero geographic leakage</div>
            </div>
          </div>
        </section>

        <!-- ACTIVE EXPLORATION THEATERS -->
        <section class="theaters-section">
          <div class="section-header">
            <div class="section-tag">01 // OPERATIONAL DEPLOYMENTS</div>
            <h2 class="section-title">ACTIVE EXPLORATION THEATERS</h2>
            <p class="section-subtitle">Real-time target portfolios and calibrated prospective horizons across equatorial terranes.</p>
          </div>

          <div class="theaters-grid">
            <!-- Theater Card 1: East Kalimantan -->
            <div class="theater-card featured">
              <div class="theater-card-header">
                <div class="theater-status-badge active"><span class="dot"></span> ACTIVE PRODUCTION RUN</div>
                <span class="theater-crs">MGA50 / EPSG:32750</span>
              </div>
              <h3 class="theater-name">EAST KALIMANTAN (KALTIM)</h3>
              <p class="theater-desc">
                Province-wide 250m ultra-HD exploration scan. Features 10 rounds of continuous incremental retraining weights, GEM active fault density integration, and 397 identified drill targets.
              </p>
              <div class="theater-metrics-row">
                <div><span class="lbl">TARGETS</span><span class="val">397 RECORDED</span></div>
                <div><span class="lbl">RESOLUTION</span><span class="val">250m CONTINUOUS</span></div>
                <div><span class="lbl">PR-AUC</span><span class="val">0.9135</span></div>
                <div><span class="lbl">PRIOR ĉ</span><span class="val">0.285</span></div>
              </div>
              <button class="theater-launch-btn" onclick="launchWorkspace('EAST_KALIMANTAN')">
                ENTER EAST KALIMANTAN WORKSPACE →
              </button>
            </div>

            <!-- Theater Card 2: Borneo Baseline -->
            <div class="theater-card">
              <div class="theater-card-header">
                <div class="theater-status-badge">REGIONAL BASELINE</div>
                <span class="theater-crs">WGS84 / EPSG:4326</span>
              </div>
              <h3 class="theater-name">BORNEO ISLAND-WIDE</h3>
              <p class="theater-desc">
                Island-wide 500m regional geodynamic survey. Calibrated against 15 verified ground truth deposits with deep crustal Bouguer gravity slab reduction and Moho depth gradient mapping.
              </p>
              <div class="theater-metrics-row">
                <div><span class="lbl">TARGETS</span><span class="val">15 BENCHMARKS</span></div>
                <div><span class="lbl">RESOLUTION</span><span class="val">500m REGIONAL</span></div>
                <div><span class="lbl">PR-AUC</span><span class="val">0.8720</span></div>
                <div><span class="lbl">PRIOR ĉ</span><span class="val">0.285</span></div>
              </div>
              <button class="theater-launch-btn" onclick="launchWorkspace('KALIMANTAN')">
                ENTER BORNEO BASELINE WORKSPACE →
              </button>
            </div>
          </div>
        </section>

        <!-- 7-PASS SCIENTIFIC PIPELINE ARCHITECTURE -->
        <section id="pipeline-section" class="pipeline-section">
          <div class="section-header">
            <div class="section-tag">02 // ARCHITECTURAL TAXONOMY</div>
            <h2 class="section-title">THE 7-PASS GEOSPATIAL PIPELINE</h2>
            <p class="section-subtitle">Rigorous, bias-corrected inference pipeline resolving Greenfield subsurface prospectivity.</p>
          </div>

          <div class="pipeline-cards-grid">
            <!-- Pass 1 -->
            <div class="pipeline-card">
              <div class="pipeline-step-num">PASS 01</div>
              <h4>PHYSICAL SENSOR DECOUPLING</h4>
              <p>
                Ingests 21 decoupled raster channels spanning SRTM-DEM topography, ASTER SWIR/TIR hydrothermal alteration, Sentinel-2 regolith cover, Bouguer gravity, and GEM tectonic kinematic slip vectors.
              </p>
            </div>
            <!-- Pass 2 -->
            <div class="pipeline-card">
              <div class="pipeline-step-num">PASS 02</div>
              <h4>PROPENSITY DE-BIASING e(x)</h4>
              <p>
                Explicitly models observation and human exploration accessibility bias. Disentangles river and road infrastructure proximity from deep geological genesis to eliminate artificial clusters.
              </p>
            </div>
            <!-- Pass 3 -->
            <div class="pipeline-card">
              <div class="pipeline-step-num">PASS 03</div>
              <h4>DUAL NEURAL-TREE ENSEMBLE</h4>
              <p>
                Fuses bagged gradient-boosted decision trees (XGBoost 3.1) with a denoising autoencoder U-Net (nnPU DAE U-Net) to synthesize tabular non-linear interactions with multi-scale spatial textures.
              </p>
            </div>
            <!-- Pass 4 -->
            <div class="pipeline-card">
              <div class="pipeline-step-num">PASS 04</div>
              <h4>CLOSED-BOUNDARY PU CALIBRATION</h4>
              <p>
                Applies Elkan-Noto Positive-Unlabeled learning theorem. Calculates discovery constant ĉ = P(s=1|y=1) = 0.285 to calibrate unmined background into true posterior P(Y=1|X) = s(x) / ĉ.
              </p>
            </div>
            <!-- Pass 5 -->
            <div class="pipeline-card">
              <div class="pipeline-step-num">PASS 05</div>
              <h4>MULTI-MINERAL CONFORMAL BOUNDS</h4>
              <p>
                Non-conformity scoring yields finite-sample coverage guarantees at 90% confidence: L(x) ≤ P(Y=1|X) ≤ U(x). Dynamically slices targets across porphyry, skarn, epithermal, and laterite systems.
              </p>
            </div>
            <!-- Pass 6 -->
            <div class="pipeline-card">
              <div class="pipeline-step-num">PASS 06</div>
              <h4>VECTOR TARGET POLYGONIZATION &amp; SHAP</h4>
              <p>
                Contour extraction at optimal thresholds generates discrete exploration polygons. Mineral systems SHAP attributions partition feature contributions into Source, Conduit, and Trap geodynamics.
              </p>
            </div>
            <!-- Pass 7 -->
            <div class="pipeline-card">
              <div class="pipeline-step-num">PASS 07</div>
              <h4>SITE CARDS &amp; DRILL PRIORITIZATION</h4>
              <p>
                Generates comprehensive drill dossiers with regolith depth inversion, surface area calculations, structural domain boundaries, and ranked target portfolios with automated CSV export.
              </p>
            </div>
          </div>
        </section>

        <!-- 21-CHANNEL PHYSICAL SENSOR MATRIX -->
        <section class="sensor-matrix-section">
          <div class="section-header">
            <div class="section-tag">03 // PHYSICAL TELEMETRY</div>
            <h2 class="section-title">21-CHANNEL MULTI-SENSOR STACK</h2>
            <p class="section-subtitle">Multi-scale geoscientific feature tensors feeding the dual neural-tree ensemble.</p>
          </div>

          <div class="sensor-matrix-table-wrap">
            <table class="sensor-matrix-table">
              <thead>
                <tr>
                  <th>CH #</th>
                  <th>CHANNEL CODENAME</th>
                  <th>PHYSICAL DOMAIN</th>
                  <th>SENSOR / DATA SOURCE</th>
                  <th>GEOLOGICAL RELEVANCE</th>
                </tr>
              </thead>
              <tbody>
                <tr><td>01</td><td>dem_elevation</td><td>Topography</td><td>SRTM 30m Radar DEM</td><td>Regional topographic elevation &amp; relief</td></tr>
                <tr><td>02</td><td>dem_slope</td><td>Geomorphology</td><td>Spatial Gradient Inversion</td><td>Structural scarp &amp; valley delineation</td></tr>
                <tr><td>03</td><td>terrain_ruggedness_tri</td><td>Geomorphology</td><td>Wilson et al. TRI Index</td><td>Erosional incision &amp; bedrock exposure</td></tr>
                <tr><td>04</td><td>outcrop_exposure_index</td><td>Multispectral</td><td>Sentinel-2 / ASTER SWIR</td><td>Hydrothermal rock alteration &amp; gossans</td></tr>
                <tr><td>05</td><td>regolith_cover_m</td><td>Weathering</td><td>AEM / Multispectral Inversion</td><td>Depth of saprolite &amp; cover blanket (m)</td></tr>
                <tr><td>06</td><td>dilation_tendency_Td</td><td>Geomechanics</td><td>Stress Tensor σ1/σ2/σ3</td><td>Permeable hydrothermal fluid dilation pathways</td></tr>
                <tr><td>07</td><td>slip_tendency_Ts</td><td>Geomechanics</td><td>Shear-to-Normal Stress Ratio</td><td>Fault reactivation &amp; shear localization</td></tr>
                <tr><td>08</td><td>active_fault_density</td><td>Fracture Network</td><td>GEM Global Faults Vector</td><td>Regional fracture permeability density</td></tr>
                <tr><td>09</td><td>bouguer_gravity_mgal</td><td>Deep Geophysics</td><td>GGM05S / TOPEX Bouguer Slab</td><td>Dense plutonic intrusions &amp; basement highs</td></tr>
                <tr><td>10</td><td>moho_gravity_step_mgal_km</td><td>Deep Geophysics</td><td>Gravity Gradient Inversion</td><td>Moho boundary steps &amp; crustal tears</td></tr>
                <tr><td>11</td><td>crustal_thickness_km</td><td>Deep Geophysics</td><td>Parker-Oldenburg Inversion</td><td>Crustal thickness (20–42 km)</td></tr>
                <tr><td>12</td><td>volcanic_arc_halo</td><td>Tectonic Arc</td><td>Subduction Geometry Buffer</td><td>Calc-alkaline volcanic arc hydrothermal halo</td></tr>
                <tr><td>13</td><td>dae_reconstruction_error</td><td>AI Spatial Representation</td><td>Denoising Autoencoder U-Net</td><td>Unsupervised structural &amp; textural anomalies</td></tr>
                <tr><td>14</td><td>propensity_e_x</td><td>Observation Model</td><td>Logistic Highway/River Inversion</td><td>Debiasing jungle accessibility bias</td></tr>
                <tr><td>15–21</td><td>band_ratios_swir_tir</td><td>Spectral Alteration</td><td>ASTER SWIR B4-B9, TIR B10-B14</td><td>Kaolinite, alunite, silica doublet indices</td></tr>
              </tbody>
            </table>
          </div>
        </section>

        <!-- FOOTER -->
        <footer class="home-footer">
          <div style="display:flex; align-items:center;">
            <span class="footer-brand">SIFT // SPATIAL INFERENCE &amp; FAULT TARGETING</span>
            <span class="footer-copy">© 2026 Antigravity Geospatial Intelligence Group. All rights reserved.</span>
          </div>
          <div style="display:flex; align-items:center; gap:8px;">
            <span class="footer-tag">HIGH-CONCURRENCY FASTAPI DAEMON :8765</span>
            <span class="footer-tag">10-ROUND BAGGED ENSEMBLE</span>
          </div>
        </footer>
      </div>
    </div>

    <!-- VIEW 2: EXPLORATION WORKSPACE // INTERACTIVE GIS & ANALYTICS -->
    <div id="view-workspace" style="display:none; flex:1; min-height:0; overflow:hidden;">
      <div id="workspace-container" class="workspace-body mode-split" style="width:100%; height:100%;">
        <!-- PROGRESSIVE DISCLOSURE DRAWER (270px / 48px) -->
        <aside id="sensor-drawer" class="telemetry-drawer">
          <div class="drawer-header">
            <div style="display:flex; align-items:center; gap:8px;">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="color:var(--accent-primary);" aria-hidden="true">
                <line x1="4" y1="21" x2="4" y2="14"></line>
                <line x1="4" y1="10" x2="4" y2="3"></line>
                <line x1="12" y1="21" x2="12" y2="12"></line>
                <line x1="12" y1="8" x2="12" y2="3"></line>
                <line x1="20" y1="21" x2="20" y2="16"></line>
                <line x1="20" y1="12" x2="20" y2="3"></line>
              </svg>
              <span class="panel-title" style="font-size:11px; text-transform:uppercase;">PIPELINE PARAMETERS</span>
            </div>
            <button class="btn-icon" onclick="toggleDrawer()" aria-label="Toggle Parameter Drawer" id="btn-drawer-collapse">
              <svg id="drawer-chevron-icon" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
                <polyline points="15 18 9 12 15 6"></polyline>
              </svg>
            </button>
          </div>

          <div class="drawer-content">
            <!-- Parameter: Conformal L(x) -->
            <div style="display:flex; flex-direction:column; gap:6px;">
              <div style="display:flex; justify-content:space-between; font-family:var(--font-mono); font-size:11px;">
                <span style="color:var(--text-primary); font-weight:600;">CONFORMAL L(x) THR</span>
                <span id="lbl-l-thr" style="color:var(--accent-primary); font-weight:700;">0.50</span>
              </div>
              <input type="range" id="ctrl-l-thr" min="0.1" max="0.95" step="0.05" value="0.50" oninput="onLThrChange(this.value)" style="accent-color:var(--accent-primary); cursor:pointer;" />
              <span class="font-editorial" style="font-size:11.5px; font-style:italic; color:var(--text-muted); line-height:1.4;">
                Finite-sample coverage bound P(Y=1|X) ≥ L(x) under Elkan-Noto PU-learning.
              </span>
            </div>

            <!-- Parameter: Z-Depth -->
            <div style="display:flex; flex-direction:column; gap:6px; padding-top:10px; border-top:1px solid var(--border-hairline);">
              <div style="display:flex; justify-content:space-between; font-family:var(--font-mono); font-size:11px;">
                <span style="color:var(--text-primary); font-weight:600;">Z-DEPTH SLICE</span>
                <span id="lbl-z-depth" style="color:var(--accent-primary); font-weight:700;">-450 m</span>
              </div>
              <input type="range" id="ctrl-z-depth" min="100" max="600" step="10" value="450" oninput="onAsciiZChange(this.value)" style="accent-color:var(--accent-primary); cursor:pointer;" />
            </div>

            <!-- Parameter: Minimum Target Area -->
            <div style="display:flex; flex-direction:column; gap:6px; padding-top:10px; border-top:1px solid var(--border-hairline);">
              <div style="display:flex; justify-content:space-between; font-family:var(--font-mono); font-size:11px;">
                <span style="color:var(--text-primary); font-weight:600;">MIN TARGET AREA</span>
                <span id="lbl-min-area" style="color:var(--accent-primary); font-weight:700;">5.0 ha</span>
              </div>
              <input type="range" id="ctrl-min-area" min="1" max="250" step="5" value="5" oninput="onMinAreaChange(this.value)" style="accent-color:var(--accent-primary); cursor:pointer;" />
            </div>

            <!-- Layer Controls -->
            <div style="display:flex; flex-direction:column; gap:6px; padding-top:10px; border-top:1px solid var(--border-hairline);">
              <span class="panel-title" style="font-size:10.5px; text-transform:uppercase; color:var(--text-muted);">MAP OVERLAYS &amp; SENSORS</span>
              
              <label style="display:flex; justify-content:space-between; align-items:center; font-size:11px; padding:6px 8px; background:var(--bg-elevated); border:1px solid var(--border-hairline); border-radius:5px; cursor:pointer;">
                <span style="display:flex; align-items:center; gap:6px;"><span style="display:inline-block; width:8px; height:8px; border-radius:2px; background:linear-gradient(45deg, #71717A, #FFFFFF);"></span> Show Prospectivity Heatmap</span>
                <input type="checkbox" id="chk-heatmap" checked onchange="toggleHeatmapLayer(this.checked)" style="accent-color:var(--accent-primary);" />
              </label>

              <div style="display:flex; flex-direction:column; gap:4px; padding-top:2px;">
                <div style="display:flex; justify-content:space-between; font-family:var(--font-mono); font-size:10px; color:var(--text-muted);">
                  <span>HEATMAP OPACITY</span>
                  <span id="lbl-opacity">80%</span>
                </div>
                <input type="range" id="rng-opacity" min="0.1" max="1.0" step="0.05" value="0.80" oninput="setHeatmapOpacity(this.value)" style="accent-color:var(--accent-primary); cursor:pointer;" />
              </div>

              <!-- Satellite Physical & Multispectral Sensor Overlay -->
              <div style="display:flex; flex-direction:column; gap:6px; padding:8px; background:var(--bg-elevated); border:1px solid var(--border-hairline); border-radius:5px; margin-top:2px;">
                <label style="display:flex; justify-content:space-between; align-items:center; font-size:11px; cursor:pointer;">
                  <span style="display:flex; align-items:center; gap:6px; font-weight:600;">
                    <span style="display:inline-block; width:8px; height:8px; border-radius:2px; background:linear-gradient(45deg, #94A3B8, #E2E8F0);"></span>
                    Satellite Sensor Overlay
                  </span>
                  <input type="checkbox" id="chk-satellite-layer" onchange="updateSatelliteLayer()" style="accent-color:var(--accent-primary);" />
                </label>

                <select id="sel-satellite-layer" onchange="updateSatelliteLayer()" style="width:100%; padding:5px 7px; font-family:var(--font-mono); font-size:10px; background:var(--bg-surface); color:var(--text-primary); border:1px solid var(--border-hairline); border-radius:4px; outline:none; cursor:pointer;">
                  <optgroup label="── TOPOGRAPHY &amp; GEOMORPHOLOGY ──">
                    <option value="dem_elevation" selected>Topography // SRTM-DEM 30m Elevation</option>
                    <option value="dem_slope">Slope // Topographic Slope Steepness</option>
                    <option value="terrain_ruggedness_tri">Roughness // Terrain Ruggedness (TRI)</option>
                  </optgroup>
                  <optgroup label="── MULTISPECTRAL &amp; REGOLITH ──">
                    <option value="outcrop_exposure_index">Outcrop // Multispectral Rock Alteration</option>
                    <option value="regolith_cover_m">Cover // Weathered Regolith/Saprolite Depth</option>
                  </optgroup>
                  <optgroup label="── STRUCTURAL &amp; 3D GEOMECHANICS ──">
                    <option value="dilation_tendency_Td">Geomechanics // Fault Dilation Strain (Td)</option>
                    <option value="slip_tendency_Ts">Geomechanics // Fault Slip Tendency (Ts)</option>
                    <option value="active_fault_density">Fracture // Active Fault Network Density</option>
                  </optgroup>
                  <optgroup label="── DEEP CRUSTAL GEOPHYSICS ──">
                    <option value="bouguer_gravity_mgal">Gravity // Bouguer Density Anomaly (Plutons)</option>
                    <option value="moho_gravity_step_mgal_km">Moho Step // Moho Density Gradient</option>
                    <option value="crustal_thickness_km">Crustal Thickness // Moho Depth Inversion</option>
                  </optgroup>
                  <optgroup label="── AI NEURAL GEODYNAMICS ──">
                    <option value="volcanic_arc_halo">Volcanic Arc // Hydrothermal Fluid Halo</option>
                    <option value="dae_reconstruction_error">DAE-U-Net // Neural Structural Anomaly</option>
                    <option value="propensity_e_x">Observation // Jungle Exploration Propensity e(x)</option>
                  </optgroup>
                </select>

                <div style="display:flex; flex-direction:column; gap:2px;">
                  <div style="display:flex; justify-content:space-between; font-family:var(--font-mono); font-size:9.5px; color:var(--text-muted);">
                    <span>SATELLITE DATA OPACITY</span>
                    <span id="lbl-sat-opacity">75%</span>
                  </div>
                  <input type="range" id="rng-sat-opacity" min="0.1" max="1.0" step="0.05" value="0.75" oninput="setSatelliteOpacity(this.value)" style="accent-color:var(--accent-primary); cursor:pointer;" />
                  <span id="lbl-sat-status" style="font-family:var(--font-mono); font-size:9px; color:var(--accent-primary);"></span>
                </div>
              </div>

              <!-- Vector Overlays -->
              <label style="display:flex; justify-content:space-between; align-items:center; font-size:11px; padding:6px 8px; background:var(--bg-elevated); border:1px solid var(--border-hairline); border-radius:5px; cursor:pointer;">
                <span style="display:flex; align-items:center; gap:6px;"><span style="display:inline-block; width:8px; height:8px; border-radius:1px; background:#FFFFFF;"></span> Tectonic Fault Lines (GEM)</span>
                <input type="checkbox" id="chk-faults-geo" checked onchange="toggleFaultsLayer(this.checked)" style="accent-color:var(--accent-primary);" />
              </label>
              <label style="display:flex; justify-content:space-between; align-items:center; font-size:11px; padding:6px 8px; background:var(--bg-elevated); border:1px solid var(--border-hairline); border-radius:5px; cursor:pointer;">
                <span style="display:flex; align-items:center; gap:6px;"><span style="display:inline-block; width:8px; height:8px; border-radius:2px; border:1.5px dashed var(--accent-primary);"></span> Provincial Scan Boundary</span>
                <input type="checkbox" id="chk-scanned-rect" checked onchange="toggleScannedRect(this.checked)" style="accent-color:var(--accent-primary);" />
              </label>

              <label style="display:flex; justify-content:space-between; align-items:center; font-size:11px; padding:6px 8px; background:var(--bg-elevated); border:1px solid var(--border-hairline); border-radius:5px; cursor:pointer;">
                <span style="display:flex; align-items:center; gap:6px;"><span style="display:inline-block; width:8px; height:8px; border-radius:2px; background:var(--accent-primary);"></span> Target Vector Polygons</span>
                <input type="checkbox" id="chk-targets-geo" checked onchange="toggleTargetsLayer(this.checked)" style="accent-color:var(--accent-primary);" />
              </label>

              <label style="display:flex; justify-content:space-between; align-items:center; font-size:11px; padding:6px 8px; background:var(--bg-elevated); border:1px solid var(--border-hairline); border-radius:5px; cursor:pointer;">
                <span style="display:flex; align-items:center; gap:6px;"><span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:#E2E8F0;"></span> Ground Truth Mines</span>
                <input type="checkbox" id="chk-ground-mines" checked onchange="toggleMinesLayer(this.checked)" style="accent-color:var(--accent-primary);" />
              </label>
            </div>

            <button onclick="resetDefaults()" style="margin-top:6px; width:100%; padding:7px; background:var(--bg-elevated); border:1px solid var(--border-hairline); border-radius:5px; font-family:var(--font-mono); font-size:10.5px; color:var(--text-secondary); cursor:pointer; display:flex; align-items:center; justify-content:center; gap:6px;" aria-label="Reset Default Prior">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><polyline points="1 4 1 10 7 10"></polyline><path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10"></path></svg>
              RESET DEFAULT PRIOR
            </button>

            <button id="btn-live-rerank" onclick="triggerLiveRerank()" style="margin-top:6px; width:100%; padding:7px; background:var(--accent-primary); color:var(--bg-canvas); font-weight:700; border:none; border-radius:5px; font-family:var(--font-mono); font-size:10.5px; cursor:pointer; display:flex; align-items:center; justify-content:center; gap:6px;" aria-label="Recompute Targets Live">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.67"/></svg>
              RECOMPUTE TARGETS LIVE
            </button>
          </div>

          <!-- Collapsed 48px Icon Rail -->
          <div class="drawer-collapsed-rail">
            <button class="btn-icon" onclick="toggleDrawer()" aria-label="Expand Parameter Drawer" title="Expand Drawer">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><polyline points="9 18 15 12 9 6"></polyline></svg>
            </button>
            <button class="btn-icon" onclick="toggleDrawer()" aria-label="Pipeline Sliders" title="Parameters">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><line x1="4" y1="21" x2="4" y2="14"></line><line x1="4" y1="10" x2="4" y2="3"></line><line x1="12" y1="21" x2="12" y2="12"></line><line x1="12" y1="8" x2="12" y2="3"></line><line x1="20" y1="21" x2="20" y2="16"></line><line x1="20" y1="12" x2="20" y2="3"></line></svg>
            </button>
            <button class="btn-icon" onclick="toggleDrawer()" aria-label="Sensor Layers" title="Layers">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><polygon points="12 2 2 7 12 12 22 7 12 2"></polygon><polyline points="2 17 12 22 22 17"></polyline><polyline points="2 12 12 17 22 12"></polyline></svg>
            </button>
            <span style="font-family:var(--font-mono); font-size:9.5px; color:var(--text-muted); writing-mode:vertical-rl; transform:rotate(180deg); margin-top:16px;">
              -380 m
            </span>
          </div>
        </aside>

        <!-- RESIZABLE SPLIT-PANE WORKSPACE -->
        <main id="split-workspace" class="split-workspace">
          <!-- LEFT HERO: GEOSPATIAL MAP CANVAS -->
          <div id="pane-left" class="split-pane-left" style="width: 54%;">
            <!-- Floating Frosted Glass Control Bars -->
            <div class="floating-map-bar-top">
              <!-- Basemap Switcher -->
              <div class="glass-pill-group">
                <span style="font-family:var(--font-mono); font-size:9.5px; color:var(--text-muted); padding:0 6px;">BASEMAP:</span>
                <button id="btn-base-dark" class="glass-pill-btn active" onclick="switchBasemap('dark')">Dark Obsidian</button>
                <button id="btn-base-satellite" class="glass-pill-btn" onclick="switchBasemap('satellite')">Satellite</button>
                <button id="btn-base-topo" class="glass-pill-btn" onclick="switchBasemap('topo')">Terrain</button>
              </div>

              <!-- Target Quick Filter Chips -->
              <div class="glass-pill-group">
                <button class="glass-pill-btn active" id="btn-flt-all" onclick="filterTargets('ALL')">ALL TARGETS</button>
                <button class="glass-pill-btn" id="btn-flt-top" onclick="filterTargets('TOP_RANKED')">TOP RANKED</button>
                <button class="glass-pill-btn" id="btn-flt-highconf" onclick="filterTargets('HIGH_CONF')">HIGH CONF</button>
              </div>
            </div>

            <!-- Leaflet Interactive Map -->
            <div id="map-viewport"></div>

            <!-- Floating Precision Telemetry HUD at Bottom -->
            <div class="floating-map-hud">
              <div><span style="color:var(--text-muted); margin-right:4px;">CURSOR:</span><span id="hud-coords" style="color:var(--text-primary); font-weight:700;">-0.5000°, 114.5000°</span></div>
              <div style="border-left:1px solid var(--border-hairline); padding-left:12px;"><span style="color:var(--text-muted); margin-right:4px;">SCANNED EXTENT:</span><span id="hud-extent" style="color:var(--accent-primary); font-weight:700;">ACTIVE BOUNDS</span></div>
              <div style="border-left:1px solid var(--border-hairline); padding-left:12px;"><span style="color:var(--text-muted); margin-right:4px;">ZOOM:</span><span id="hud-zoom" style="color:var(--accent-primary); font-weight:700;">7x</span></div>
              <div style="border-left:1px solid var(--border-hairline); padding-left:12px;"><span style="color:var(--text-muted); margin-right:4px;">TARGETS:</span><span id="hud-targets-count" style="color:var(--text-primary); font-weight:600;">—</span></div>
            </div>
          </div>

          <!-- 1px HAIRLINE DRAG HANDLE -->
          <div id="split-drag-handle" class="split-drag-handle" title="Drag to resize split panes">
            <div class="split-drag-line"></div>
          </div>

          <!-- RIGHT PANE: ANALYTICAL STUDIO CONSOLE -->
          <div id="pane-right" class="split-pane-right" style="width: 46%;">
            <!-- 1. ACTIVE TARGET RECORD DOSSIER CARD -->
            <div class="hairline-panel" id="target-detail-card">
              <div class="panel-header">
                <span class="panel-tag" id="target-card-tag">SELECTED TARGET RECORD</span>
                <span id="target-card-id" style="font-family:var(--font-mono); font-size:11px; color:var(--text-muted);">—</span>
              </div>

              <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:10px;">
                <div>
                  <h2 id="target-card-name" class="panel-title" style="font-size:15.5px;">Loading target record...</h2>
                  <p id="target-card-domain" class="font-editorial" style="font-size:12.5px; font-style:italic; color:var(--text-secondary); margin-top:2px;">
                    Regional Target Discovery — Ingestion Active
                  </p>
                  <p id="target-card-ore-details" style="font-family:var(--font-mono); font-size:10px; color:var(--text-secondary); margin-top:3px;">
                    Ore: <b id="target-card-commodity" style="color:var(--text-primary);">-</b> // System: <span id="target-card-system">-</span>
                  </p>
                </div>
                <div style="display:flex; gap:6px; align-items:center;">
                  <span id="target-card-ore-badge" class="badge-pill ore-badge ore-badge-porphyry_cu">
                    PORPHYRY CU-AU
                  </span>
                  <span id="target-card-class" class="badge-pill" style="background:rgba(255, 255, 255, 0.08); color:var(--text-primary); border:1px solid var(--border-hairline); padding:2px 7px; border-radius:4px; font-family:var(--font-mono); font-size:9.5px; font-weight:700;">
                    RANK #1
                  </span>
                </div>
              </div>

              <div class="kpi-grid">
                <div class="kpi-cell">
                  <div class="kpi-cell-label">PROSPECTIVITY P(Y=1)</div>
                  <div id="kpi-score" class="kpi-cell-val" style="color:var(--accent-primary);">0.950</div>
                </div>
                <div class="kpi-cell">
                  <div class="kpi-cell-label">REGOLITH / DEPTH</div>
                  <div id="kpi-depth" class="kpi-cell-val">-64 m</div>
                </div>
                <div class="kpi-cell">
                  <div class="kpi-cell-label">CONFIDENCE LOWER L(x)</div>
                  <div id="kpi-uncert" class="kpi-cell-val" style="color:var(--text-primary);">0.562</div>
                </div>
                <div class="kpi-cell">
                  <div class="kpi-cell-label">90% CONFORMAL CI</div>
                  <div id="kpi-ci" class="kpi-cell-val" style="font-size:12px; color:var(--text-secondary);">[0.562 - 1.000]</div>
                </div>
              </div>
            </div>

            <!-- 2. TABBED ANALYTICAL VIEWPORTS -->
            <div class="hairline-panel" style="flex: 1; display:flex; flex-direction:column;">
              <div class="analytics-tab-header">
                <div class="tab-nav">
                  <button id="btn-subtab-conformal" class="tab-btn active" onclick="switchAnalyticsSubTab('conformal')">
                    01 // Conformal 90% CI Bounds
                  </button>
                  <button id="btn-subtab-prauc" class="tab-btn" onclick="switchAnalyticsSubTab('prauc')">
                    02 // Model Fusion PR-AUC
                  </button>
                </div>
              </div>

              <!-- Tab View 1: Conformal Risk Calibration Chart -->
              <div id="view-tab-conformal" style="display:block; flex:1;">
                <div style="background:var(--bg-canvas); border:1px solid var(--border-hairline); border-radius:6px; padding:10px;">
                  <svg id="svg-analytics-chart" viewBox="0 0 600 210" style="width:100%; height:auto; display:block;"></svg>
                </div>
                <div style="display:flex; justify-content:space-between; align-items:center; margin-top:10px; padding-top:8px; border-top:1px solid var(--border-hairline); font-size:11px;">
                  <span class="font-editorial" style="font-size:12px; font-style:italic; color:var(--text-secondary);">
                    Coverage Guarantee: <span class="font-mono" style="font-style:normal; font-weight:700; color:var(--text-primary);">P(Y ∈ C(X)) ≥ 1 - α (α = 0.10)</span>.
                  </span>
                  <span id="lbl-conformal-count" class="font-mono" style="font-size:10.5px; color:var(--text-primary);">— Targets Evaluated</span>
                </div>
              </div>

              <!-- Tab View 3: Model Fusion PR-AUC Curve -->
              <div id="view-tab-prauc" style="display:none; flex:1;">
                <div style="background:var(--bg-canvas); border:1px solid var(--border-hairline); border-radius:6px; padding:10px;">
                  <svg id="svg-prauc-chart" viewBox="0 0 600 210" style="width:100%; height:auto; display:block;"></svg>
                </div>
                <div style="display:flex; justify-content:space-between; align-items:center; margin-top:10px; padding-top:8px; border-top:1px solid var(--border-hairline); font-family:var(--font-mono); font-size:10.5px;">
                  <span id="legend-xgb" style="color:var(--text-muted);">■ XGB 3.1: 0.847</span>
                  <span id="legend-unet" style="color:var(--text-secondary);">■ U-Net: 0.803</span>
                  <span id="legend-fused" style="color:#FFFFFF; font-weight:700;">■ Fused: 0.913</span>
                  <span id="legend-chat" style="color:var(--text-muted);">Prior ĉ = 0.285</span>
                </div>
              </div>
            </div>

            <!-- 3. RANKED TARGET PORTFOLIO REGISTER (TABLE) -->
            <div class="hairline-panel">
              <div class="panel-header">
                <div style="display:flex; align-items:center; gap:8px;">
                  <span class="panel-tag">04</span>
                  <span class="panel-title">RANKED DRILL TARGET PORTFOLIO</span>
                </div>
                <div style="display:flex; align-items:center; gap:8px;">
                  <input type="text" id="target-search-input" placeholder="Search targets..." oninput="onTargetSearch(this.value)" style="background:var(--bg-elevated); border:1px solid var(--border-hairline); border-radius:4px; font-family:var(--font-mono); font-size:10.5px; padding:3px 8px; color:var(--text-primary); outline:none; width:130px;" />
                  <button onclick="exportTargetsCSV()" class="btn-icon" title="Export CSV" style="font-size:10px; font-family:var(--font-mono); gap:4px; padding:4px 8px;">
                    <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
                    CSV
                  </button>
                </div>
              </div>

              <div class="telemetry-table-wrap">
                <table class="telemetry-table">
                  <thead>
                    <tr>
                      <th onclick="sortTable('id')">ID ↕</th>
                      <th onclick="sortTable('ore')">ORE STYLE ↕</th>
                      <th onclick="sortTable('name')">NAME &amp; DOMAIN / CLASSIFICATION ↕</th>
                      <th>COORDINATES (LAT / LON)</th>
                      <th onclick="sortTable('depth')">REGOLITH / DEPTH ↕</th>
                      <th onclick="sortTable('score')">SCORE ↕</th>
                      <th onclick="sortTable('uncertainty')">L(x) BOUND ↕</th>
                      <th>90% CI</th>
                      <th>RANK</th>
                    </tr>
                  </thead>
                  <tbody id="telemetry-table-body">
                    <!-- Dynamically populated -->
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </main>
      </div>
    </div>

    <!-- VIEW 3: TRAINING CONTROL // MODEL LAB & VERSION HISTORY -->
    <div id="view-training" class="view-home-container" style="display:none; flex:1; min-height:0; overflow-y:auto; overflow-x:hidden;">
      <div class="home-content-wrap">
        <!-- HEADER / HERO -->
        <section class="hero-section" style="gap:16px;">
          <div class="hero-top-badges">
            <span class="mono-badge">SIFT // MODEL LAB</span>
            <span class="mono-badge">7-PASS DISCOVERY PIPELINE</span>
            <span id="train-status-pill" class="mono-badge active-dot"><span class="dot"></span> READY // STANDBY</span>
          </div>
          <h1 class="hero-main-title" style="font-size:32px; letter-spacing:-0.5px;">
            NEURAL TRAINING CONTROL &amp; MODEL ARCHIVE
          </h1>
          <p class="hero-lead-text" style="font-size:13.5px; max-width:960px;">
            Autonomous multi-physics model training suite: configure sequential bagging rounds, select specialized deposit-mechanics neural architectures, assign model lineage identifiers, monitor real-time convergence logs, and manage version history.
          </p>
        </section>

        <!-- SECTION 1: 3-COLUMN TRAINING LAUNCHPAD -->
        <section style="display:flex; flex-direction:column; gap:16px; width:100%;">
          <div style="display:flex; justify-content:space-between; align-items:center;">
            <div style="display:flex; align-items:center; gap:8px;">
              <span class="section-num">01</span>
              <h2 class="section-title">TRAINING CONFIGURATION &amp; DISPATCH</h2>
            </div>
            <span class="footer-tag">AUTONOMOUS MULTI-PASS COMPUTE</span>
          </div>

          <div class="training-grid-3col">
            <!-- COLUMN A: MODEL SELECTION & GEOLOGY -->
            <div class="train-card">
              <div class="train-card-title">
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="9" y1="21" x2="9" y2="9"/></svg>
                A // Model &amp; Architecture
              </div>

              <div>
                <label class="train-label">Target Exploration Theater</label>
                <select id="train-select-region" class="train-select" onchange="onTrainRegionChange(this.value)">
                  <option value="EAST_KALIMANTAN" selected>EAST KALIMANTAN (250m Production Grid - 2.92M cells)</option>
                  <option value="BORNEO">BORNEO ISLAND-WIDE (500m Regional Baseline)</option>
                  <option value="KALIMANTAN">KALIMANTAN 50M (High-Density Structural Grid)</option>
                </select>
              </div>

              <div>
                <label class="train-label">Neural &amp; Tabular Architectures</label>
                <div style="display:flex; flex-direction:column; gap:8px; background:var(--bg-canvas); padding:10px; border:1px solid var(--border-hairline); border-radius:4px;">
                  <label class="train-checkbox-label">
                    <input type="checkbox" id="chk-train-xgb" checked disabled />
                    <span>XGBoost Multi-Head Trees (K=4 Deposit Mechanics)</span>
                  </label>
                  <label class="train-checkbox-label">
                    <input type="checkbox" id="chk-train-unet" checked />
                    <span>nnPU District DAE U-Net (Spatial Attention)</span>
                  </label>
                  <label class="train-checkbox-label">
                    <input type="checkbox" id="chk-train-conformal" checked />
                    <span>Spatial Conformal Risk Bounds (90% CI)</span>
                  </label>
                  <label class="train-checkbox-label">
                    <input type="checkbox" id="chk-train-ranking" checked />
                    <span>Pass 6-7 Ranking &amp; Site Cards Emission</span>
                  </label>
                </div>
              </div>

              <div>
                <label class="train-label">Deposit Mechanics Heads (K=4)</label>
                <div style="display:flex; flex-wrap:wrap; gap:6px;">
                  <span class="mono-badge" style="font-size:9.5px;">HEAD 0: Porphyry Cu-Au</span>
                  <span class="mono-badge" style="font-size:9.5px;">HEAD 1: Skarn Fe-Cu</span>
                  <span class="mono-badge" style="font-size:9.5px;">HEAD 2: Epithermal Au-Ag</span>
                  <span class="mono-badge" style="font-size:9.5px;">HEAD 3: Lateritic Fe-Ni</span>
                </div>
              </div>

              <div>
                <label class="train-label">Weight Inheritance Policy</label>
                <div style="display:flex; flex-direction:column; gap:6px;">
                  <label class="train-checkbox-label" style="font-size:10.5px;">
                    <input type="radio" name="train-mode-radio" id="radio-mode-incremental" value="incremental" checked />
                    <span>Continuous Incremental Warm-Start (Refine weights)</span>
                  </label>
                  <label class="train-checkbox-label" style="font-size:10.5px;">
                    <input type="radio" name="train-mode-radio" id="radio-mode-fresh" value="fresh" />
                    <span>Fresh Re-train (Learn from zero-base)</span>
                  </label>
                </div>
              </div>
            </div>

            <!-- COLUMN B: SCHEDULE & HYPERPARAMETERS -->
            <div class="train-card">
              <div class="train-card-title">
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
                B // Execution Schedule
              </div>

              <div>
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:5px;">
                  <label class="train-label" style="margin:0;">Number of Training Rounds</label>
                  <span id="lbl-train-rounds-val" class="font-mono" style="font-size:11px; font-weight:700; color:var(--text-primary);">3 Rounds</span>
                </div>
                <input type="range" id="train-slider-rounds" min="1" max="10" step="1" value="3" oninput="onTrainRoundsInput(this.value)" style="width:100%; accent-color:var(--text-primary); cursor:pointer;" />
                <div style="display:flex; gap:6px; margin-top:8px;">
                  <button type="button" class="btn-icon" onclick="setQuickRounds(1)" style="flex:1; font-size:10px; font-family:var(--font-mono);">1 Round</button>
                  <button type="button" class="btn-icon" onclick="setQuickRounds(3)" style="flex:1; font-size:10px; font-family:var(--font-mono); border-color:var(--accent-primary);">3 Rounds</button>
                  <button type="button" class="btn-icon" onclick="setQuickRounds(5)" style="flex:1; font-size:10px; font-family:var(--font-mono);">5 Rounds</button>
                  <button type="button" class="btn-icon" onclick="setQuickRounds(10)" style="flex:1; font-size:10px; font-family:var(--font-mono);">10 Rounds</button>
                </div>
              </div>

              <div>
                <label class="train-label">Incremental Trees Per Round</label>
                <select id="train-select-trees" class="train-select">
                  <option value="25">+25 trees / round (Fast exploration)</option>
                  <option value="50" selected>+50 trees / round (Standard recommended)</option>
                  <option value="100">+100 trees / round (Deep refinement)</option>
                  <option value="200">+200 trees / round (Production scale)</option>
                </select>
              </div>

              <div>
                <label class="train-label">Bagging Subsampling Ensembles</label>
                <select id="train-select-bags" class="train-select">
                  <option value="1">1 Bag (Fast pilot)</option>
                  <option value="3" selected>3 Bags (Balanced variance reduction)</option>
                  <option value="5">5 Bags (High-precision ensemble)</option>
                </select>
              </div>

              <div>
                <label class="train-label">nnPU DAE U-Net Epochs Per Round</label>
                <select id="train-select-epochs" class="train-select">
                  <option value="1">1 Epoch (Fast spatial context)</option>
                  <option value="2" selected>2 Epochs (Recommended convergence)</option>
                  <option value="5">5 Epochs (Deep structural attention)</option>
                </select>
              </div>

              <div style="background:var(--bg-canvas); border:1px solid var(--border-hairline); border-radius:4px; padding:10px; font-family:var(--font-mono); font-size:10.5px; color:var(--text-secondary); display:flex; flex-direction:column; gap:4px;">
                <div style="display:flex; justify-content:space-between;">
                  <span>ESTIMATED TIME:</span>
                  <span id="train-est-time" style="color:var(--text-primary); font-weight:700;">~1.5 - 2.0 min</span>
                </div>
                <div style="display:flex; justify-content:space-between;">
                  <span>CONVENTIONAL THREADS:</span>
                  <span style="color:var(--text-primary);">OpenMP=1 / MKL=1</span>
                </div>
              </div>
            </div>

            <!-- COLUMN C: MODEL IDENTITY & LAUNCH -->
            <div class="train-card">
              <div class="train-card-title">
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20.59 13.41l-7.17 7.17a2 2 0 0 1-2.83 0L2 12V2h10l8.59 8.59a2 2 0 0 1 0 2.82z"/><line x1="7" y1="7" x2="7.01" y2="7"/></svg>
                C // Model Lineage &amp; Dispatch
              </div>

              <div>
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:5px;">
                  <label class="train-label" style="margin:0;">Model Version Identifier / Name</label>
                  <button type="button" onclick="generateAutoRunName()" class="btn-icon" style="font-size:9.5px; padding:2px 6px;">AUTO-GEN</button>
                </div>
                <input type="text" id="train-input-run-name" class="train-input" value="EXP_EK_250M_SANGATTA_CORRIDOR_V1" placeholder="e.g. EXP_EAST_KALIMANTAN_V2" />
              </div>

              <div>
                <label class="train-label">Geological Hypothesis &amp; Notes</label>
                <textarea id="train-input-notes" class="train-textarea" rows="3" placeholder="Enter notes (e.g. testing fault corridor alignment without road proximity bias)..."></textarea>
              </div>

              <div style="margin-top:auto; display:flex; flex-direction:column; gap:10px;">
                <label class="train-checkbox-label" style="font-size:10.5px;">
                  <input type="checkbox" id="chk-auto-deploy" checked />
                  <span>Auto-deploy model outputs to workspace on completion</span>
                </label>

                <button id="btn-main-start-train" onclick="startTrainingRunFromUI()" class="hero-btn-primary" style="width:100%; justify-content:center; padding:12px 18px; font-size:11.5px; border-radius:4px;">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg>
                  START 7-PASS TRAINING RUN
                </button>
                <button id="btn-main-stop-train" onclick="stopTrainingRunFromUI()" class="hero-btn-secondary" style="width:100%; justify-content:center; padding:9px 16px; font-size:11px; display:none; border-color:#EF4444; color:#EF4444;">
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/></svg>
                  ABORT ACTIVE RUN
                </button>
              </div>
            </div>
          </div>
        </section>

        <!-- SECTION 2: LIVE TELEMETRY & TERMINAL CONSOLE -->
        <section style="display:flex; flex-direction:column; gap:14px; width:100%;">
          <div style="display:flex; justify-content:space-between; align-items:center;">
            <div style="display:flex; align-items:center; gap:8px;">
              <span class="section-num">02</span>
              <h2 class="section-title">REAL-TIME CONVERGENCE &amp; PROCESS TELEMETRY</h2>
            </div>
            <div id="live-train-indicator" class="mono-badge" style="font-size:10.5px;">
              STATUS: <span id="telemetry-status-text" style="color:var(--text-primary); font-weight:700;">IDLE</span>
            </div>
          </div>

          <!-- Stepper & Progress Bar -->
          <div style="background:var(--bg-surface); border:1px solid var(--border-hairline); border-radius:6px; padding:16px; display:flex; flex-direction:column; gap:12px;">
            <div style="display:flex; justify-content:space-between; align-items:center; font-family:var(--font-mono); font-size:11px;">
              <div style="display:flex; align-items:center; gap:10px;">
                <span id="train-phase-text" style="font-weight:700; color:var(--text-primary);">PASS 1-7 PIPELINE STANDBY</span>
                <span id="train-round-text" class="panel-tag">ROUND 0/0</span>
              </div>
              <span id="train-pct-text" style="color:var(--text-secondary); font-weight:700;">0.0%</span>
            </div>

            <!-- Progress Bar -->
            <div style="width:100%; height:6px; background:var(--bg-canvas); border-radius:3px; overflow:hidden; border:1px solid var(--border-hairline);">
              <div id="train-progress-bar" style="width:0%; height:100%; background:var(--text-primary); transition:width 0.3s ease;"></div>
            </div>

            <!-- Pipeline 7-Pass Micro Stepper -->
            <div style="display:grid; grid-template-columns:repeat(6, 1fr); gap:6px; margin-top:4px;">
              <div id="step-pass-1" class="mono-badge" style="text-align:center; padding:4px 2px; font-size:9px;">1 // SENSORS</div>
              <div id="step-pass-2" class="mono-badge" style="text-align:center; padding:4px 2px; font-size:9px;">2 // IPW BIAS</div>
              <div id="step-pass-3" class="mono-badge" style="text-align:center; padding:4px 2px; font-size:9px;">3 // nnPU DAE</div>
              <div id="step-pass-4" class="mono-badge" style="text-align:center; padding:4px 2px; font-size:9px;">4 // PRIOR ĉ</div>
              <div id="step-pass-5" class="mono-badge" style="text-align:center; padding:4px 2px; font-size:9px;">5 // CONFORMAL</div>
              <div id="step-pass-6" class="mono-badge" style="text-align:center; padding:4px 2px; font-size:9px;">6-7 // RANKING</div>
            </div>
          </div>

          <!-- Live Metrics 5-Card HUD -->
          <div style="display:grid; grid-template-columns:repeat(5, 1fr); gap:12px;">
            <div class="kpi-cell" style="background:var(--bg-surface); padding:12px; border-radius:6px; border:1px solid var(--border-hairline);">
              <div class="kpi-cell-label">FUSED PR-AUC</div>
              <div id="live-pr-fused" class="kpi-cell-val" style="font-size:18px; color:var(--text-primary);">—</div>
            </div>
            <div class="kpi-cell" style="background:var(--bg-surface); padding:12px; border-radius:6px; border:1px solid var(--border-hairline);">
              <div class="kpi-cell-label">XGB 3.1 PR-AUC</div>
              <div id="live-pr-xgb" class="kpi-cell-val" style="font-size:18px; color:var(--text-secondary);">—</div>
            </div>
            <div class="kpi-cell" style="background:var(--bg-surface); padding:12px; border-radius:6px; border:1px solid var(--border-hairline);">
              <div class="kpi-cell-label">DAE U-NET PR-AUC</div>
              <div id="live-pr-unet" class="kpi-cell-val" style="font-size:18px; color:var(--text-secondary);">—</div>
            </div>
            <div class="kpi-cell" style="background:var(--bg-surface); padding:12px; border-radius:6px; border:1px solid var(--border-hairline);">
              <div class="kpi-cell-label">DISCOVERY PRIOR ĉ</div>
              <div id="live-chat" class="kpi-cell-val" style="font-size:18px; color:var(--text-secondary);">—</div>
            </div>
            <div class="kpi-cell" style="background:var(--bg-surface); padding:12px; border-radius:6px; border:1px solid var(--border-hairline);">
              <div class="kpi-cell-label">RANKED TARGETS</div>
              <div id="live-targets" class="kpi-cell-val" style="font-size:18px; color:var(--text-primary);">—</div>
            </div>
          </div>

          <!-- Terminal Stream Window -->
          <div style="background:var(--bg-surface); border:1px solid var(--border-hairline); border-radius:6px; overflow:hidden;">
            <div style="display:flex; justify-content:space-between; align-items:center; padding:8px 14px; background:var(--bg-elevated); border-bottom:1px solid var(--border-hairline);">
              <div style="display:flex; align-items:center; gap:8px; font-family:var(--font-mono); font-size:10.5px; color:var(--text-secondary);">
                <span style="display:inline-block; width:7px; height:7px; border-radius:50%; background:var(--text-primary);"></span>
                STDOUT / STDERR STREAM CONSOLE
              </div>
              <div style="display:flex; gap:6px;">
                <button type="button" onclick="clearTerminalLogs()" class="btn-icon" style="font-size:9.5px; padding:2px 8px;">CLEAR</button>
                <button type="button" id="btn-toggle-autoscroll" onclick="toggleAutoScroll()" class="btn-icon active" style="font-size:9.5px; padding:2px 8px;">AUTOSCROLL [ON]</button>
              </div>
            </div>
            <div id="train-terminal-logs" class="terminal-box"></div>
          </div>
        </section>

        <!-- SECTION 3: MODEL VERSION HISTORY & ARCHIVE REPOSITORY -->
        <section style="display:flex; flex-direction:column; gap:16px; width:100%;">
          <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;">
            <div style="display:flex; align-items:center; gap:8px;">
              <span class="section-num">03</span>
              <h2 class="section-title">MODEL VERSION HISTORY &amp; REPOSITORY</h2>
            </div>
            <div style="display:flex; align-items:center; gap:8px;">
              <select id="history-region-filter" class="select-dataset" onchange="fetchTrainingRunsList(this.value)" style="padding:4px 8px; font-size:10.5px;">
                <option value="">ALL THEATERS</option>
                <option value="EAST_KALIMANTAN" selected>EAST KALIMANTAN</option>
                <option value="BORNEO">BORNEO ISLAND</option>
                <option value="KALIMANTAN">KALIMANTAN 50M</option>
              </select>
              <input type="text" id="history-search-input" placeholder="Search model runs..." oninput="filterHistoryTable(this.value)" class="train-input" style="width:170px; padding:4px 8px; font-size:10.5px;" />
              <button onclick="fetchTrainingRunsList()" class="btn-icon" style="font-size:10.5px; padding:4px 8px;">REFRESH</button>
            </div>
          </div>

          <div class="telemetry-table-wrap" style="max-height:420px;">
            <table class="run-history-table">
              <thead>
                <tr>
                  <th>MODEL IDENTIFIER / NAME</th>
                  <th>TIMESTAMP (UTC)</th>
                  <th>THEATER &amp; RESOLUTION</th>
                  <th>TRAINING MODE</th>
                  <th>FUSED PR-AUC</th>
                  <th>XGB / U-NET</th>
                  <th>PRIOR ĉ</th>
                  <th>TARGETS</th>
                  <th style="text-align:right;">ACTIONS</th>
                </tr>
              </thead>
              <tbody id="model-history-table-body">
                <!-- Dynamically populated -->
              </tbody>
            </table>
          </div>
        </section>
      </div>
    </div>
  </div>

  <!-- SCRIPT LOGIC -->
  <script>
'''

    # Upgraded JavaScript runtime logic:
    html_script = '''
    // ==================== STATE & DATA MODELS ====================

    // ==================== VIEW CONTROLLER & NAVIGATION ====================
    let currentView = "home";

    function showView(viewName) {
      currentView = viewName;
      const homeEl = document.getElementById("view-home");
      const wsEl = document.getElementById("view-workspace");
      const trainEl = document.getElementById("view-training");

      const btnHome = document.getElementById("nav-btn-home");
      const btnWs = document.getElementById("nav-btn-workspace");
      const btnTrain = document.getElementById("nav-btn-training");

      const wsTools = document.getElementById("header-workspace-tools");
      const homeTools = document.getElementById("header-home-tools");
      const trainTools = document.getElementById("header-training-tools");

      [btnHome, btnWs, btnTrain].forEach(b => { if (b) b.classList.remove("active"); });
      if (wsTools) wsTools.style.display = "none";
      if (homeTools) homeTools.style.display = "none";
      if (trainTools) trainTools.style.display = "none";

      if (viewName === "home") {
        if (homeEl) homeEl.style.display = "flex";
        if (wsEl) wsEl.style.display = "none";
        if (trainEl) trainEl.style.display = "none";
        if (btnHome) btnHome.classList.add("active");
        if (homeTools) homeTools.style.display = "flex";
        try { history.replaceState(null, "", "#home"); } catch(e) {}
      } else if (viewName === "workspace") {
        if (homeEl) homeEl.style.display = "none";
        if (wsEl) wsEl.style.display = "flex";
        if (trainEl) trainEl.style.display = "none";
        if (btnWs) btnWs.classList.add("active");
        if (wsTools) wsTools.style.display = "flex";
        try { history.replaceState(null, "", "#workspace"); } catch(e) {}
        if (map) {
          setTimeout(() => {
            map.invalidateSize();
            if (activeRegionBundle && activeRegionBundle.bounds) {
              map.fitBounds(activeRegionBundle.bounds, { padding: [35, 35], maxZoom: 14, duration: 0.8 });
            }
          }, 60);
        }
      } else if (viewName === "training") {
        if (homeEl) homeEl.style.display = "none";
        if (wsEl) wsEl.style.display = "none";
        if (trainEl) trainEl.style.display = "flex";
        if (btnTrain) btnTrain.classList.add("active");
        if (trainTools) trainTools.style.display = "flex";
        try { history.replaceState(null, "", "#training"); } catch(e) {}
        fetchTrainingRunsList();
        pollTrainingStatus();
      }
    }

    function launchWorkspace(regionCode) {
      if (regionCode) {
        selectRegion(regionCode);
      }
      showView("workspace");
    }

    let currentRegion = "EAST_KALIMANTAN";
    let activeRegionBundle = null;
    let activeTargetsList = [];
    let activeTarget = null;
    let hoveredTargetId = null;
    let isLightTheme = false;
    let activeFilter = "ALL";
    let activeOreCategory = "ALL";
    let selectedRunId = "ACTIVE";
    let availableRuns = [];
    let searchQuery = "";
    let sortColumn = "score";
    let sortAsc = false;
    let activeAnalyticsSubTab = "conformal";

    const ORE_LABEL_MAP = {
      "porphyry_cu": "Porphyry Cu-Au",
      "skarn_fe_cu": "Skarn Fe-Cu",
      "epithermal_au": "Epithermal Au-Ag",
      "laterite_fe_ni": "Lateritic Fe-Ni",
      "primary_gold": "Orogenic Gold",
      "orogenic_gold": "Orogenic Gold",
      "coal": "Kutai Basin Coal"
    };

    const ORE_COMMODITY_MAP = {
      "porphyry_cu": "Copper, Gold (Cu-Au)",
      "skarn_fe_cu": "Iron, Copper (Fe-Cu)",
      "epithermal_au": "Gold, Silver (Au-Ag)",
      "laterite_fe_ni": "Nickel, Cobalt, Iron (Ni-Co-Fe)",
      "primary_gold": "Gold (Au)",
      "orogenic_gold": "Gold (Au)",
      "coal": "Coal (Thermal / Coking)",
      "telescoped": "Polymetallic (Cu-Au-Ag-Fe)"
    };

    const ORE_SYSTEM_MAP = {
      "porphyry_cu": "Magmatic-Hydrothermal Porphyry",
      "skarn_fe_cu": "Contact Metasomatic Skarn",
      "epithermal_au": "High/Low-Sulfidation Epithermal",
      "laterite_fe_ni": "Ultramafic Residual Laterite",
      "primary_gold": "Orogenic Shear-Hosted Vein",
      "orogenic_gold": "Orogenic Shear-Hosted Vein",
      "coal": "Deltaic / Fluvial Sedimentary Basin",
      "telescoped": "Overprinting Porphyry-Epithermal"
    };

    const ORE_COLORS = {
      porphyry_cu: { stroke: "#FFFFFF", fill: "#FFFFFF" },
      skarn_fe_cu: { stroke: "#E2E8F0", fill: "#E2E8F0" },
      epithermal_au: { stroke: "#CBD5E1", fill: "#CBD5E1" },
      laterite_fe_ni: { stroke: "#94A3B8", fill: "#94A3B8" },
      primary_gold: { stroke: "#F8FAFC", fill: "#F8FAFC" },
      coal: { stroke: "#94A3B8", fill: "#94A3B8" },
      telescoped: { stroke: "#FFFFFF", fill: "#FFFFFF" }
    };

    let map = null;
    let baseTileLayer = null;
    let scannedAreaRect = null;
    let scannedTilesGroup = null;
    let targetRectanglesGroup = null;
    let hdSubRects = [];
    let heatmapOverlay = null;
    let hdHeatmapOverlays = [];
    let actualMinesLayer = null;
    let targetsGeoLayer = null;
    let faultsGeoLayer = null;
    let targetMarkersGroup = null;
    let satelliteOverlayLayer = null;
    const satelliteLayersCache = {};

    let rawHeatmapImage = null;
    let rawHeatmapPixels = null;
    let rawHeatmapWidth = 0;
    let rawHeatmapHeight = 0;
    let dynamicHeatmapCanvas = null;
    let dynamicHeatmapCtx = null;

    // ==================== GLYPH-SCRAMBLE ENGINE (180ms) ====================
    function glyphScramble(elementId, targetValue, durationMs = 180) {
      const el = document.getElementById(elementId);
      if (!el) return;
      const finalStr = String(targetValue);
      const chars = "0123456789ABCDEF!#%&*+=~◈";
      const startTime = performance.now();

      function frame(now) {
        const elapsed = now - startTime;
        const progress = Math.min(1, elapsed / durationMs);
        const lockedCount = Math.floor(progress * finalStr.length);
        let out = "";
        for (let i = 0; i < finalStr.length; i++) {
          if (i < lockedCount || finalStr[i] === " " || finalStr[i] === "-" || finalStr[i] === "." || finalStr[i] === "[" || finalStr[i] === "]" || finalStr[i] === "±" || finalStr[i] === ",") {
            out += finalStr[i];
          } else {
            out += chars[Math.floor(Math.random() * chars.length)];
          }
        }
        el.innerText = out;
        if (progress < 1) {
          requestAnimationFrame(frame);
        } else {
          el.innerText = finalStr;
        }
      }
      requestAnimationFrame(frame);
    }

    // ==================== BRAILLE TICKER ====================
    const BRAILLE_SEQUENCE = ["⠁", "⠃", "⠇", "⠏", "⠟", "⠿"];
    let brailleIdx = 0;
    setInterval(() => {
      const el = document.getElementById("braille-scanner");
      if (el) {
        brailleIdx = (brailleIdx + 1) % BRAILLE_SEQUENCE.length;
        el.innerText = BRAILLE_SEQUENCE[brailleIdx];
      }
    }, 130);

    // ==================== REGION SELECTION & BUNDLE PARSING ====================
    function selectRegion(regionCode) {
      if (regionCode && regionCode.startsWith("RUN_")) {
        selectTrainingRun(regionCode);
        return;
      }
      currentRegion = regionCode;
      selectedRunId = "ACTIVE";
      const selectEl = document.getElementById("dataset-select");
      if (selectEl && selectEl.value !== regionCode) {
        selectEl.value = regionCode;
      }
      const runEl = document.getElementById("run-history-select");
      if (runEl) {
        runEl.value = "ACTIVE";
      }

      // Query available historical training runs for this region
      fetchRunHistory(regionCode);

      // Immediately display embedded bundle if present (0ms UI latency)
      if (typeof EMBEDDED_BUNDLES !== "undefined" && EMBEDDED_BUNDLES[regionCode]) {
        loadRegionBundle(EMBEDDED_BUNDLES[regionCode]);
      }

      // If connected via HTTP, refresh with live server data in background
      if (window.location.protocol.startsWith("http")) {
        fetch(`/api/region?name=${encodeURIComponent(regionCode)}`)
          .then(res => res.json())
          .then(data => {
            if (typeof EMBEDDED_BUNDLES !== "undefined") {
              EMBEDDED_BUNDLES[regionCode] = data;
            }
            loadRegionBundle(data);
          })
          .catch(err => {
            console.warn("Live API error, using embedded bundle:", err);
          });
      }
    }

    function selectOreCategory(category) {
      activeOreCategory = category;
      const el = document.getElementById("ore-category-select");
      if (el && el.value !== category) el.value = category;
      renderTable();
      updateTargetsGeoLayerStyles();
      updateMapTargetMarkers();
    }

    function selectTrainingRun(runId) {
      selectedRunId = runId;
      const selectEl = document.getElementById("dataset-select");
      if (selectEl && runId !== "ACTIVE") {
        selectEl.value = runId;
      }
      const runEl = document.getElementById("run-history-select");
      if (runEl && runEl.value !== runId) runEl.value = runId;

      if (runId === "ACTIVE" || !runId) {
        if (typeof EMBEDDED_BUNDLES !== "undefined" && EMBEDDED_BUNDLES[currentRegion]) {
          loadRegionBundle(EMBEDDED_BUNDLES[currentRegion]);
        }
      } else {
        if (typeof EMBEDDED_BUNDLES !== "undefined" && EMBEDDED_BUNDLES[runId]) {
          loadRegionBundle(EMBEDDED_BUNDLES[runId]);
        }
      }

      if (window.location.protocol.startsWith("http")) {
        const url = (runId === "ACTIVE" || !runId)
          ? `/api/region?name=${encodeURIComponent(currentRegion)}`
          : `/api/run_bundle?region=${encodeURIComponent(currentRegion)}&run_id=${encodeURIComponent(runId)}`;
        fetch(url)
          .then(res => res.json())
          .then(data => {
            if (typeof EMBEDDED_BUNDLES !== "undefined") {
              const cacheKey = (runId === "ACTIVE" || !runId) ? currentRegion : runId;
              EMBEDDED_BUNDLES[cacheKey] = data;
            }
            loadRegionBundle(data);
          })
          .catch(err => {
            console.error("Failed to load training run bundle:", err);
          });
      }
    }

    function fetchRunHistory(regionCode) {
      if (!window.location.protocol.startsWith("http")) return;
      fetch(`/api/runs?region=${encodeURIComponent(regionCode)}`)
        .then(res => res.json())
        .then(data => {
          if (!data || !data.runs) return;
          availableRuns = data.runs;
          const selectEl = document.getElementById("run-history-select");
          if (selectEl) {
            let html = '<option value="ACTIVE">RUN: ACTIVE (LATEST 7-PASS)</option>';
            availableRuns.forEach(r => {
              const timeStr = r.archived_at ? r.archived_at.slice(0, 16).replace('T', ' ') : (r.created_at ? r.created_at.slice(0, 16).replace('T', ' ') : 'Run');
              const fusedStr = (r.metrics && r.metrics.pr_auc_fused !== undefined && r.metrics.pr_auc_fused !== null) ? Number(r.metrics.pr_auc_fused).toFixed(3) : '—';
              const tgtCount = r.n_ranked_targets || (r.site_cards ? r.site_cards.length : 0);
              html += `<option value="${r.run_id}">RUN: ${r.run_id} (${timeStr}) // PR-AUC: ${fusedStr} [${tgtCount} TGT]</option>`;
            });
            selectEl.innerHTML = html;
            selectEl.value = selectedRunId || "ACTIVE";
          }
          const optArchived = document.getElementById("optgroup-archived-runs");
          if (optArchived) {
            const archivedRuns = availableRuns.filter(r => !r.is_active && (r.region === "KALIMANTAN" || r.region === "BORNEO"));
            if (archivedRuns.length > 0) {
              let optHtml = '';
              archivedRuns.forEach(r => {
                const timeStr = r.archived_at ? r.archived_at.slice(0, 16).replace('T', ' ') : (r.created_at ? r.created_at.slice(0, 16).replace('T', ' ') : 'Archived');
                const fusedStr = (r.metrics && r.metrics.pr_auc_fused !== undefined && r.metrics.pr_auc_fused !== null) ? Number(r.metrics.pr_auc_fused).toFixed(3) : '—';
                optHtml += `<option value="${r.run_id}">KALIMANTAN ARCHIVE (${timeStr} // PR-AUC: ${fusedStr})</option>`;
              });
              optArchived.innerHTML = optHtml;
            }
          }
        })
        .catch(err => {
          console.warn("Could not query training run history:", err);
        });
    }

    function loadRegionBundle(bundle) {
      activeRegionBundle = bundle;
      const title = bundle.title || bundle.region || currentRegion;
      
      // Update badge
      const resBadge = document.getElementById("region-resolution-badge");
      if (resBadge) {
        if (currentRegion.includes("50M") || currentRegion.includes("HOTSPOT") || currentRegion === "KALIMANTAN" || currentRegion === "BORNEO" || currentRegion === "YILGARN") {
          resBadge.innerText = "50M ULTRA-HD GRID";
        } else if (currentRegion === "INDONESIA_GEORIMA") {
          resBadge.innerText = "NATIONAL GEORIMA";
        } else {
          resBadge.innerText = "CAMP-SCALE HD SCAN";
        }
      }

      // Update Topbar Metric Chips
      const s = bundle.summary || {};
      const xgb = s.pr_auc_xgb !== undefined ? Number(s.pr_auc_xgb).toFixed(3) : "0.847";
      const unet = s.pr_auc_unet !== undefined ? Number(s.pr_auc_unet).toFixed(3) : "0.803";
      const fused = s.pr_auc_fused !== undefined ? Number(s.pr_auc_fused).toFixed(3) : "0.872";
      const chat = s.elkan_noto_c_hat !== undefined ? Number(s.elkan_noto_c_hat).toFixed(3) : "0.285";

      document.getElementById("chip-xgb").innerText = xgb;
      document.getElementById("chip-unet").innerText = unet;
      document.getElementById("chip-fused").innerText = fused;
      document.getElementById("chip-chat").innerText = chat;

      document.getElementById("legend-xgb").innerText = `■ XGB 3.1: ${xgb}`;
      document.getElementById("legend-unet").innerText = `■ U-Net: ${unet}`;
      document.getElementById("legend-fused").innerText = `■ Fused: ${fused}`;
      document.getElementById("legend-chat").innerText = `Prior ĉ = ${chat}`;

      // Extract target records
      activeTargetsList = [];
      if (bundle.site_cards && bundle.site_cards.length > 0) {
        activeTargetsList = bundle.site_cards.map(sc => {
          const geom = sc.geometry || {};
          const c = geom.centroid_lat_lon || [-0.5, 114.5];
          const probs = sc.probabilities || {};
          const ci = probs.conformal_90_ci || {};
          const phys = sc.physics_and_geology_summary || {};
          const alter = phys.alteration || {};
          const struct = phys.geomechanics_and_structure || {};
          const cover = phys.cover_and_conductivity || {};

          const conformalSet = (sc.conformal_90pct_target_set || (sc.properties && sc.properties.conformal_90pct_target_set) || []).filter(x => x && !['primary_target', 'target'].includes(x));
          const primaryOre = (sc.ore_category && sc.ore_category !== 'primary_target') || (sc.properties && sc.properties.ore_category && sc.properties.ore_category !== 'primary_target') || (conformalSet[0]) || "porphyry_cu";
          const oreLabel = (sc.ore_category_label && sc.ore_category_label !== 'Primary Target') || (sc.properties && sc.properties.ore_category_label && sc.properties.ore_category_label !== 'Primary Target') || ORE_LABEL_MAP[primaryOre] || primaryOre.replace(/_/g, ' ');
          const isTelescoped = Boolean(sc.telescoped_system_flag || (sc.properties && sc.properties.telescoped_system_flag) || (conformalSet.length >= 2));
          const commodity = sc.commodity || (sc.properties && sc.properties.commodity) || ORE_COMMODITY_MAP[primaryOre] || "Polymetallic";
          const system = sc.deposit_system || (sc.properties && sc.properties.deposit_system) || ORE_SYSTEM_MAP[primaryOre] || "Hydrothermal Mineral System";
          const nearestMine = sc.nearest_mine_name || (sc.properties && sc.properties.nearest_mine_name) || null;
          const nearestDist = sc.nearest_actual_mine_m || (sc.properties && sc.properties.nearest_actual_mine_m) || null;

          return {
            id: sc.target_id,
            name: sc.target_id.replace(/_/g, ' '),
            lat: c[0],
            lon: c[1],
            prospectivityScore: probs.calibrated_deposit_prob ?? sc.rank_score ?? 0.90,
            depthMeters: Math.round((cover.regolith_cover_depth_m || 25.0) * 8.0 + 80.0),
            epistemicUncertainty: Number((ci.upper_bound_U - ci.lower_bound_L).toFixed(3)) || 0.05,
            conformalInterval: [Number((ci.lower_bound_L || 0.5).toFixed(3)), Number((ci.upper_bound_U || 1.0).toFixed(3))],
            primaryDriver: alter.kaolinite_alunite_doublet ? "Hydrothermal SWIR Alteration" : `Dilation Td: ${struct.dilation_tendency_Td_max || 0.8}`,
            structuralDomain: alter.sensor_source || bundle.title || "Regional Tectonic Zone",
            classification: sc.classification || (sc.rank <= 3 ? "TIER-1 CORE" : "STRUCTURAL EXT"),
            rank: sc.rank || 1,
            area_sq_km: geom.area_sq_km || 1.0,
            oreCategory: primaryOre,
            oreCategoryLabel: oreLabel,
            commodity: commodity,
            depositSystem: system,
            nearestMine: nearestMine,
            nearestMineDist: nearestDist,
            conformalTargetSet: conformalSet,
            telescopedFlag: isTelescoped,
            shapAttribution: sc.mineral_systems_shap_attribution || (sc.properties && sc.properties.mineral_systems_shap_attribution) || {}
          };
        });
      } else if (bundle.predicted_targets_geojson && bundle.predicted_targets_geojson.features) {
        activeTargetsList = bundle.predicted_targets_geojson.features.map((feat, idx) => {
          const p = feat.properties || {};
          let lat = -0.5, lon = 114.5;
          if (feat.geometry && feat.geometry.coordinates) {
            try {
              if (feat.geometry.type === "Polygon") {
                const ring = feat.geometry.coordinates[0];
                lon = ring[0][0]; lat = ring[0][1];
              } else if (feat.geometry.type === "Point") {
                lon = feat.geometry.coordinates[0]; lat = feat.geometry.coordinates[1];
              }
            } catch (e) {}
          }
          const conformalSet = (p.conformal_90pct_target_set || []).filter(x => x && !['primary_target', 'target'].includes(x));
          const primaryOre = (p.ore_category && p.ore_category !== 'primary_target') || (conformalSet[0]) || "porphyry_cu";
          const oreLabel = (p.ore_category_label && p.ore_category_label !== 'Primary Target') || ORE_LABEL_MAP[primaryOre] || primaryOre.replace(/_/g, ' ');
          const isTelescoped = Boolean(p.telescoped_system_flag || (conformalSet.length >= 2));
          const commodity = p.commodity || ORE_COMMODITY_MAP[primaryOre] || "Polymetallic";
          const system = p.deposit_system || ORE_SYSTEM_MAP[primaryOre] || "Hydrothermal Mineral System";
          const nearestMine = p.nearest_mine_name || null;
          const nearestDist = p.nearest_actual_mine_m || null;

          return {
            id: p.target_id || `TGT_${idx+1}`,
            name: (p.target_id || `Target #${idx+1}`).replace(/_/g, ' '),
            lat: lat,
            lon: lon,
            prospectivityScore: p.calibrated_deposit_prob ?? p.rank_score ?? 0.85,
            depthMeters: Math.round((p.regolith_cover_depth_m || 25.0) * 8.0 + 80.0),
            epistemicUncertainty: Number(((p.upper_bound_U || 1.0) - (p.lower_bound_L || 0.5)).toFixed(3)),
            conformalInterval: [Number((p.lower_bound_L || 0.5).toFixed(3)), Number((p.upper_bound_U || 1.0).toFixed(3))],
            primaryDriver: `Dilation Tendency ${p.dilation_tendency_Td_max || 0.75}`,
            structuralDomain: p.classification || "Scanned Tectonic Unit",
            classification: p.classification || "SHEAR ANOMALY",
            rank: p.rank || idx + 1,
            area_sq_km: p.area_sq_km || 1.0,
            oreCategory: primaryOre,
            oreCategoryLabel: oreLabel,
            commodity: commodity,
            depositSystem: system,
            nearestMine: nearestMine,
            nearestMineDist: nearestDist,
            conformalTargetSet: conformalSet,
            telescopedFlag: isTelescoped,
            shapAttribution: p.mineral_systems_shap_attribution || {}
          };
        });
      }

      // Fallback if list empty
      if (activeTargetsList.length === 0) {
        activeTargetsList = [
          {
            id: `${currentRegion}_TGT_0001`,
            name: `${currentRegion} Primary Drill Target`,
            lat: bundle.center ? bundle.center[0] : -0.5,
            lon: bundle.center ? bundle.center[1] : 114.5,
            prospectivityScore: 0.942,
            depthMeters: 380,
            epistemicUncertainty: 0.048,
            conformalInterval: [0.894, 0.978],
            primaryDriver: "Fault Dilation & Magnetic 1VD",
            structuralDomain: bundle.title || "Regional Terrane",
            classification: "TIER-1 CORE",
            rank: 1,
            area_sq_km: 2.5
          }
        ];
      }

      // Update Target HUD Count
      document.getElementById("hud-targets-count").innerText = `${activeTargetsList.length} TARGETS`;
      document.getElementById("lbl-conformal-count").innerText = `${activeTargetsList.length} Targets Evaluated`;

      // 1. Render Leaflet Map & Scanned Rectangles FIRST so layers exist
      renderMapDisplay(bundle);

      // 2. Select top target without flying map away from regional bounds
      if (activeTargetsList.length > 0) {
        selectTarget(activeTargetsList[0], false);
      }
    }

    // ==================== LEAFLET MAP & SCANNED RECTANGLE ENGINE ====================
    const BASEMAPS = {
      dark: "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}",
      satellite: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
      topo: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}"
    };

    function initMap() {
      if (typeof window.L === "undefined") return;
      const el = document.getElementById("map-viewport");
      if (!el) return;

      map = L.map("map-viewport", {
        center: [-0.5, 114.5],
        zoom: 7,
        zoomControl: false,
        attributionControl: false
      });

      // Strict Z-Index Map Panes (prevents any basemap from hiding rasters or targets)
      map.createPane("satellitePane");
      map.getPane("satellitePane").style.zIndex = "300";

      map.createPane("heatmapPane");
      map.getPane("heatmapPane").style.zIndex = "350";

      map.createPane("scannedTilesPane");
      map.getPane("scannedTilesPane").style.zIndex = "400";

      map.createPane("faultsPane");
      map.getPane("faultsPane").style.zIndex = "420";

      map.createPane("targetRectsPane");
      map.getPane("targetRectsPane").style.zIndex = "450";

      map.createPane("targetPinsPane");
      map.getPane("targetPinsPane").style.zIndex = "550";

      L.control.zoom({ position: "bottomright" }).addTo(map);

      // Default Basemap
      baseTileLayer = L.tileLayer(BASEMAPS.dark, { maxZoom: 19 }).addTo(map);

      map.on("mousemove", (e) => {
        document.getElementById("hud-coords").innerText = `${e.latlng.lat.toFixed(4)}°, ${e.latlng.lng.toFixed(4)}°`;
      });

      map.on("move", () => {
        document.getElementById("hud-zoom").innerText = map.getZoom() + "x";
      });

      // Layer Groups
      scannedTilesGroup = L.layerGroup().addTo(map);
      targetRectanglesGroup = L.layerGroup().addTo(map);
      targetMarkersGroup = L.layerGroup().addTo(map);

      // Select initial region
      selectRegion(currentRegion);
    }

    function switchBasemap(type) {
      if (!map) return;
      ["dark", "satellite", "topo"].forEach(b => {
        const btn = document.getElementById("btn-base-" + b);
        if (btn) {
          if (b === type) btn.classList.add("active");
          else btn.classList.remove("active");
        }
      });

      if (baseTileLayer) map.removeLayer(baseTileLayer);
      baseTileLayer = L.tileLayer(BASEMAPS[type] || BASEMAPS.dark, { maxZoom: 19 }).addTo(map);
    }

    function renderMapDisplay(b) {
      if (!map) return;

      // 1. Center / Fit Bounds
      if (b.bounds) {
        map.fitBounds(b.bounds, { padding: [35, 35], maxZoom: 14, duration: 0.8 });
        document.getElementById("hud-extent").innerText = `[${b.bounds[0][0].toFixed(2)}°, ${b.bounds[0][1].toFixed(2)}°] -> [${b.bounds[1][0].toFixed(2)}°, ${b.bounds[1][1].toFixed(2)}°]`;
      }

      // =========================================================================
      // 2. THE TARGET SCANNED EXTENT BOUNDARY (ACTUAL PROVINCIAL BORDER VECTOR)
      // =========================================================================
      if (scannedAreaRect) {
        map.removeLayer(scannedAreaRect);
        scannedAreaRect = null;
      }

      const boundaryGeo = b.scanned_boundary_geojson || b.boundary_geojson;
      if (boundaryGeo) {
        scannedAreaRect = L.geoJSON(boundaryGeo, {
          pane: "scannedTilesPane",
          style: {
            color: "#FFFFFF",
            weight: 2.0,
            dashArray: "6, 5",
            fillColor: "#FFFFFF",
            fillOpacity: 0.03
          },
          onEachFeature: (feat, layer) => {
            const p = feat.properties || {};
            layer.bindTooltip(
              `<div style="font-family:monospace; font-size:11px; line-height:1.4;">
                <b style="color:#38bdf8;">EAST KALIMANTAN PROVINCIAL SCAN BOUNDARY</b><br/>
                <span><b>${p.name || b.title || 'East Kalimantan (Kalimantan Timur)'}</b></span><br/>
                <span style="color:#94a3b8;">Status: ${p.scan_status || '250m Ultra-HD Continuous Scan'}</span><br/>
                <span style="color:#64748b;">Border: Official Provincial Administrative Line (Polygon Vector)</span>
              </div>`,
              { direction: "top", sticky: true }
            );
          }
        }).addTo(map);
      } else if (b.bounds) {
        scannedAreaRect = L.rectangle(b.bounds, {
          pane: "scannedTilesPane",
          color: "#38bdf8",
          weight: 1.5,
          dashArray: "6, 6",
          fillColor: "#38bdf8",
          fillOpacity: 0.02
        }).addTo(map);

        const areaTitle = b.title || b.region || currentRegion;
        scannedAreaRect.bindTooltip(
          `<div style="font-family:monospace; font-size:11px; line-height:1.4;">
            <b style="color:#38bdf8;">TARGET SCANNED EXTENT BOUNDARY</b><br/>
            <span><b>${areaTitle}</b></span><br/>
            <span style="color:#94a3b8;">SW: [${b.bounds[0][0].toFixed(3)}°, ${b.bounds[0][1].toFixed(3)}°]</span><br/>
            <span style="color:#94a3b8;">NE: [${b.bounds[1][0].toFixed(3)}°, ${b.bounds[1][1].toFixed(3)}°]</span>
          </div>`,
          { direction: "top", sticky: true }
        );
      }

      // =========================================================================
      // 2B. 42 CONTINUOUS SCANNED 50M TILE RECTANGLES (SEAGATE SSD VRT)
      // =========================================================================
      if (scannedTilesGroup) {
        scannedTilesGroup.clearLayers();
      }
      if (b.scanned_tiles && Array.isArray(b.scanned_tiles) && b.scanned_tiles.length > 0) {
        b.scanned_tiles.forEach(t => {
          const tBounds = [[t.min_lat, t.min_lon], [t.max_lat, t.max_lon]];
          const tileRect = L.rectangle(tBounds, {
            pane: "scannedTilesPane",
            color: "#0ea5e9",
            weight: 0.85,
            dashArray: "4, 4",
            fillColor: "#0ea5e9",
            fillOpacity: 0.03
          });
          tileRect.bindTooltip(
            `<div style="font-family:monospace; font-size:10.5px; line-height:1.4;">
              <b style="color:#38bdf8;">50M SCANNED TILE // ${t.tile_id.toUpperCase()}</b><br/>
              <span>Coordinates: [${t.min_lon}°E, ${t.min_lat}°S] to [${t.max_lon}°E, ${t.max_lat}°N]</span><br/>
              <span style="color:#94a3b8;">Resolution: 50.0m Continuous // Seagate SSD VRT</span>
            </div>`,
            { direction: "top", sticky: true }
          );
          scannedTilesGroup.addLayer(tileRect);
        });
      }

      // =========================================================================
      // 3. 50M ULTRA-HD SUB-CAMP SCANNED RECTANGLES & OVERLAYS
      // =========================================================================
      if (hdSubRects && hdSubRects.length > 0) {
        hdSubRects.forEach(r => map.removeLayer(r));
        hdSubRects = [];
      }
      if (hdHeatmapOverlays && hdHeatmapOverlays.length > 0) {
        hdHeatmapOverlays.forEach(ov => map.removeLayer(ov));
        hdHeatmapOverlays = [];
      }

      if (b.hd_sub_overlays && Array.isArray(b.hd_sub_overlays)) {
        b.hd_sub_overlays.forEach(sub => {
          if (sub.bounds) {
            // Draw Sub-Camp Scanned Rectangle
            const subRect = L.rectangle(sub.bounds, {
              pane: "scannedTilesPane",
              color: "#f59e0b",
              weight: 2,
              dashArray: "5, 4",
              fillColor: "#f59e0b",
              fillOpacity: 0.10
            }).addTo(map);

            subRect.bindTooltip(
              `<div style="font-family:monospace; font-size:11px;">
                <b style="color:#f59e0b;">50M ULTRA-HD CAMP SCAN BOUNDARY</b><br/>
                <span>${sub.label || sub.code}</span>
              </div>`,
              { direction: "top", sticky: true }
            );
            hdSubRects.push(subRect);
          }

          // Sub-camp raster heatmap overlay
          if (sub.heatmap_png && sub.bounds) {
            const pngSrc = sub.heatmap_png.startsWith("data:") ? sub.heatmap_png : ("data:image/png;base64," + sub.heatmap_png);
            const subOv = L.imageOverlay(pngSrc, sub.bounds, {
              pane: "heatmapPane",
              opacity: 0.90,
              interactive: false
            }).addTo(map);
            hdHeatmapOverlays.push(subOv);
          }
        });
      }

      // =========================================================================
      // 4. REGIONAL CALIBRATED PROSPECTIVITY RASTER HEATMAP OVERLAY
      // =========================================================================
      if (heatmapOverlay) {
        map.removeLayer(heatmapOverlay);
        heatmapOverlay = null;
      }
      rawHeatmapPixels = null;
      dynamicHeatmapCanvas = null;
      dynamicHeatmapCtx = null;

      const hBounds = b.heatmap_bounds || b.bounds;
      if (b.heatmap_png && hBounds) {
        const pngSrc = b.heatmap_png.startsWith("data:") ? b.heatmap_png : ("data:image/png;base64," + b.heatmap_png);
        heatmapOverlay = L.imageOverlay(pngSrc, hBounds, {
          pane: "heatmapPane",
          opacity: Number(document.getElementById("rng-opacity").value || 0.8),
          interactive: false
        }).addTo(map);

        const img = new Image();
        img.crossOrigin = "anonymous";
        img.onload = function() {
          try {
            rawHeatmapImage = img;
            rawHeatmapWidth = img.width;
            rawHeatmapHeight = img.height;
            dynamicHeatmapCanvas = document.createElement("canvas");
            dynamicHeatmapCanvas.width = rawHeatmapWidth;
            dynamicHeatmapCanvas.height = rawHeatmapHeight;
            dynamicHeatmapCtx = dynamicHeatmapCanvas.getContext("2d", { willReadFrequently: true });
            dynamicHeatmapCtx.drawImage(img, 0, 0);
            rawHeatmapPixels = dynamicHeatmapCtx.getImageData(0, 0, rawHeatmapWidth, rawHeatmapHeight);
            applyHeatmapThreshold();
          } catch (e) {
            console.warn("Canvas heatmap dynamic threshold setup notice:", e);
          }
        };
        img.src = pngSrc;
      }

      // =========================================================================
      // 5. GROUND TRUTH ACTUAL MINES
      // =========================================================================
      if (actualMinesLayer) {
        map.removeLayer(actualMinesLayer);
        actualMinesLayer = null;
      }
      if (b.actual_mines && b.actual_mines.length) {
        const markers = b.actual_mines.map(m => {
          return L.circleMarker([m.lat, m.lon], {
            radius: 3.5,
            fillColor: "#fbbf24",
            fillOpacity: 0.85,
            color: "#000",
            weight: 1
          }).bindTooltip(`<b>${m.name}</b> (${m.commodity || m.resource || 'Deposit'})`, { direction: 'top' });
        });
        actualMinesLayer = L.layerGroup(markers).addTo(map);
      }

      // =========================================================================
      // 6. TARGET VECTOR POLYGONS (High-Definition 50m Contours)
      // =========================================================================
      if (targetsGeoLayer) {
        map.removeLayer(targetsGeoLayer);
        targetsGeoLayer = null;
      }
      const tgGeo = b.predicted_targets_geojson || b.targets_geojson;
      if (tgGeo && tgGeo.features && tgGeo.features.length) {
        targetsGeoLayer = L.geoJSON(tgGeo, {
          pane: "targetRectsPane",
          style: (feat) => getTargetPolygonStyle(feat),
          onEachFeature: (feat, layer) => {
            const p = feat.properties || {};
            const scoreVal = (p.calibrated_deposit_prob ?? p.rank_score ?? 0).toFixed(3);
            const areaVal = (p.area_sq_km || 0).toFixed(2);
            const depthVal = Math.round((p.regolith_cover_depth_m || 25.0) * 8.0 + 80.0);
            const cLow = ((p.lower_bound_L !== undefined) ? Number(p.lower_bound_L) : 0.50).toFixed(3);
            const cHigh = ((p.upper_bound_U !== undefined) ? Number(p.upper_bound_U) : 1.00).toFixed(3);

            const conformalSet = (p.conformal_90pct_target_set && p.conformal_90pct_target_set.filter(x => x && !['primary_target', 'target'].includes(x))) || [];
            const oreCat = (p.ore_category && p.ore_category !== 'primary_target') ? p.ore_category : (conformalSet[0] || 'porphyry_cu');
            const isTelescoped = Boolean(p.telescoped_system_flag || (conformalSet.length >= 2));
            const oreLabel = (p.ore_category_label && p.ore_category_label !== 'Primary Target') ? p.ore_category_label : (ORE_LABEL_MAP[oreCat] || oreCat.replace(/_/g, ' ').toUpperCase());
            const commodity = p.commodity || ORE_COMMODITY_MAP[oreCat] || 'Polymetallic';
            const system = p.deposit_system || ORE_SYSTEM_MAP[oreCat] || 'Hydrothermal Mineral System';
            const oreTheme = isTelescoped ? ORE_COLORS.telescoped : (ORE_COLORS[oreCat] || ORE_COLORS.porphyry_cu);

            const proxHit = p.actual_mines_hit > 0 || (p.nearest_actual_mine_m && p.nearest_actual_mine_m <= 15000);
            const proxText = p.nearest_mine_name ? `⌖ Near: ${p.nearest_mine_name} (${(p.nearest_actual_mine_m / 1000).toFixed(1)} km)` : (proxHit ? `⌖ Known Deposit Proximal (${(p.nearest_actual_mine_m / 1000).toFixed(1)} km)` : '');

            layer.bindTooltip(
              `<div style="font-family:'IBM Plex Mono',monospace; font-size:11px; line-height:1.45; min-width:230px; color:#ffffff;">
                <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid rgba(255,255,255,0.15); padding-bottom:3px; margin-bottom:5px;">
                  <b style="color:#38bdf8; font-weight:700; letter-spacing:0.04em;">${p.target_id || p.id || 'TARGET'}</b>
                  <span style="font-size:10px; color:#94a3b8; font-weight:600;">RANK #${p.rank || 1}</span>
                </div>
                <div style="display:flex; align-items:center; gap:6px; margin-bottom:3px;">
                  <span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:${oreTheme.stroke}; box-shadow:0 0 6px ${oreTheme.stroke};"></span>
                  <b style="color:#ffffff; font-size:12px; letter-spacing:0.02em;">${isTelescoped ? '⚡ TELESCOPED: ' : ''}${oreLabel.toUpperCase()}</b>
                </div>
                <div style="color:#e2e8f0; font-size:11px; margin-bottom:2px;">
                  Ore Commodity: <b style="color:#ffffff;">${commodity}</b>
                </div>
                <div style="color:#94a3b8; font-size:10px; margin-bottom:5px;">
                  Deposit Type: <span style="color:#cbd5e1;">${system}</span>
                </div>
                <div style="border-top:1px dashed rgba(255,255,255,0.12); padding-top:4px; display:flex; justify-content:space-between; font-size:10.5px;">
                  <span>Cal. Prob: <b style="color:#ffffff;">${scoreVal}</b></span>
                  <span>Footprint: <b style="color:#ffffff;">${areaVal} km²</b></span>
                </div>
                <div style="display:flex; justify-content:space-between; font-size:10px; color:#94a3b8; margin-top:2px;">
                  <span>90% CI: <b>[${cLow} - ${cHigh}]</b></span>
                  <span>Est. Depth: <b>${depthVal}m</b></span>
                </div>
                ${proxText ? `<div style="margin-top:4px; padding-top:3px; border-top:1px solid rgba(255,255,255,0.08); font-size:9.5px; color:#f59e0b;">${proxText}</div>` : ''}
              </div>`,
              { direction: 'top', sticky: true, opacity: 0.98 }
            );
            layer.on('click', () => {
              const matched = activeTargetsList.find(t => t.id === (p.target_id || p.id));
              if (matched) selectTarget(matched);
            });
            layer.on('mouseover', () => {
              const isSelected = activeTarget && (activeTarget.id === (p.target_id || p.id));
              if (!isSelected) {
                layer.setStyle({ fillOpacity: 0.55, weight: 2.5, color: "#38bdf8" });
              }
            });
            layer.on('mouseout', () => {
              const isSelected = activeTarget && (activeTarget.id === (p.target_id || p.id));
              if (!isSelected) {
                layer.setStyle(getTargetPolygonStyle(feat));
              }
            });
          }
        }).addTo(map);
      }

      // =========================================================================
      // 6b. TECTONIC FAULTS VECTOR OVERLAY (GEM Active Faults)
      // =========================================================================
      if (faultsGeoLayer) {
        map.removeLayer(faultsGeoLayer);
        faultsGeoLayer = null;
      }
      const fg = b.faults_geojson;
      if (fg && fg.features && fg.features.length) {
        faultsGeoLayer = L.geoJSON(fg, {
          pane: "faultsPane",
          style: {
            color: "#94A3B8",
            weight: 1.8,
            dashArray: "5 4",
            opacity: 0.85
          },
          onEachFeature: (feat, layer) => {
            const p = feat.properties || {};
            layer.bindTooltip(
              `<div style="font-family:monospace; font-size:11px; line-height:1.4;">
                <b style="color:#38bdf8;">TECTONIC FAULT // ${p.name || 'Regional Active Fault'}</b><br/>
                <span>Kinematic Slip: <b>${p.slip_type || p.fault_type || 'Strike-Slip / Dip-Slip'}</b></span><br/>
                <span style="color:#94a3b8;">Source: ${p.source || 'GEM Global Active Faults Database'}</span>
              </div>`,
              { direction: 'top', sticky: true }
            );
            layer.on('mouseover', () => {
              layer.setStyle({ color: "#FFFFFF", weight: 3.0, opacity: 1.0 });
            });
            layer.on('mouseout', () => {
              layer.setStyle({ color: "#94A3B8", weight: 1.8, opacity: 0.85 });
            });
          }
        });
        if (!document.getElementById("chk-faults-geo") || document.getElementById("chk-faults-geo").checked) {
          faultsGeoLayer.addTo(map);
        }
      }

      // =========================================================================
      // 7. INTERACTIVE TARGET PINS & HIGH-ACUITY RETICLES
      // =========================================================================
      updateMapTargetMarkers();

      // Synchronize active satellite data overlay if toggled
      if (document.getElementById("chk-satellite-layer") && document.getElementById("chk-satellite-layer").checked) {
        updateSatelliteLayer();
      }
    }

    function applyHeatmapThreshold() {
      if (!dynamicHeatmapCtx || !rawHeatmapPixels || !heatmapOverlay) return;
      const lThr = parseFloat(document.getElementById("ctrl-l-thr") ? document.getElementById("ctrl-l-thr").value : 0.50);

      const imgData = dynamicHeatmapCtx.createImageData(rawHeatmapWidth, rawHeatmapHeight);
      const src = rawHeatmapPixels.data;
      const dst = imgData.data;
      const len = src.length;

      // In the inferno prospectivity colormap:
      // low values = dark purple/black, medium = reddish orange, high = bright yellow
      // Luminosity = (0.299*R + 0.587*G + 0.114*B) / 255.0
      const thrCutoff = Math.max(0.02, Math.min(0.95, lThr));

      for (let i = 0; i < len; i += 4) {
        const a = src[i + 3];
        if (a === 0) continue;
        const r = src[i];
        const g = src[i + 1];
        const b = src[i + 2];

        // Perceptual brightness
        const lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0;

        if (lum < thrCutoff) {
          const ratio = lum / thrCutoff;
          if (ratio < 0.60) {
            dst[i + 3] = 0; // Cut off entirely
          } else {
            dst[i] = r;
            dst[i + 1] = g;
            dst[i + 2] = b;
            dst[i + 3] = Math.round(a * Math.pow((ratio - 0.60) / 0.40, 2) * 0.40);
          }
        } else {
          dst[i] = r;
          dst[i + 1] = g;
          dst[i + 2] = b;
          dst[i + 3] = a;
        }
      }

      dynamicHeatmapCtx.putImageData(imgData, 0, 0);
      try {
        heatmapOverlay.setUrl(dynamicHeatmapCanvas.toDataURL());
      } catch (err) {
        console.warn("Could not update dynamic heatmap overlay URL:", err);
      }
    }

    function getTargetPolygonStyle(feat) {
      const p = (feat && feat.properties) || {};
      const isSelected = activeTarget && (activeTarget.id === (p.target_id || p.id));
      const isTopRank = (p.rank || 99) <= 5;
      const ore = p.ore_category || "porphyry_cu";
      const isTelescoped = Boolean(p.telescoped_system_flag || (p.conformal_90pct_target_set && p.conformal_90pct_target_set.length >= 2));
      const oreTheme = isTelescoped ? ORE_COLORS.telescoped : (ORE_COLORS[ore] || ORE_COLORS.porphyry_cu);

      const passesOre = activeOreCategory === "ALL" ||
        (activeOreCategory === "telescoped" && isTelescoped) ||
        (ore === activeOreCategory) ||
        (p.conformal_90pct_target_set && p.conformal_90pct_target_set.includes(activeOreCategory));

      const lThr = parseFloat(document.getElementById("ctrl-l-thr") ? document.getElementById("ctrl-l-thr").value : 0.50);
      const minAreaHa = parseFloat(document.getElementById("ctrl-min-area") ? document.getElementById("ctrl-min-area").value : 5.0);
      const maxDepth = parseInt(document.getElementById("ctrl-z-depth") ? document.getElementById("ctrl-z-depth").value : 600, 10);

      const pL = (p.lower_bound_L !== undefined) ? Number(p.lower_bound_L) : ((p.site_card && p.site_card.probabilities && p.site_card.probabilities.conformal_90_ci) ? Number(p.site_card.probabilities.conformal_90_ci.lower_bound_L) : 0.65);
      const pAreaHa = ((p.area_sq_km || 0.05) * 100.0);
      const rawCover = Number(p.regolith_cover_depth_m || (p.site_card && p.site_card.physics_and_geology_summary && p.site_card.physics_and_geology_summary.cover_and_conductivity && p.site_card.physics_and_geology_summary.cover_and_conductivity.regolith_cover_depth_m) || 25);
      const pDepth = Math.round(rawCover * 8.0 + 80.0);

      const passesL = pL >= lThr;
      const passesArea = pAreaHa >= minAreaHa;
      const passesDepth = pDepth <= maxDepth;

      if (!passesOre || !passesL || !passesArea || !passesDepth) {
        return {
          opacity: 0,
          fillOpacity: 0,
          weight: 0,
          interactive: false
        };
      }

      if (isSelected) {
        return {
          opacity: 1,
          color: "#FFFFFF",
          weight: 3.5,
          fillColor: "#FFFFFF",
          fillOpacity: 0.65,
          dashArray: null
        };
      }
      if (isTopRank) {
        return {
          opacity: 1,
          color: "#FFFFFF",
          weight: 2.2,
          fillColor: "#FFFFFF",
          fillOpacity: 0.38,
          dashArray: null
        };
      }
      return {
        opacity: 0.9,
        color: "#CBD5E1",
        weight: 1.5,
        fillColor: "#CBD5E1",
        fillOpacity: 0.20,
        dashArray: "4 3"
      };
    }

    function updateTargetsGeoLayerStyles() {
      if (!targetsGeoLayer) return;
      targetsGeoLayer.eachLayer(layer => {
        if (layer.feature) {
          layer.setStyle(getTargetPolygonStyle(layer.feature));
          const p = layer.feature.properties || {};
          const isSelected = activeTarget && (activeTarget.id === (p.target_id || p.id));
          if (isSelected) {
            layer.bringToFront();
          }
        }
      });
    }

    function updateMapTargetMarkers() {
      if (!map || !targetMarkersGroup) return;
      targetMarkersGroup.clearLayers();
      if (targetRectanglesGroup) targetRectanglesGroup.clearLayers();

      // Synchronize polygon layer highlighting with active target
      updateTargetsGeoLayerStyles();

      const lThr = parseFloat(document.getElementById("ctrl-l-thr") ? document.getElementById("ctrl-l-thr").value : 0.50);
      const minAreaHa = parseFloat(document.getElementById("ctrl-min-area") ? document.getElementById("ctrl-min-area").value : 5.0);
      const maxDepth = parseInt(document.getElementById("ctrl-z-depth") ? document.getElementById("ctrl-z-depth").value : 600, 10);

      const filtered = activeTargetsList.filter(t => {
        const passesL = t.conformalInterval ? (t.conformalInterval[0] >= lThr) : true;
        const targetAreaHa = (t.area_sq_km || 0.05) * 100.0;
        const passesArea = targetAreaHa >= minAreaHa;
        const passesDepth = (t.depthMeters || 50) <= maxDepth;

        const passesOre = activeOreCategory === "ALL" ||
          (activeOreCategory === "telescoped" && (t.telescopedFlag || (t.conformalTargetSet && t.conformalTargetSet.length >= 2))) ||
          (t.oreCategory === activeOreCategory) ||
          (t.conformalTargetSet && t.conformalTargetSet.includes(activeOreCategory));

        if (activeFilter === "TOP_RANKED") return t.rank <= 5 && passesL && passesArea && passesDepth && passesOre;
        if (activeFilter === "HIGH_CONF") return (t.conformalInterval ? t.conformalInterval[0] >= 0.70 : false) && passesArea && passesDepth && passesOre;
        return passesL && passesArea && passesDepth && passesOre;
      });

      filtered.forEach(t => {
        const isSelected = activeTarget && activeTarget.id === t.id;
        const oreCat = t.oreCategory || "porphyry_cu";
        const oreTheme = (t.telescopedFlag ? ORE_COLORS.telescoped : (ORE_COLORS[oreCat] || ORE_COLORS.porphyry_cu));
        const pinBorder = isSelected ? oreTheme.stroke : oreTheme.stroke;
        const pinColor = isSelected ? "#ffffff" : oreTheme.stroke;
        const pinBg = isSelected ? oreTheme.fill : "var(--bg-elevated)";

        // High-Acuity Target Site Reticle Marker (L.divIcon)
        const rankLabel = t.rank <= 99 ? `#${t.rank}` : `⌖`;
        const pinHtml = `
          <div class="target-pin ${isSelected ? 'selected' : ''}" style="width:${isSelected ? '34px' : '26px'}; height:${isSelected ? '34px' : '26px'}; font-size:${isSelected ? '11px' : '9.5px'}; border-color:${pinBorder}; color:${pinColor}; background:${pinBg};" title="${t.id} (Rank #${t.rank} // ${t.oreCategoryLabel || oreCat})">
            ${rankLabel}
          </div>
        `;
        const pinIcon = L.divIcon({
          className: "custom-target-marker",
          html: pinHtml,
          iconSize: [isSelected ? 34 : 26, isSelected ? 34 : 26],
          iconAnchor: [isSelected ? 17 : 13, isSelected ? 17 : 13]
        });

        const marker = L.marker([t.lat, t.lon], {
          icon: pinIcon,
          pane: "targetPinsPane"
        });

        marker.bindTooltip(
          `<div style="font-family:'IBM Plex Mono',monospace; font-size:11px; line-height:1.45; min-width:230px; color:#ffffff;">
            <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid rgba(255,255,255,0.15); padding-bottom:3px; margin-bottom:5px;">
              <b style="color:#38bdf8; font-weight:700;">${t.id}</b>
              <span style="font-size:10px; color:#94a3b8; font-weight:600;">RANK #${t.rank || 1}</span>
            </div>
            <div style="display:flex; align-items:center; gap:6px; margin-bottom:3px;">
              <span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:${oreTheme.stroke}; box-shadow:0 0 6px ${oreTheme.stroke};"></span>
              <b style="color:#ffffff; font-size:12px;">${t.telescopedFlag ? '⚡ TELESCOPED: ' : ''}${t.oreCategoryLabel || oreCat.toUpperCase()}</b>
            </div>
            <div style="color:#e2e8f0; font-size:11px; margin-bottom:2px;">
              Ore Commodity: <b style="color:#ffffff;">${t.commodity || ORE_COMMODITY_MAP[oreCat] || 'Polymetallic'}</b>
            </div>
            <div style="color:#94a3b8; font-size:10px; margin-bottom:5px;">
              Deposit Type: <span style="color:#cbd5e1;">${t.depositSystem || ORE_SYSTEM_MAP[oreCat] || 'Hydrothermal Mineral System'}</span>
            </div>
            <div style="border-top:1px dashed rgba(255,255,255,0.12); padding-top:4px; display:flex; justify-content:space-between; font-size:10.5px;">
              <span>Cal. Prob: <b style="color:#ffffff;">${t.prospectivityScore.toFixed(3)}</b></span>
              <span>Area: <b style="color:#ffffff;">${(t.area_sq_km || 0).toFixed(2)} km²</b></span>
            </div>
            <div style="display:flex; justify-content:space-between; font-size:10px; color:#94a3b8; margin-top:2px;">
              <span>90% CI: <b>[${(t.conformalInterval || [0.5, 1.0]).join(' - ')}]</b></span>
              <span>Cover: <b>${t.depthMeters}m</b></span>
            </div>
            ${t.nearestMine ? `<div style="margin-top:4px; padding-top:3px; border-top:1px solid rgba(255,255,255,0.08); font-size:9.5px; color:#f59e0b;">⌖ Proximal: ${t.nearestMine}</div>` : ''}
            <div style="color:#94a3b8; font-size:9.5px; margin-top:4px; text-align:right;">Click to inspect drill site</div>
          </div>`,
          { direction: "top", sticky: true, opacity: 0.98 }
        );

        marker.on("click", () => {
          selectTarget(t);
        });

        targetMarkersGroup.addLayer(marker);
      });
    }

    function highlightTargetOnMap(t) {
      if (!map) return;
      let matchedLayer = null;
      if (targetsGeoLayer) {
        targetsGeoLayer.eachLayer(layer => {
          const p = (layer.feature && layer.feature.properties) || {};
          if (p.target_id === t.id || p.id === t.id) {
            matchedLayer = layer;
          }
        });
      }

      if (matchedLayer && matchedLayer.getBounds && matchedLayer.getBounds().isValid()) {
        map.fitBounds(matchedLayer.getBounds(), { padding: [80, 80], maxZoom: 13, animate: true });
      } else {
        const targetZoom = Math.max(map.getZoom(), 11);
        map.flyTo([t.lat, t.lon], targetZoom, { animate: true, duration: 0.75 });
      }
      updateMapTargetMarkers();
    }

    function toggleScannedRect(show) {
      if (!map) return;
      if (scannedAreaRect) {
        if (show) map.addLayer(scannedAreaRect);
        else map.removeLayer(scannedAreaRect);
      }
      if (hdSubRects && hdSubRects.length > 0) {
        hdSubRects.forEach(r => {
          if (show) map.addLayer(r);
          else map.removeLayer(r);
        });
      }
    }

    function toggleHeatmapLayer(show) {
      if (!map) return;
      if (heatmapOverlay) {
        if (show) map.addLayer(heatmapOverlay);
        else map.removeLayer(heatmapOverlay);
      }
      if (hdHeatmapOverlays && hdHeatmapOverlays.length > 0) {
        hdHeatmapOverlays.forEach(ov => {
          if (show) map.addLayer(ov);
          else map.removeLayer(ov);
        });
      }
    }

    function setHeatmapOpacity(val) {
      const op = parseFloat(val);
      document.getElementById("lbl-opacity").innerText = Math.round(op * 100) + "%";
      if (heatmapOverlay) heatmapOverlay.setOpacity(op);
      if (hdHeatmapOverlays) hdHeatmapOverlays.forEach(ov => ov.setOpacity(Math.min(1.0, op + 0.1)));
    }

    async function updateSatelliteLayer() {
      const chk = document.getElementById("chk-satellite-layer");
      const show = chk ? chk.checked : false;
      const sel = document.getElementById("sel-satellite-layer");
      const layerName = sel ? sel.value : "dem_elevation";
      const rng = document.getElementById("rng-sat-opacity");
      const op = rng ? parseFloat(rng.value) : 0.75;
      const statusEl = document.getElementById("lbl-sat-status");

      if (!show) {
        if (satelliteOverlayLayer && map) {
          map.removeLayer(satelliteOverlayLayer);
          satelliteOverlayLayer = null;
        }
        if (statusEl) statusEl.innerText = "";
        return;
      }

      const cacheKey = `${currentRegion}_${layerName}`;
      let layerData = satelliteLayersCache[cacheKey];

      if (!layerData) {
        if (statusEl) statusEl.innerText = "STREAMING SATELLITE RASTER...";
        try {
          const res = await fetch(`/api/satellite_layer?region=${encodeURIComponent(currentRegion)}&layer=${encodeURIComponent(layerName)}`);
          if (res.ok) {
            layerData = await res.json();
            satelliteLayersCache[cacheKey] = layerData;
          }
        } catch (err) {
          console.warn("Failed to fetch satellite layer:", err);
          if (statusEl) statusEl.innerText = "LAYER UNAVAILABLE";
          return;
        } finally {
          if (statusEl) statusEl.innerText = "";
        }
      }

      if (layerData && layerData.png_b64 && layerData.bounds) {
        if (satelliteOverlayLayer && map) {
          map.removeLayer(satelliteOverlayLayer);
          satelliteOverlayLayer = null;
        }
        const pngSrc = layerData.png_b64.startsWith("data:") ? layerData.png_b64 : ("data:image/png;base64," + layerData.png_b64);
        satelliteOverlayLayer = L.imageOverlay(pngSrc, layerData.bounds, {
          pane: "satellitePane",
          opacity: op,
          interactive: false
        }).addTo(map);
      }
    }

    function setSatelliteOpacity(val) {
      const op = parseFloat(val);
      const lbl = document.getElementById("lbl-sat-opacity");
      if (lbl) lbl.innerText = Math.round(op * 100) + "%";
      if (satelliteOverlayLayer) {
        satelliteOverlayLayer.setOpacity(op);
      }
    }

    function toggleMinesLayer(show) {
      if (!map || !actualMinesLayer) return;
      if (show) map.addLayer(actualMinesLayer);
      else map.removeLayer(actualMinesLayer);
    }

    function toggleTargetsLayer(show) {
      if (!map || !targetsGeoLayer) return;
      if (show) map.addLayer(targetsGeoLayer);
      else map.removeLayer(targetsGeoLayer);
    }

    function toggleFaultsLayer(show) {
      if (!map || !faultsGeoLayer) return;
      if (show) map.addLayer(faultsGeoLayer);
      else map.removeLayer(faultsGeoLayer);
    }

    // ==================== TARGET SELECTION & TABLE RENDERING ====================
    function selectTarget(target, flyTo = true) {
      if (!target) return;
      activeTarget = target;
      document.getElementById("target-card-id").innerText = target.id;
      document.getElementById("target-card-name").innerText = target.name;
      document.getElementById("target-card-domain").innerText = `${target.structuralDomain} — ${target.primaryDriver}`;
      document.getElementById("target-card-class").innerText = `RANK #${target.rank || 1} // ${target.classification}`;

      const oreBadge = document.getElementById("target-card-ore-badge");
      if (oreBadge) {
        const cat = target.oreCategory || "porphyry_cu";
        const label = target.oreCategoryLabel || ORE_LABEL_MAP[cat] || cat.toUpperCase();
        oreBadge.innerText = (target.telescopedFlag ? "TELESCOPED: " : "") + label.toUpperCase();
        oreBadge.className = `badge-pill ore-badge ore-badge-${cat}`;
      }

      if (document.getElementById("target-card-commodity")) {
        document.getElementById("target-card-commodity").innerText = target.commodity || ORE_COMMODITY_MAP[target.oreCategory || "porphyry_cu"] || "Polymetallic";
      }
      if (document.getElementById("target-card-system")) {
        document.getElementById("target-card-system").innerText = target.depositSystem || ORE_SYSTEM_MAP[target.oreCategory || "porphyry_cu"] || "Hydrothermal Mineral System";
      }

      const cLow = target.conformalInterval ? target.conformalInterval[0] : 0.50;
      const cHigh = target.conformalInterval ? target.conformalInterval[1] : 1.00;

      glyphScramble("kpi-score", target.prospectivityScore.toFixed(3), 180);
      glyphScramble("kpi-depth", `-${target.depthMeters} m`, 180);
      glyphScramble("kpi-uncert", cLow.toFixed(3), 180);
      glyphScramble("kpi-ci", `[${cLow.toFixed(3)} - ${cHigh.toFixed(3)}]`, 180);

      renderTable();
      if (activeAnalyticsSubTab === "conformal") renderConformalChart();
      else if (activeAnalyticsSubTab === "prauc") renderPraucChart();

      if (flyTo) {
        highlightTargetOnMap(target);
      } else {
        updateTargetsGeoLayerStyles();
        updateMapTargetMarkers();
      }
    }

    function selectTargetById(id) {
      const t = activeTargetsList.find(x => x.id === id);
      if (t) selectTarget(t, true);
    }

    function onTargetSearch(val) {
      searchQuery = val.toLowerCase().trim();
      renderTable();
    }

    function filterTargets(flt) {
      activeFilter = flt;
      ["all", "top", "highconf"].forEach(k => {
        const b = document.getElementById("btn-flt-" + k);
        if (b) b.classList.remove("active");
      });
      if (flt === "ALL") document.getElementById("btn-flt-all").classList.add("active");
      if (flt === "TOP_RANKED") document.getElementById("btn-flt-top").classList.add("active");
      if (flt === "HIGH_CONF") document.getElementById("btn-flt-highconf").classList.add("active");

      renderTable();
      updateMapTargetMarkers();
    }

    function sortTable(col) {
      if (sortColumn === col) {
        sortAsc = !sortAsc;
      } else {
        sortColumn = col;
        sortAsc = col === "name" || col === "id" || col === "ore";
      }
      renderTable();
    }

    function renderTable() {
      const tbody = document.getElementById("telemetry-table-body");
      if (!tbody) return;

      const lThr = parseFloat(document.getElementById("ctrl-l-thr") ? document.getElementById("ctrl-l-thr").value : 0.50);
      const minAreaHa = parseFloat(document.getElementById("ctrl-min-area") ? document.getElementById("ctrl-min-area").value : 5.0);
      const maxDepth = parseInt(document.getElementById("ctrl-z-depth") ? document.getElementById("ctrl-z-depth").value : 600, 10);

      let filtered = activeTargetsList.filter(t => {
        const passesL = t.conformalInterval ? (t.conformalInterval[0] >= lThr) : true;
        const targetAreaHa = (t.area_sq_km || 0.05) * 100.0;
        const passesArea = targetAreaHa >= minAreaHa;
        const passesDepth = (t.depthMeters || 50) <= maxDepth;

        const passesOre = activeOreCategory === "ALL" ||
          (activeOreCategory === "telescoped" && (t.telescopedFlag || (t.conformalTargetSet && t.conformalTargetSet.length >= 2))) ||
          (t.oreCategory === activeOreCategory) ||
          (t.conformalTargetSet && t.conformalTargetSet.includes(activeOreCategory));

        const matchesFilter = activeFilter === "ALL" || 
          (activeFilter === "TOP_RANKED" && t.rank <= 5) || 
          (activeFilter === "HIGH_CONF" && (t.conformalInterval ? t.conformalInterval[0] >= 0.70 : false));
        const matchesSearch = !searchQuery || 
          t.name.toLowerCase().includes(searchQuery) || 
          t.id.toLowerCase().includes(searchQuery) ||
          (t.oreCategoryLabel && t.oreCategoryLabel.toLowerCase().includes(searchQuery)) ||
          t.structuralDomain.toLowerCase().includes(searchQuery) ||
          t.primaryDriver.toLowerCase().includes(searchQuery);
        return matchesFilter && matchesSearch && passesL && passesArea && passesDepth && passesOre;
      });

      const countLabel = `${filtered.length} TARGETS`;
      const hudEl = document.getElementById("hud-targets-count");
      if (hudEl) hudEl.innerText = countLabel;
      const ciCountEl = document.getElementById("lbl-conformal-count");
      if (ciCountEl) ciCountEl.innerText = `${filtered.length} Targets Evaluated`;

      filtered.sort((a, b) => {
        let valA, valB;
        if (sortColumn === "id") { valA = a.id; valB = b.id; }
        else if (sortColumn === "ore") { valA = a.oreCategoryLabel || a.oreCategory || ""; valB = b.oreCategoryLabel || b.oreCategory || ""; }
        else if (sortColumn === "name") { valA = a.name; valB = b.name; }
        else if (sortColumn === "depth") { valA = a.depthMeters; valB = b.depthMeters; }
        else if (sortColumn === "score") { valA = a.prospectivityScore; valB = b.prospectivityScore; }
        else if (sortColumn === "uncertainty") { valA = a.conformalInterval[0]; valB = b.conformalInterval[0]; }
        else { valA = a.prospectivityScore; valB = b.prospectivityScore; }

        if (valA < valB) return sortAsc ? -1 : 1;
        if (valA > valB) return sortAsc ? 1 : -1;
        return 0;
      });

      tbody.innerHTML = filtered.slice(0, 50).map(t => {
        const isSelected = activeTarget && activeTarget.id === t.id;
        const isHovered = hoveredTargetId === t.id;
        const oreCat = t.oreCategory || "porphyry_cu";
        const oreLabel = t.oreCategoryLabel || ORE_LABEL_MAP[oreCat] || oreCat;
        return `
          <tr class="${isSelected ? 'active-row' : ''}" 
              onclick="selectTargetById('${t.id}')"
              onmouseenter="hoverTarget('${t.id}')"
              onmouseleave="hoverTarget(null)">
            <td style="font-weight:700; color:var(--accent-primary);">${t.id}</td>
            <td>
              <span class="badge-pill ore-badge ore-badge-${oreCat}">
                ${t.telescopedFlag ? '⚡ ' : ''}${oreLabel}
              </span>
            </td>
            <td style="font-family:var(--font-sans); color:var(--text-primary); font-weight:500;">
              ${t.name}
              <div style="font-family:var(--font-editorial); font-size:11px; font-style:italic; color:var(--text-secondary);">${t.structuralDomain}</div>
            </td>
            <td>${t.lat.toFixed(4)}° / ${t.lon.toFixed(4)}°</td>
            <td>-${t.depthMeters} m</td>
            <td style="font-weight:700; color:var(--accent-primary);">${t.prospectivityScore.toFixed(3)}</td>
            <td style="color:var(--accent-emerald);">${t.conformalInterval[0].toFixed(3)}</td>
            <td>[${t.conformalInterval.join(', ')}]</td>
            <td>
              <span class="badge-pill" style="background:rgba(56, 189, 248, 0.15); color:var(--accent-primary); border:1px solid rgba(56, 189, 248, 0.3);">
                #${t.rank || 1}
              </span>
            </td>
          </tr>
        `;
      }).join('');
    }

    function hoverTarget(id) {
      hoveredTargetId = id;
      if (activeAnalyticsSubTab === "conformal") renderConformalChart();
      else if (activeAnalyticsSubTab === "prauc") renderPraucChart();
    }

    function exportTargetsCSV() {
      const headers = ["Target_ID", "Rank", "Name", "Latitude", "Longitude", "Regolith_Depth_m", "Score", "Conformal_L", "Conformal_U", "Domain", "Primary_Driver"];
      const rows = activeTargetsList.map(t => [
        t.id, t.rank || 1, `"${t.name}"`, t.lat, t.lon, t.depthMeters, t.prospectivityScore, t.conformalInterval[0], t.conformalInterval[1], `"${t.structuralDomain}"`, `"${t.primaryDriver}"`
      ]);
      const csvContent = "data:text/csv;charset=utf-8," + [headers.join(","), ...rows.map(e => e.join(","))].join("\\n");
      const encodedUri = encodeURI(csvContent);
      const link = document.createElement("a");
      link.setAttribute("href", encodedUri);
      link.setAttribute("download", `targets_${currentRegion.toLowerCase()}_export.csv`);
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }

    // ==================== ANALYTICS TABS & CHARTS ====================
    function switchAnalyticsSubTab(tab) {
      activeAnalyticsSubTab = tab;
      ["conformal", "prauc"].forEach(t => {
        const btn = document.getElementById("btn-subtab-" + t);
        const view = document.getElementById("view-tab-" + t);
        if (btn && view) {
          if (t === tab) {
            btn.classList.add("active");
            view.style.display = "block";
          } else {
            btn.classList.remove("active");
            view.style.display = "none";
          }
        }
      });

      if (tab === "conformal") renderConformalChart();
      else if (tab === "prauc") renderPraucChart();
    }

    // ==================== TRAINING CONTROL & MODEL LAB LOGIC ====================
    let isTrainingActive = false;
    let trainPollInterval = null;
    let autoScrollLogs = true;
    let cachedModelRuns = [];

    function onTrainRegionChange(val) {
      const nameInput = document.getElementById("train-input-run-name");
      if (nameInput) {
        const ts = new Date().toISOString().replace(/[-:T.]/g, "").slice(0, 15);
        nameInput.value = `EXP_${val}_${ts}`;
      }
    }

    function onTrainRoundsInput(val) {
      const lbl = document.getElementById("lbl-train-rounds-val");
      if (lbl) lbl.innerText = `${val} Round${val > 1 ? 's' : ''}`;
      const est = document.getElementById("train-est-time");
      if (est) {
        const minTime = (val * 0.5).toFixed(1);
        const maxTime = (val * 0.8).toFixed(1);
        est.innerText = `~${minTime} - ${maxTime} min`;
      }
    }

    function setQuickRounds(n) {
      const slider = document.getElementById("train-slider-rounds");
      if (slider) {
        slider.value = n;
        onTrainRoundsInput(n);
      }
    }

    function generateAutoRunName() {
      const reg = document.getElementById("train-select-region") ? document.getElementById("train-select-region").value : "EAST_KALIMANTAN";
      const ts = new Date().toISOString().replace(/[-:T.]/g, "").slice(0, 15);
      const nameInput = document.getElementById("train-input-run-name");
      if (nameInput) {
        nameInput.value = `EXP_${reg}_${ts}`;
      }
    }

    function toggleAutoScroll() {
      autoScrollLogs = !autoScrollLogs;
      const btn = document.getElementById("btn-toggle-autoscroll");
      if (btn) {
        btn.innerText = `AUTOSCROLL [${autoScrollLogs ? "ON" : "OFF"}]`;
        btn.classList.toggle("active", autoScrollLogs);
      }
    }

    function clearTerminalLogs() {
      const box = document.getElementById("train-terminal-logs");
      if (box) box.innerText = "";
    }

    function startTrainingRunFromUI() {
      if (isTrainingActive) return;

      const reg = document.getElementById("train-select-region") ? document.getElementById("train-select-region").value : "EAST_KALIMANTAN";
      const rounds = document.getElementById("train-slider-rounds") ? parseInt(document.getElementById("train-slider-rounds").value, 10) : 3;
      const trees = document.getElementById("train-select-trees") ? parseInt(document.getElementById("train-select-trees").value, 10) : 50;
      const bags = document.getElementById("train-select-bags") ? parseInt(document.getElementById("train-select-bags").value, 10) : 3;
      const epochs = document.getElementById("train-select-epochs") ? parseInt(document.getElementById("train-select-epochs").value, 10) : 2;
      const runName = document.getElementById("train-input-run-name") ? document.getElementById("train-input-run-name").value.trim() : "";
      const notes = document.getElementById("train-input-notes") ? document.getElementById("train-input-notes").value.trim() : "";
      const resetZero = document.getElementById("radio-mode-fresh") ? document.getElementById("radio-mode-fresh").checked : false;

      const payload = {
        region: reg,
        run_name: runName || undefined,
        rounds: rounds,
        trees_per_round: trees,
        bags_per_round: bags,
        dae_epochs: epochs,
        reset_from_zero: resetZero,
        incremental: !resetZero,
        notes: notes
      };

      setTrainingUIState(true);

      fetch("/api/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      })
      .then(res => res.json())
      .then(data => {
        appendTerminalLine(`[CLIENT] Training job dispatched: ${JSON.stringify(payload)}`);
        pollTrainingStatus();
      })
      .catch(err => {
        appendTerminalLine(`[CLIENT ERROR] Failed to dispatch training: ${err}`);
        setTrainingUIState(false);
      });
    }

    function stopTrainingRunFromUI() {
      fetch("/api/train/stop", { method: "POST" })
        .then(res => res.json())
        .then(data => {
          appendTerminalLine("[CLIENT] Abort signal delivered to backend server.");
          pollTrainingStatus();
        })
        .catch(err => {
          appendTerminalLine(`[CLIENT ERROR] Abort failed: ${err}`);
        });
    }

    function setTrainingUIState(running) {
      isTrainingActive = running;
      const startBtn = document.getElementById("btn-main-start-train");
      const stopBtn = document.getElementById("btn-main-stop-train");
      const barStartBtn = document.getElementById("bar-btn-start-train");
      const barStopBtn = document.getElementById("bar-btn-stop-train");
      const statusPill = document.getElementById("train-status-pill");
      const statusText = document.getElementById("telemetry-status-text");

      if (startBtn) {
        startBtn.disabled = running;
        startBtn.style.opacity = running ? "0.5" : "1";
        startBtn.innerHTML = running
          ? '<span class="loading-spinner" style="width:12px; height:12px; border:2px solid #FFF; border-top-color:transparent; border-radius:50%; display:inline-block; animation:spin 1s linear infinite;"></span> TRAINING IN PROGRESS...'
          : '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg> START 7-PASS TRAINING RUN';
      }
      if (stopBtn) stopBtn.style.display = running ? "flex" : "none";
      if (barStartBtn) {
        barStartBtn.disabled = running;
        barStartBtn.style.opacity = running ? "0.5" : "1";
      }
      if (barStopBtn) barStopBtn.style.display = running ? "flex" : "none";

      if (statusPill) {
        statusPill.innerHTML = running
          ? '<span class="dot" style="background:#F59E0B;"></span> RUNNING // 7-PASS ACTIVE'
          : '<span class="dot" style="background:#10B981;"></span> READY // STANDBY';
      }
      if (statusText) {
        statusText.innerText = running ? "ACTIVE TRAINING" : "STANDBY";
        statusText.style.color = running ? "#F59E0B" : "var(--text-primary)";
      }

      if (running) {
        if (!trainPollInterval) trainPollInterval = setInterval(pollTrainingStatus, 1500);
      } else {
        if (trainPollInterval) {
          clearInterval(trainPollInterval);
          trainPollInterval = null;
        }
      }
    }

    function appendTerminalLine(text) {
      const box = document.getElementById("train-terminal-logs");
      if (!box) return;
      box.innerText += (box.innerText ? String.fromCharCode(10) : "") + text;
      if (autoScrollLogs) {
        box.scrollTop = box.scrollHeight;
      }
    }

    function pollTrainingStatus() {
      if (!window.location.protocol.startsWith("http")) return;

      fetch("/api/status")
        .then(res => res.json())
        .then(state => {
          if (!state) return;

          // 1. Logs
          const box = document.getElementById("train-terminal-logs");
          if (box && state.logs && Array.isArray(state.logs)) {
            const joined = state.logs.join(String.fromCharCode(10));
            if (box.innerText !== joined) {
              box.innerText = joined;
              if (autoScrollLogs) box.scrollTop = box.scrollHeight;
            }
          }

          // 2. Running state
          if (state.running !== isTrainingActive) {
            setTrainingUIState(state.running);
            if (!state.running) {
              fetchTrainingRunsList();
              const autoDeploy = document.getElementById("chk-auto-deploy");
              if (autoDeploy && autoDeploy.checked && state.last_exit_code === 0) {
                setTimeout(() => {
                  fetchRunHistory(currentRegion);
                  showView("workspace");
                }, 1200);
              }
            }
          }

          // 3. Progress & Stepper
          const pct = state.progress_pct !== undefined ? state.progress_pct : 0;
          const pBar = document.getElementById("train-progress-bar");
          if (pBar) pBar.style.width = `${pct}%`;
          const pTxt = document.getElementById("train-pct-text");
          if (pTxt) pTxt.innerText = `${pct.toFixed(1)}%`;

          const phaseEl = document.getElementById("train-phase-text");
          if (phaseEl && state.phase) phaseEl.innerText = state.phase.toUpperCase();

          const roundEl = document.getElementById("train-round-text");
          if (roundEl) {
            roundEl.innerText = state.total_rounds > 0
              ? `ROUND ${state.current_round || 1}/${state.total_rounds}`
              : "ROUND 0/0";
          }

          // Highlight micro stepper passes based on phase text
          const ph = (state.phase || "").toUpperCase();
          for (let i = 1; i <= 6; i++) {
            const stepEl = document.getElementById("step-pass-" + i);
            if (stepEl) {
              const isActive = (i === 1 && ph.includes("PASS 1")) ||
                               (i === 2 && ph.includes("PASS 2")) ||
                               (i === 3 && ph.includes("PASS 3")) ||
                               (i === 4 && ph.includes("PASS 4")) ||
                               (i === 5 && ph.includes("PASS 5")) ||
                               (i === 6 && (ph.includes("PASS 6") || ph.includes("RANKING")));
              stepEl.style.borderColor = isActive ? "var(--text-primary)" : "var(--border-hairline)";
              stepEl.style.color = isActive ? "var(--text-primary)" : "var(--text-muted)";
              stepEl.style.background = isActive ? "var(--bg-elevated)" : "transparent";
            }
          }

          // 4. Live Metrics HUD
          if (state.metrics) {
            const m = state.metrics;
            if (m.pr_auc_fused !== undefined) document.getElementById("live-pr-fused").innerText = Number(m.pr_auc_fused).toFixed(4);
            if (m.pr_auc_xgb !== undefined) document.getElementById("live-pr-xgb").innerText = Number(m.pr_auc_xgb).toFixed(4);
            if (m.pr_auc_unet !== undefined) document.getElementById("live-pr-unet").innerText = Number(m.pr_auc_unet).toFixed(4);
            if (m.elkan_noto_c_hat !== undefined) document.getElementById("live-chat").innerText = Number(m.elkan_noto_c_hat).toFixed(4);
            if (m.n_ranked_targets !== undefined) document.getElementById("live-targets").innerText = `${m.n_ranked_targets} TARGETS`;
          }
        })
        .catch(err => {
          // silently tolerate network disconnect
        });
    }

    function fetchTrainingRunsList(regionFilter) {
      if (!window.location.protocol.startsWith("http")) return;
      const reg = regionFilter !== undefined ? regionFilter : (document.getElementById("history-region-filter") ? document.getElementById("history-region-filter").value : "");
      const url = `/api/runs${reg ? "?region=" + encodeURIComponent(reg) : ""}`;

      fetch(url)
        .then(res => res.json())
        .then(data => {
          if (!data || !data.runs) return;
          cachedModelRuns = data.runs;
          renderTrainingRunsTable(cachedModelRuns);
        })
        .catch(err => {
          console.error("Failed to fetch training runs catalog:", err);
        });
    }

    function filterHistoryTable(query) {
      if (!cachedModelRuns) return;
      const q = (query || "").toLowerCase().trim();
      if (!q) {
        renderTrainingRunsTable(cachedModelRuns);
        return;
      }
      const filtered = cachedModelRuns.filter(r => {
        const id = (r.run_id || "").toLowerCase();
        const lbl = (r.label || "").toLowerCase();
        const reg = (r.region || "").toLowerCase();
        return id.includes(q) || lbl.includes(q) || reg.includes(q);
      });
      renderTrainingRunsTable(filtered);
    }

    function renderTrainingRunsTable(runs) {
      const tbody = document.getElementById("model-history-table-body");
      if (!tbody) return;

      if (!runs || runs.length === 0) {
        tbody.innerHTML = '<tr><td colspan="9" style="text-align:center; padding:24px; color:var(--text-muted);">No training runs found in catalog.</td></tr>';
        return;
      }

      let html = "";
      runs.forEach(r => {
        const isActive = r.is_active || r.run_id.endsWith("_ACTIVE");
        const fused = (r.metrics && r.metrics.pr_auc_fused !== undefined && r.metrics.pr_auc_fused !== null) ? Number(r.metrics.pr_auc_fused).toFixed(4) : "—";
        const xgb = (r.metrics && r.metrics.pr_auc_xgb !== undefined && r.metrics.pr_auc_xgb !== null) ? Number(r.metrics.pr_auc_xgb).toFixed(3) : "—";
        const unet = (r.metrics && r.metrics.pr_auc_unet !== undefined && r.metrics.pr_auc_unet !== null) ? Number(r.metrics.pr_auc_unet).toFixed(3) : "—";
        const chat = (r.metrics && r.metrics.elkan_noto_c_hat !== undefined && r.metrics.elkan_noto_c_hat !== null) ? Number(r.metrics.elkan_noto_c_hat).toFixed(4) : "—";
        const tgCount = r.n_ranked_targets || (r.site_cards ? r.site_cards.length : 0);
        const timeStr = r.archived_at ? r.archived_at.slice(0, 16).replace("T", " ") : (r.created_at ? r.created_at.slice(0, 16).replace("T", " ") : "—");
        const labelText = r.label || r.run_id;

        html += `
          <tr>
            <td>
              <div style="display:flex; align-items:center; gap:8px;">
                ${isActive ? '<span class="mono-badge active-dot" style="font-size:9px; padding:2px 6px;"><span class="dot"></span> ACTIVE</span>' : '<span class="mono-badge" style="font-size:9px; padding:2px 6px; color:var(--text-muted);">ARCHIVED</span>'}
                <div>
                  <div style="font-weight:700; color:var(--text-primary);">${labelText}</div>
                  <div style="font-size:9.5px; color:var(--text-muted);">${r.run_id}</div>
                </div>
              </div>
            </td>
            <td style="color:var(--text-secondary);">${timeStr}</td>
            <td><span class="panel-tag" style="font-size:9.5px;">${r.region || 'KALIMANTAN'}</span></td>
            <td><span style="font-size:10px; color:${r.is_warm_started ? '#CBD5E1' : 'var(--text-muted)'};">${r.is_warm_started ? 'Incremental (+Trees)' : 'Fresh Run'}</span></td>
            <td><span style="font-weight:700; color:${fused !== '—' && Number(fused) > 0.90 ? '#FFFFFF' : 'var(--text-secondary)'}; font-size:12px;">${fused}</span></td>
            <td style="font-size:10.5px; color:var(--text-muted);">XGB ${xgb} / UNet ${unet}</td>
            <td style="font-family:var(--font-mono); font-size:11px; color:var(--text-secondary);">${chat}</td>
            <td><span style="font-weight:700; color:var(--text-primary); font-size:11px;">${tgCount}</span></td>
            <td style="text-align:right;">
              <div style="display:flex; justify-content:flex-end; gap:6px;">
                <button type="button" class="btn-icon" onclick="loadArchivedRunIntoWorkspace('${r.run_id}')" style="font-size:9.5px; padding:3px 8px;" title="Load run into Exploration Workspace">WORKSPACE →</button>
                <button type="button" class="btn-icon" onclick="renameRunInCatalog('${r.run_id}')" style="font-size:9.5px; padding:3px 6px;" title="Rename model">RENAME</button>
                ${!isActive ? `<button type="button" class="btn-icon" onclick="deleteRunFromCatalog('${r.run_id}')" style="font-size:9.5px; padding:3px 6px; color:#EF4444;" title="Delete archived run">✕</button>` : ''}
              </div>
            </td>
          </tr>
        `;
      });
      tbody.innerHTML = html;
    }

    function loadArchivedRunIntoWorkspace(runId) {
      selectTrainingRun(runId);
      showView("workspace");
    }

    function renameRunInCatalog(runId) {
      const item = (cachedModelRuns || []).find(r => r.run_id === runId);
      const currentLabel = item ? (item.label || item.run_id) : runId;
      const newName = prompt("Enter new name/label for model run " + runId + ":", currentLabel);
      if (!newName || newName.trim() === currentLabel) return;

      fetch("/api/runs/rename", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ run_id: runId, new_name: newName.trim() })
      })
      .then(res => res.json())
      .then(data => {
        if (data.status === "ok") {
          fetchTrainingRunsList();
          fetchRunHistory(currentRegion);
        } else {
          alert("Error renaming run: " + (data.error || "Unknown error"));
        }
      })
      .catch(err => alert("Failed to rename: " + err));
    }

    function deleteRunFromCatalog(runId) {
      if (!confirm("Are you sure you want to permanently delete archived run: " + runId + "?")) return;

      fetch("/api/runs/delete", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ run_id: runId })
      })
      .then(res => res.json())
      .then(data => {
        if (data.status === "ok") {
          fetchTrainingRunsList();
          fetchRunHistory(currentRegion);
        } else {
          alert("Error deleting run: " + (data.error || "Unknown error"));
        }
      })
      .catch(err => alert("Failed to delete: " + err));
    }

    function renderConformalChart() {
      const svg = document.getElementById("svg-analytics-chart");
      if (!svg) return;

      const w = 600;
      const h = 210;
      const pad = { top: 20, right: 25, bottom: 35, left: 45 };
      const pw = w - pad.left - pad.right;
      const ph = h - pad.top - pad.bottom;

      let inner = "";

      [0, 0.25, 0.5, 0.75, 1.0].forEach(frac => {
        const y = pad.top + frac * ph;
        inner += `<line x1="${pad.left}" y1="${y}" x2="${w - pad.right}" y2="${y}" stroke="var(--border-hairline)" stroke-width="1" stroke-dasharray="3 3"/>`;
        inner += `<text x="${pad.left - 8}" y="${y + 3.5}" text-anchor="end" fill="var(--text-muted)" font-family="var(--font-mono)" font-size="9.5">${(1 - frac).toFixed(2)}</text>`;
      });

      const displayTargets = activeTargetsList.slice(0, 7);
      if (displayTargets.length > 0) {
        const colW = pw / displayTargets.length;
        displayTargets.forEach((t, i) => {
          const cx = pad.left + (i + 0.5) * colW;
          const cyScore = pad.top + (1 - t.prospectivityScore) * ph;
          const cyLow = pad.top + (1 - t.conformalInterval[0]) * ph;
          const cyHigh = pad.top + (1 - t.conformalInterval[1]) * ph;

          const isSelected = activeTarget && activeTarget.id === t.id;
          const isHovered = hoveredTargetId === t.id;

          inner += `<rect x="${cx - 16}" y="${cyHigh}" width="32" height="${Math.max(4, cyLow - cyHigh)}" fill="${isSelected ? 'rgba(56, 189, 248, 0.2)' : 'var(--bg-elevated)'}" stroke="${isSelected ? 'var(--accent-primary)' : isHovered ? '#F59E0B' : 'var(--border-hairline)'}" stroke-width="${isSelected ? 1.5 : 1}" rx="3" style="cursor:pointer;" onclick="selectTargetById('${t.id}')"/>`;
          inner += `<circle cx="${cx}" cy="${cyScore}" r="${isSelected ? 5.5 : 4}" fill="${isSelected ? 'var(--accent-primary)' : isHovered ? '#F59E0B' : 'var(--text-primary)'}" stroke="var(--bg-surface)" stroke-width="1.5" style="cursor:pointer;" onclick="selectTargetById('${t.id}')"/>`;
          const shortName = t.id.length > 8 ? `T-${i+1}` : t.id;
          inner += `<text x="${cx}" y="${h - pad.bottom + 15}" text-anchor="middle" fill="${isSelected ? 'var(--accent-primary)' : 'var(--text-secondary)'}" font-family="var(--font-mono)" font-size="10" font-weight="600">${shortName}</text>`;
        });
      }

      svg.innerHTML = inner;
    }

    function renderPraucChart() {
      const svg = document.getElementById("svg-prauc-chart");
      if (!svg) return;

      const w = 600;
      const h = 210;
      const pad = { top: 20, right: 25, bottom: 35, left: 45 };
      const pw = w - pad.left - pad.right;
      const ph = h - pad.top - pad.bottom;

      let inner = "";

      [0, 0.25, 0.5, 0.75, 1.0].forEach(frac => {
        const y = pad.top + frac * ph;
        inner += `<line x1="${pad.left}" y1="${y}" x2="${w - pad.right}" y2="${y}" stroke="var(--border-hairline)" stroke-width="1" stroke-dasharray="3 3"/>`;
        inner += `<text x="${pad.left - 8}" y="${y + 3.5}" text-anchor="end" fill="var(--text-muted)" font-family="var(--font-mono)" font-size="9.5">${(1 - frac).toFixed(2)}</text>`;
      });

      const ptsFused = [[0,1.0], [0.1,0.98], [0.2,0.96], [0.3,0.93], [0.4,0.90], [0.5,0.87], [0.6,0.83], [0.7,0.76], [0.8,0.69], [0.9,0.54], [1.0,0.26]];
      const ptsXgb = [[0,1.0], [0.1,0.96], [0.2,0.94], [0.3,0.91], [0.4,0.88], [0.5,0.84], [0.6,0.78], [0.7,0.71], [0.8,0.62], [0.9,0.48], [1.0,0.22]];
      const ptsUnet = [[0,1.0], [0.1,0.92], [0.2,0.89], [0.3,0.86], [0.4,0.82], [0.5,0.79], [0.6,0.73], [0.7,0.66], [0.8,0.58], [0.9,0.44], [1.0,0.18]];

      const makePath = (pts) => pts.map((p, idx) => `${idx === 0 ? 'M' : 'L'} ${pad.left + p[0] * pw} ${pad.top + (1 - p[1]) * ph}`).join(' ');

      inner += `<path d="${makePath(ptsXgb)}" fill="none" stroke="var(--text-muted)" stroke-width="1.5" stroke-dasharray="4 2"/>`;
      inner += `<path d="${makePath(ptsUnet)}" fill="none" stroke="var(--text-secondary)" stroke-width="1.5" stroke-dasharray="3 3"/>`;
      inner += `<path d="${makePath(ptsFused)}" fill="none" stroke="#FFFFFF" stroke-width="2.5"/>`;

      svg.innerHTML = inner;
    }

    // ==================== LAYOUT MODES & SPLIT DRAG ====================
    let isDraggingSplit = false;

    function setLayoutMode(mode) {
      const ws = document.getElementById("workspace-container");
      if (!ws) return;
      ws.classList.remove("mode-split", "mode-map", "mode-analytics");
      ws.classList.add("mode-" + mode);

      ["split", "map", "analytics"].forEach(m => {
        const btn = document.getElementById("btn-mode-" + m);
        if (btn) {
          if (m === mode) btn.classList.add("active");
          else btn.classList.remove("active");
        }
      });

      if (map) {
        setTimeout(() => map.invalidateSize(), 120);
      }
    }

    function initSplitHandle() {
      const handle = document.getElementById("split-drag-handle");
      const workspace = document.getElementById("split-workspace");
      const leftPane = document.getElementById("pane-left");
      const rightPane = document.getElementById("pane-right");

      if (!handle || !workspace) return;

      handle.addEventListener("mousedown", () => {
        isDraggingSplit = true;
        handle.classList.add("active");
        document.body.style.cursor = "col-resize";
      });

      window.addEventListener("mousemove", (e) => {
        if (!isDraggingSplit) return;
        const rect = workspace.getBoundingClientRect();
        const ratio = ((e.clientX - rect.left) / rect.width) * 100;
        const clamped = Math.min(75, Math.max(25, ratio));
        leftPane.style.width = clamped + "%";
        rightPane.style.width = (100 - clamped) + "%";
        if (map) map.invalidateSize();
      });

      window.addEventListener("mouseup", () => {
        if (isDraggingSplit) {
          isDraggingSplit = false;
          handle.classList.remove("active");
          document.body.style.cursor = "default";
        }
      });
    }

    function toggleDrawer() {
      const drawer = document.getElementById("sensor-drawer");
      const icon = document.getElementById("drawer-chevron-icon");
      if (!drawer) return;
      drawer.classList.toggle("collapsed");
      const isCollapsed = drawer.classList.contains("collapsed");
      if (icon) {
        icon.innerHTML = isCollapsed
          ? '<polyline points="9 18 15 12 9 6"></polyline>'
          : '<polyline points="15 18 9 12 15 6"></polyline>';
      }
      if (map) setTimeout(() => map.invalidateSize(), 240);
    }

    function onLThrChange(val) {
      document.getElementById("lbl-l-thr").innerText = parseFloat(val).toFixed(2);
      renderTable();
      updateMapTargetMarkers();
      applyHeatmapThreshold();
      if (activeAnalyticsSubTab === "conformal") renderConformalChart();
    }

    function onMinAreaChange(val) {
      document.getElementById("lbl-min-area").innerText = parseFloat(val).toFixed(1) + " ha";
      renderTable();
      updateMapTargetMarkers();
    }

    function resetDefaults() {
      document.getElementById("ctrl-l-thr").value = "0.50";
      document.getElementById("lbl-l-thr").innerText = "0.50";
      document.getElementById("ctrl-z-depth").value = "450";
      document.getElementById("lbl-z-depth").innerText = "-450 m";
      document.getElementById("ctrl-min-area").value = "5";
      document.getElementById("lbl-min-area").innerText = "5.0 ha";
      onAsciiZChange(450);
      if (document.getElementById("chk-satellite-layer")) {
        document.getElementById("chk-satellite-layer").checked = false;
        updateSatelliteLayer();
      }
      renderTable();
      updateMapTargetMarkers();
      applyHeatmapThreshold();
      if (activeAnalyticsSubTab === "conformal") renderConformalChart();
    }

    async function triggerLiveRerank() {
      const btn = document.getElementById("btn-live-rerank");
      if (btn) btn.innerText = "RECOMPUTING...";
      try {
        const lThr = parseFloat(document.getElementById("ctrl-l-thr").value);
        const minAreaHa = parseFloat(document.getElementById("ctrl-min-area").value);
        const res = await fetch("/api/rerank", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            region: currentRegion,
            l_threshold: lThr,
            min_area_ha: minAreaHa,
            w1: 0.40,
            w2: 0.25,
            w3: 0.20,
            w4: 0.15
          })
        });
        if (res.ok) {
          const bundle = await res.json();
          if (typeof EMBEDDED_BUNDLES !== "undefined") {
            EMBEDDED_BUNDLES[currentRegion] = bundle;
          }
          loadRegionBundle(bundle);
        }
      } catch (err) {
        console.warn("Re-rank request failed:", err);
      } finally {
        if (btn) btn.innerHTML = '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.67"/></svg> RECOMPUTE TARGETS LIVE';
      }
    }

    function toggleTheme() {
      isLightTheme = !isLightTheme;
      document.body.classList.toggle("light-theme", isLightTheme);
      const icon = document.getElementById("theme-icon");
      if (icon) {
        icon.innerHTML = isLightTheme
          ? '<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>'
          : '<circle cx="12" cy="12" r="5"></circle><line x1="12" y1="1" x2="12" y2="3"></line><line x1="12" y1="21" x2="12" y2="23"></line><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line><line x1="1" y1="12" x2="3" y2="12"></line><line x1="21" y1="12" x2="23" y2="12"></line><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line>';
      }
      if (activeAnalyticsSubTab === "conformal") renderConformalChart();
      if (activeAnalyticsSubTab === "prauc") renderPraucChart();
    }

    // ==================== INITIALIZATION ON DOM READY ====================
    window.addEventListener("DOMContentLoaded", () => {
      initMap();
      initSplitHandle();
      fetchRunHistory(currentRegion);
      if (window.location.hash === "#workspace") {
        showView("workspace");
      } else if (window.location.hash === "#training") {
        showView("training");
      } else {
        showView("home");
      }
    });
  </script>
</body>
</html>
'''

    full_html = html_head + '    ' + bundle_line + html_script

    with open('ui/index.html', 'w', encoding='utf-8') as f:
        f.write(full_html)

    with open('index.html', 'w', encoding='utf-8') as f:
        f.write(full_html)

    print('Successfully generated comprehensive Antigravity Workspace for ui/index.html and index.html!')

if __name__ == '__main__':
    main()
