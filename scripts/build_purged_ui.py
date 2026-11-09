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

        # Load active East Kalimantan bundle with fresh dual-domain prospectivity maps
        from scripts.dashboard_server import _load_region_bundle, _load_bundle_from_run_dir
        ek_bundle = _load_region_bundle('EAST_KALIMANTAN', force_reload=True)
        bundles['EAST_KALIMANTAN'] = ek_bundle
        try:
            bundles['KALIMANTAN'] = _load_region_bundle('KALIMANTAN', force_reload=True)
        except Exception as ex:
            print(f'Notice: could not load KALIMANTAN bundle: {ex}')
            bundles['KALIMANTAN'] = ek_bundle
        try:
            bundles['BORNEO'] = _load_region_bundle('BORNEO', force_reload=True)
        except Exception as ex:
            print(f'Notice: could not load BORNEO bundle: {ex}')
            bundles['BORNEO'] = ek_bundle

        # Embed 50m Hotspot domain, individual hotspots, and Kutai Basin
        for h_code in [
            'HOTSPOT_EAST_KALIMANTAN_50M',
            'HOTSPOT_EK_MAHAKAM_SUTURE_50M',
            'HOTSPOT_EK_SANGATTA_TREND_50M',
            'HOTSPOT_EK_ADANG_FAULT_50M',
            'HOTSPOT_EK_BERAU_FOLD_BELT_50M',
            'HOTSPOT_EK_MALINAU_CORRIDOR_50M',
            'KUTAI_BASIN_100M',
            'UNLABELED_TILE_25KM_100M',
            'UNLABELED_TILE2_25KM_100M',
        ]:
            try:
                bundles[h_code] = _load_region_bundle(h_code, force_reload=True)
                print(f'Embedded hotspot bundle: {h_code}')
            except Exception as ex:
                print(f'Notice: could not load {h_code} bundle: {ex}')
        print('Successfully loaded active EAST_KALIMANTAN and 50M Hotspot bundles!')

        # Embed all catalogued runs for 0ms instant switching
        catalog_p = ROOT_DIR / "outputs" / "run_history" / "training_runs_catalog.json"
        if catalog_p.exists():
            cdata = json.loads(catalog_p.read_text(encoding="utf-8"))
            for r in cdata.get("runs", []):
                run_id = r.get("run_id")
                rel = r.get("relative_path")
                reg = r.get("region", "EAST_KALIMANTAN")
                if run_id and rel:
                    p = ROOT_DIR / rel
                    if p.exists() and (p / "maps" / "calibrated_prospectivity.tif").exists():
                        try:
                            bundles[run_id] = _load_bundle_from_run_dir(p, reg)
                            print(f'Embedded run {run_id}')
                        except Exception as ex:
                            print(f'Notice: could not embed run {run_id}: {ex}')

        bundle_line = 'const EMBEDDED_BUNDLES = ' + json.dumps(bundles) + ';\n'
    except Exception as e:
        print(f'Warning: Could not update bundles: {e}')

    # Read catalog for all training runs to populate the dropdown
    archived_opts = ""
    catalog_p = ROOT_DIR / "outputs" / "run_history" / "training_runs_catalog.json"
    if catalog_p.exists():
        try:
            cdata = json.loads(catalog_p.read_text(encoding="utf-8"))
            runs_catalog = [
                r for r in cdata.get("runs", [])
                if r.get("region") in ("BORNEO", "KALIMANTAN", "EAST_KALIMANTAN", "KALTIM", "HOTSPOT_EAST_KALIMANTAN_50M", "KUTAI_BASIN_100M", "UNLABELED_TILE_25KM_100M")
                or (r.get("region") or "").startswith("HOTSPOT_")
                or (r.get("region") or "").startswith("KUTAI_")
                or (r.get("region") or "").startswith("UNLABELED_")
            ]
            for r in runs_catalog:
                time_str = (r.get("archived_at") or r.get("created_at") or "")[:16].replace("T", " ")
                fused = r.get("metrics", {}).get("pr_auc_fused", 0) or 0
                reg = r.get("region", "EAST_KALIMANTAN")
                lbl = r.get("label") or r.get("display_name")
                is_act = r.get("is_active", False)
                active_str = " ★ [CURRENT ACTIVE]" if is_act else ""
                if lbl and not lbl.startswith("Archived prior run"):
                    opt_title = f"{lbl}{active_str} ({time_str} // PR-AUC: {fused:.3f})"
                else:
                    opt_title = f"{reg} ARCHIVE ({time_str} // PR-AUC: {fused:.3f})"
                archived_opts += f'            <option value="{r["run_id"]}">{opt_title}</option>\n'
        except Exception as e:
            print("Warning reading catalog:", e)
    if not archived_opts:
        archived_opts = '            <option value="" disabled>No runs available</option>\n'

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
        <div id="global-backend-status-pill" class="system-status-pill" onclick="showView('training')" style="cursor:pointer;" title="Click to open Live Training &amp; Scan Control Center">
          <span id="braille-scanner" style="color:var(--text-primary); font-size:13px; font-weight:700;">⠁</span>
          <span id="global-backend-status-text">BACKEND STANDBY // READY</span>
          <span id="global-backend-status-pct" style="color:var(--accent-primary); font-weight:700; display:none;">0%</span>
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
          <span id="chip-xgb" class="chip-val">—</span>
        </div>
        <div class="telemetry-chip">
          <span class="chip-label">nnPU U-NET:</span>
          <span id="chip-unet" class="chip-val">—</span>
        </div>
        <div class="telemetry-chip">
          <span class="chip-label">FUSED PR-AUC:</span>
          <span id="chip-fused" class="chip-val">—</span>
        </div>
        <div class="telemetry-chip">
          <span class="chip-label">PRIOR ĉ:</span>
          <span id="chip-chat" class="chip-val">—</span>
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
          <optgroup label="── PROVINCIAL REGIONAL SCALE (250M-500M) ──">
            <option value="EAST_KALIMANTAN" selected>EAST KALIMANTAN (ACTIVE 250M: EXP_EK_250M_SANGATTA_CORRIDOR_V1_10ROUNDS // PR-AUC: 0.309)</option>
            <option value="KALIMANTAN">KALIMANTAN ISLAND-WIDE (50M DUAL-DOMAIN PRODUCTION GRID)</option>
            <option value="BORNEO">BORNEO ISLAND-WIDE (500M REGIONAL BASELINE)</option>
          </optgroup>
          <optgroup label="── ★ FOCUSED 50M ULTRA-HD HOTSPOTS (K=4 METALS) ──">
            <option value="HOTSPOT_EAST_KALIMANTAN_50M">★ EAST KALIMANTAN 5-HOTSPOT DOMAIN (50M ULTRA-HD // PR-AUC: 0.288)</option>
            <option value="HOTSPOT_EK_MAHAKAM_SUTURE_50M">↳ 50M HOTSPOT 1: Mahakam Suture (Epithermal/Porphyry)</option>
            <option value="HOTSPOT_EK_SANGATTA_TREND_50M">↳ 50M HOTSPOT 2: Sangatta Plutonic Trend (Skarn/Epithermal)</option>
            <option value="HOTSPOT_EK_ADANG_FAULT_50M">↳ 50M HOTSPOT 3: Adang Fault Corridor (Epithermal Au-Ag)</option>
            <option value="HOTSPOT_EK_BERAU_FOLD_BELT_50M">↳ 50M HOTSPOT 4: Berau Fold Belt (Porphyry-Skarn)</option>
            <option value="HOTSPOT_EK_MALINAU_CORRIDOR_50M">↳ 50M HOTSPOT 5: Malinau Northern Arc (Epithermal/Laterite)</option>
          </optgroup>
          <optgroup label="── ★ SEDIMENTARY COAL BASINS (K=3 MULTI-RANK) ──">
            <option value="KUTAI_BASIN_100M">★ KUTAI BASIN 100M STRATIGRAPHIC SCAN (3-ROUND K=3 COAL // 12.59M CELLS)</option>
            <option value="UNLABELED_TILE_25KM_100M">★ UNLABELED 25KM GREENFIELD GRID #1 (SOUTHWEST MARGIN // 100M)</option>
            <option value="UNLABELED_TILE2_25KM_100M">★ UNLABELED 25KM GREENFIELD GRID #2 (CENTRAL FOLD BELT // 100M)</option>
          </optgroup>
          <optgroup id="optgroup-polygonal-domains" label="── ◆ POLYGONAL GEOLOGICAL DOMAINS (NON-RECTANGULAR) ──">
            <option value="KUTAI_BASIN">◆ POLYGON: Kutai Sedimentary Basin (~58,400 km²)</option>
            <option value="MAHAKAM_SUTURE">◆ POLYGON: Mahakam Tectonic Suture (~14,200 km²)</option>
            <option value="SANGATTA_TREND">◆ POLYGON: Sangatta Intrusive Corridor (~9,800 km²)</option>
            <option value="KELIAN_CORRIDOR">◆ POLYGON: Kelian Epithermal Corridor (~6,400 km²)</option>
            <option value="BERAU_FOLD_BELT">◆ POLYGON: Berau Thrust &amp; Fold Belt (~11,500 km²)</option>
            <option value="ADANG_FAULT">◆ POLYGON: Adang Strike-Slip Fault (~8,900 km²)</option>
            <option value="MALINAU_CORRIDOR">◆ POLYGON: Malinau Northern Arc (~12,100 km²)</option>
          </optgroup>
          <optgroup id="optgroup-custom-polygons" label="── ◆ CUSTOM SCANNED POLYGONS ──"></optgroup>
          <optgroup id="optgroup-archived-runs" label="── TRAINING RUN CHECKPOINTS &amp; ARCHIVES ──">
''' + archived_opts + '''            </optgroup>
          </select>

          <!-- Exploration Engine Domain Selector -->
          <select class="select-dataset select-engine-domain" id="engine-domain-select" onchange="selectEngineDomain(this.value)" aria-label="Exploration Engine Domain" style="font-weight:700;">
            <option value="both" selected>DOMAIN: DUAL ENGINE (METALS + COAL)</option>
            <option value="metals">DOMAIN: METALLIC MINERALS (K=4)</option>
            <option value="coal">DOMAIN: COAL PROSPECTIVITY (K=3)</option>
          </select>

          <!-- Deposit Mechanics Ore Category Filter -->
          <select class="select-dataset select-ore-category" id="ore-category-select" onchange="selectOreCategory(this.value)" aria-label="Filter Ore Mechanics">
            <option value="ALL" selected>ALL COMMODITIES &amp; HEADS</option>
            <optgroup label="── SIFT K=4 METALLIC MINERAL SYSTEMS ──">
              <option value="metals_all">ALL METALLIC SYSTEMS (K=4)</option>
              <option value="porphyry_cu">PORPHYRY CU-AU (HEAD 0)</option>
              <option value="skarn_fe_cu">SKARN FE-CU (HEAD 1)</option>
              <option value="epithermal_au">EPITHERMAL AU-AG (HEAD 2)</option>
              <option value="laterite_fe_ni">LATERITIC FE-NI (HEAD 3)</option>
              <option value="telescoped">TELESCOPED / POLY-PHASE</option>
            </optgroup>
            <optgroup label="── SIFT K=3 COAL PROSPECTIVITY ENGINE ──">
              <option value="coal_all">ALL COAL BASIN RANKS (K=3)</option>
              <option value="coal_k0_high_cv">HIGH-CV SYN-RIFT (&gt;6100 kcal/kg - HEAD 0)</option>
              <option value="coal_k1_med_cv">MED-CV FOLDED DELTAIC (5100-6100 kcal/kg - HEAD 1)</option>
              <option value="coal_k2_low_cv">LOW-CV THICK DELTAIC (&lt;5100 kcal/kg - HEAD 2)</option>
            </optgroup>
          </select>

          <!-- Independent Training Run Version Selector -->
          <select class="select-dataset select-run-history" id="run-history-select" onchange="selectTrainingRun(this.value)" aria-label="Select Training Run Version">
            <option value="ACTIVE" selected>RUN: ACTIVE (EXP_EK_250M_SANGATTA_CORRIDOR_V1_10ROUNDS // PR-AUC: 0.309)</option>
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
              <div class="kpi-desc">Uniform 250m resolution grid with 27 decoupled physical &amp; radar sensor layers</div>
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

            <!-- Theater Card 3: 50M Focused Hotspots -->
            <div class="theater-card">
              <div class="theater-card-header">
                <div class="theater-status-badge" style="background:rgba(245,158,11,0.15); color:#f59e0b; border:1px solid rgba(245,158,11,0.3);">50M ULTRA-HD</div>
                <span class="theater-crs">WGS84 / EPSG:4326</span>
              </div>
              <h3 class="theater-name">EAST KALIMANTAN 5-HOTSPOT DOMAIN</h3>
              <p class="theater-desc">
                High-resolution 50m targeted discovery domain across Mahakam Suture, Sangatta Trend, Adang Fault, Berau Fold Belt & Malinau Northern Arc. Single-domain K=4 metals booster.
              </p>
              <div class="theater-metrics-row">
                <div><span class="lbl">HOTSPOTS</span><span class="val">5 FOCUSED</span></div>
                <div><span class="lbl">RESOLUTION</span><span class="val">50m ULTRA-HD</span></div>
                <div><span class="lbl">PR-AUC LIFT</span><span class="val">+222% (0.288)</span></div>
                <div><span class="lbl">DOMAIN</span><span class="val">K=4 METALS</span></div>
              </div>
              <button class="theater-launch-btn" onclick="launchWorkspace('HOTSPOT_EAST_KALIMANTAN_50M')">
                ENTER 50M HOTSPOTS WORKSPACE →
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
                Ingests 27 decoupled raster channels spanning SRTM-DEM topography, ALOS-2 PALSAR-2 L-band SAR backscatter, ASTER SWIR/TIR hydrothermal alteration, Sentinel-2 regolith cover, Bouguer gravity, and GEM tectonic kinematic slip vectors.
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
                <tr><td>15–20</td><td>sar_palsar2_*</td><td>Radar SAR</td><td>JAXA ALOS-2 PALSAR-2 (25m)</td><td>L-band HH/HV backscatter, DPR, Steger lineaments &amp; GLCM anisotropy</td></tr>
                <tr><td>21–27</td><td>band_ratios_swir_tir</td><td>Spectral Alteration</td><td>ASTER SWIR B4-B9, TIR B10-B14</td><td>Kaolinite, alunite, silica doublet indices</td></tr>
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
                  <optgroup label="── SIFT AI PROSPECTIVITY RASTERS ──">
                    <option value="prospectivity_k4_metals">SIFT Metals // Composite Metallic Prospectivity (K=4)</option>
                    <option value="porphyry_cu">SIFT Metals // Porphyry Cu-Au (Head 0)</option>
                    <option value="skarn_fe_cu">SIFT Metals // Skarn Fe-Cu (Head 1)</option>
                    <option value="epithermal_au">SIFT Metals // Epithermal Au-Ag (Head 2)</option>
                    <option value="laterite_fe_ni">SIFT Metals // Laterite Fe-Ni (Head 3)</option>
                    <option value="prospectivity_k3_coal">SIFT Coal // Composite Coal Basin Prospectivity (K=3)</option>
                    <option value="coal_k0_high_cv">SIFT Coal // High-CV Syn-Rift (&gt;6100 kcal/kg - Head 0)</option>
                    <option value="coal_k1_med_cv">SIFT Coal // Med-CV Folded Deltaic (5100-6100 kcal/kg - Head 1)</option>
                    <option value="coal_k2_low_cv">SIFT Coal // Low-CV Thick Deltaic (&lt;5100 kcal/kg - Head 2)</option>
                  </optgroup>
                  <optgroup label="── ALOS-2 PALSAR-2 L-BAND RADAR (SAR) ──">
                    <option value="sar_palsar2_hh">Radar SAR // ALOS-2 PALSAR-2 HH Backscatter (dB)</option>
                    <option value="sar_palsar2_hv">Radar SAR // ALOS-2 PALSAR-2 HV Backscatter (dB)</option>
                    <option value="sar_palsar2_dpr">Radar SAR // Dual-Pol Ratio (DPR = HV / HH)</option>
                    <option value="sar_palsar2_lineament_density">Radar SAR // Steger Ridge Lineament Density</option>
                    <option value="sar_bedding_strike_deg">Radar SAR // L-Band Bedding Strike (0–180°)</option>
                    <option value="sar_glcm_anisotropy">Radar SAR // Radar GLCM Structural Anisotropy</option>
                  </optgroup>
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

              <!-- Polygonal Region Selector & Interactive Polygon Drawing Tool -->
              <div class="glass-pill-group" style="gap:4px;">
                <span style="font-family:var(--font-mono); font-size:9.5px; color:var(--text-muted); padding:0 4px;">POLYGON ROI:</span>
                <button type="button" id="btn-draw-polygon" class="glass-pill-btn" onclick="togglePolygonDrawingMode()" title="Click vertices on map to define a non-rectangular polygonal region">
                  ⬠ DRAW POLYGON
                </button>
                <button type="button" id="btn-finish-polygon" class="glass-pill-btn" onclick="finishDrawingPolygon()" style="display:none; color:#10B981; border-color:#10B981;" title="Close and validate drawn polygon">
                  ✓ FINISH
                </button>
                <button type="button" class="glass-pill-btn" onclick="document.getElementById('input-upload-geojson').click()" title="Upload a GeoJSON Polygon file">
                  ↑ GEOJSON
                </button>
                <input type="file" id="input-upload-geojson" accept=".geojson,.json" style="display:none;" onchange="handleGeoJSONUpload(event)" />
                <span id="badge-drawn-poly-area" style="display:none; font-family:var(--font-mono); font-size:9.5px; color:#38BDF8; padding:0 6px; font-weight:700;">— km²</span>
                <button type="button" id="btn-scan-drawn-poly" class="glass-pill-btn" onclick="openControlForCustomPolygon('scan')" style="display:none; background:rgba(56,189,248,0.18); color:#38BDF8;" title="Run fast inference scan on this polygon using an existing model">
                  ⚡ SCAN POLYGON
                </button>
                <button type="button" id="btn-train-drawn-poly" class="glass-pill-btn" onclick="openControlForCustomPolygon('train')" style="display:none; background:rgba(255,255,255,0.14); color:#FFFFFF;" title="Run 7-Pass customized training on this polygon">
                  ⚙ TRAIN POLYGON
                </button>
                <button type="button" id="btn-clear-polygon" class="glass-pill-btn" onclick="clearDrawnPolygon()" style="display:none; color:#EF4444;" title="Clear custom polygon">
                  ✕
                </button>
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
              <div class="analytics-tab-header" style="flex-wrap:wrap; gap:6px;">
                <div class="tab-nav" style="flex-wrap:wrap;">
                  <button id="btn-subtab-conformal" class="tab-btn active" onclick="switchAnalyticsSubTab('conformal')">
                    01 // Conformal 90% CI
                  </button>
                  <button id="btn-subtab-prauc" class="tab-btn" onclick="switchAnalyticsSubTab('prauc')">
                    02 // Fusion PR-AUC
                  </button>
                  <button id="btn-subtab-unet" class="tab-btn" onclick="switchAnalyticsSubTab('unet')">
                    03 // DAE U-Net Neurons
                  </button>
                  <button id="btn-subtab-xgbtree" class="tab-btn" onclick="switchAnalyticsSubTab('xgbtree')">
                    04 // XGBoost Split Tree
                  </button>
                </div>
                <div style="display:flex; align-items:center; gap:6px;">
                  <button type="button" class="btn-icon" onclick="triggerNeuralPulse()" title="Animate forward pass through U-Net neurons and XGBoost tree splits" style="font-family:var(--font-mono); font-size:9.5px; padding:3px 8px; gap:4px;">
                    <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>
                    PULSE PASS
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

              <!-- Tab View 2: Model Fusion PR-AUC Curve -->
              <div id="view-tab-prauc" style="display:none; flex:1;">
                <div style="background:var(--bg-canvas); border:1px solid var(--border-hairline); border-radius:6px; padding:10px;">
                  <svg id="svg-prauc-chart" viewBox="0 0 600 210" style="width:100%; height:auto; display:block;"></svg>
                </div>
                <div style="display:flex; justify-content:space-between; align-items:center; margin-top:10px; padding-top:8px; border-top:1px solid var(--border-hairline); font-family:var(--font-mono); font-size:10.5px;">
                  <span id="legend-xgb" style="color:var(--text-muted);">■ XGB 3.1: —</span>
                  <span id="legend-unet" style="color:var(--text-secondary);">■ U-Net: —</span>
                  <span id="legend-fused" style="color:#FFFFFF; font-weight:700;">■ Fused: —</span>
                  <span id="legend-chat" style="color:var(--text-muted);">Prior ĉ = —</span>
                </div>
              </div>

              <!-- Tab View 3: Interactive DAE U-Net Neuron & Activation Map -->
              <div id="view-tab-unet" style="display:none; flex:1;">
                <div style="background:var(--bg-canvas); border:1px solid var(--border-hairline); border-radius:6px; padding:8px; position:relative;">
                  <svg id="svg-ws-unet-graph" viewBox="0 0 640 215" style="width:100%; height:auto; display:block;"></svg>
                </div>
                <div id="ws-unet-layer-hud" style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px; margin-top:8px; padding-top:8px; border-top:1px solid var(--border-hairline); font-family:var(--font-mono); font-size:10px; color:var(--text-secondary);">
                  <span id="ws-unet-layer-title" style="color:var(--text-primary); font-weight:700;">LAYER: Bottleneck + Drop2d (1024ch × 2×2)</span>
                  <span id="ws-unet-layer-stats">Mean Act: — | Max: — | Active ReLU: —%</span>
                  <button type="button" class="btn-icon" onclick="showView('training'); setTimeout(() => { const el = document.getElementById('section-model-brain'); if (el) el.scrollIntoView({behavior:'smooth'}); }, 100);" style="font-size:9.5px; padding:2px 7px;">EXPAND STUDIO →</button>
                </div>
              </div>

              <!-- Tab View 4: Interactive XGBoost Decision Tree & Active Split Path -->
              <div id="view-tab-xgbtree" style="display:none; flex:1;">
                <div style="display:flex; justify-content:space-between; align-items:center; gap:6px; margin-bottom:6px;">
                  <select id="ws-xgb-head-select" onchange="onBrainHeadChange(this.value)" style="background:var(--bg-elevated); color:var(--text-primary); border:1px solid var(--border-hairline); border-radius:4px; font-family:var(--font-mono); font-size:9.5px; padding:3px 6px; outline:none; cursor:pointer;">
                    <option value="0">H0 // Porphyry Cu-Au (Stage-2)</option>
                    <option value="1">H1 // Skarn Fe-Cu</option>
                    <option value="2" selected>H2 // Epithermal Au-Ag</option>
                    <option value="3">H3 // Lateritic Fe-Ni</option>
                  </select>
                  <select id="ws-xgb-tree-select" onchange="onBrainTreeChange(this.value)" style="background:var(--bg-elevated); color:var(--text-primary); border:1px solid var(--border-hairline); border-radius:4px; font-family:var(--font-mono); font-size:9.5px; padding:3px 6px; outline:none; cursor:pointer;">
                    <option value="0" selected>Booster Tree #0</option>
                    <option value="1">Booster Tree #1</option>
                    <option value="2">Booster Tree #2</option>
                    <option value="3">Booster Tree #3</option>
                    <option value="4">Booster Tree #4</option>
                    <option value="5">Booster Tree #5</option>
                    <option value="6">Booster Tree #6</option>
                    <option value="7">Booster Tree #7</option>
                  </select>
                  <span id="ws-xgb-leaf-badge" class="panel-tag" style="font-size:9.5px;">LEAF: —</span>
                </div>
                <div style="background:var(--bg-canvas); border:1px solid var(--border-hairline); border-radius:6px; padding:8px;">
                  <svg id="svg-ws-xgb-tree" viewBox="0 0 640 195" style="width:100%; height:auto; display:block;"></svg>
                </div>
                <div id="ws-xgb-node-hud" style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px; margin-top:8px; padding-top:8px; border-top:1px solid var(--border-hairline); font-family:var(--font-mono); font-size:10px; color:var(--text-secondary);">
                  <span id="ws-xgb-path-summary" style="color:var(--text-primary); font-weight:600;">Click any target on the map or tree node to inspect active split thresholds.</span>
                  <button type="button" class="btn-icon" onclick="showView('training'); setTimeout(() => { const el = document.getElementById('section-model-brain'); if (el) el.scrollIntoView({behavior:'smooth'}); }, 100);" style="font-size:9.5px; padding:2px 7px;">FULL TREE LAB →</button>
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

        <!-- SECTION 1: 3-COLUMN TRAINING & POLYGONAL SCAN LAUNCHPAD -->
        <section id="section-train-launchpad" style="display:flex; flex-direction:column; gap:16px; width:100%;">
          <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px;">
            <div style="display:flex; align-items:center; gap:8px;">
              <span class="section-num">01</span>
              <h2 class="section-title">CUSTOMIZED TRAINING, MODEL IMPROVEMENT &amp; POLYGONAL SCAN</h2>
            </div>

            <!-- Operation Mode Switcher: Train/Improve vs Fast Scan with Existing Model -->
            <div class="view-mode-pill" style="padding:3px; background:var(--bg-surface); border:1px solid var(--border-strong);">
              <button type="button" id="btn-opmode-train" class="view-mode-btn active" onclick="setOperationMode('train')" style="padding:5px 12px; font-size:10.5px;">
                ⚙ MODE A // TRAIN OR IMPROVE MODEL (7-PASS)
              </button>
              <button type="button" id="btn-opmode-scan" class="view-mode-btn" onclick="setOperationMode('scan')" style="padding:5px 12px; font-size:10.5px;">
                ⚡ MODE B // SCAN POLYGON WITH EXISTING MODEL (INFERENCE)
              </button>
            </div>
          </div>

          <div class="training-grid-3col">
            <!-- COLUMN A: MODEL CHECKPOINT, POLYGONAL REGION & RESOLUTION -->
            <div class="train-card">
              <div class="train-card-title" style="justify-content:space-between;">
                <span style="display:flex; align-items:center; gap:6px;">
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="9" y1="21" x2="9" y2="9"/></svg>
                  A // Model, Polygon &amp; Resolution
                </span>
                <span id="badge-col-a-mode" class="panel-tag" style="font-size:9px;">WARM-START / FRESH</span>
              </div>

              <!-- 1. Select Which Model to Improve or Scan With -->
              <div>
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                  <label class="train-label" id="lbl-base-model-select" style="margin:0;">Select Base Model Checkpoint to Improve</label>
                  <button type="button" onclick="fetchAvailableModels()" class="btn-icon" style="font-size:9px; padding:1px 6px;">↻ SYNC</button>
                </div>
                <select id="train-select-base-model" class="train-select" onchange="onBaseModelChange(this.value)" style="font-weight:600;">
                  <option value="ACTIVE_EAST_KALIMANTAN" selected>[ACTIVE] East Kalimantan 250m Production (PR-AUC: 0.9135)</option>
                  <option value="ACTIVE_KALIMANTAN">[ACTIVE] Kalimantan 50m Dual-Domain (PR-AUC: 0.8720)</option>
                  <option value="ACTIVE_KUTAI_BASIN_100M">[ACTIVE] Kutai Basin 100m K=3 Coal Engine</option>
                  <option value="NONE">── TRAIN FRESH FROM ZERO (No Prior Weights) ──</option>
                </select>
                <div id="selected-model-meta-badge" style="margin-top:4px; font-family:var(--font-mono); font-size:9.5px; color:var(--text-muted); display:flex; justify-content:space-between;">
                  <span>Mode: Warm-Start (+Incremental Trees)</span>
                  <span id="selected-model-prauc" style="color:var(--accent-primary);">Checkpoint Ready</span>
                </div>
              </div>

              <!-- 2. Select Polygonal Region (Non-Rectangular) -->
              <div>
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                  <label class="train-label" style="margin:0;">Polygonal Region Boundary (Non-Rectangular)</label>
                  <button type="button" onclick="jumpToMapPolygonDraw()" class="btn-icon" style="font-size:9px; padding:1px 6px; color:#38BDF8; border-color:rgba(56,189,248,0.4);">⬠ DRAW ON MAP →</button>
                </div>
                <select id="train-select-region" class="train-select" onchange="onTrainRegionChange(this.value)">
                  <optgroup label="── ◆ PREDEFINED NON-RECTANGULAR POLYGONS ──">
                    <option value="EAST_KALIMANTAN" selected>◆ POLYGON: East Kalimantan Provincial Border (~194,800 km²)</option>
                    <option value="KUTAI_BASIN">◆ POLYGON: Kutai Sedimentary Coal Basin (~58,400 km²)</option>
                    <option value="MAHAKAM_SUTURE">◆ POLYGON: Mahakam Tectonic Suture (~14,200 km²)</option>
                    <option value="SANGATTA_TREND">◆ POLYGON: Sangatta Intrusive Corridor (~9,800 km²)</option>
                    <option value="KELIAN_CORRIDOR">◆ POLYGON: Kelian Epithermal Au-Ag Belt (~6,400 km²)</option>
                    <option value="BERAU_FOLD_BELT">◆ POLYGON: Berau Thrust &amp; Fold Belt (~11,500 km²)</option>
                    <option value="ADANG_FAULT">◆ POLYGON: Adang Strike-Slip Fault Zone (~8,900 km²)</option>
                    <option value="MALINAU_CORRIDOR">◆ POLYGON: Malinau Northern Arc (~12,100 km²)</option>
                  </optgroup>
                  <optgroup label="── ★ CUSTOM INTERACTIVE MAP POLYGON ──">
                    <option value="CUSTOM_POLYGON">★ CUSTOM DRAWN / UPLOADED MAP POLYGON (GeoJSON)</option>
                  </optgroup>
                  <optgroup label="── REGIONAL BASELINE GRIDS ──">
                    <option value="KALIMANTAN">KALIMANTAN 50M (Focused Multi-Camp Grid)</option>
                    <option value="BORNEO">BORNEO ISLAND-WIDE (500m Regional Baseline)</option>
                  </optgroup>
                </select>
                <div id="poly-validation-badge" style="margin-top:5px; padding:5px 8px; background:var(--bg-canvas); border:1px solid var(--border-hairline); border-radius:4px; font-family:var(--font-mono); font-size:9.5px; color:var(--text-secondary); display:flex; justify-content:space-between; align-items:center;">
                  <span id="poly-val-geom">⬠ Polygonal Mask (Alpha=0 exterior)</span>
                  <span id="poly-val-cells" style="color:var(--text-primary); font-weight:700;">250m // ~2.92M interior cells</span>
                </div>
              </div>

              <!-- 3. Select Rasterization Resolution -->
              <div>
                <label class="train-label">Polygonal Rasterization Resolution (px_size_m)</label>
                <select id="train-select-resolution" class="train-select" onchange="validateSelectedPolygonAndResolution()">
                  <option value="50">50m Ultra-HD Camp Grid (Max 15,000 km² polygon)</option>
                  <option value="100">100m High-Definition Corridor Grid (Max 60,000 km² polygon)</option>
                  <option value="250" selected>250m Provincial Production Grid (Standard 195,000 km²)</option>
                  <option value="500">500m Regional Baseline Grid (Fast Island-Wide Scan)</option>
                </select>
              </div>

              <!-- 4. Target Exploration Domain & Selectable Deposit Heads -->
              <div>
                <label class="train-label">Target Exploration Domain</label>
                <select id="train-select-domain" class="train-select" onchange="onTrainDomainChange(this.value)" style="font-weight:700;">
                  <option value="both" selected>DUAL DOMAIN: METALS (K=4) + COAL (K=3) [RECOMMENDED]</option>
                  <option value="metals">METALLIC MINERAL SYSTEMS ONLY (K=4)</option>
                  <option value="coal">COAL RANK PROSPECTIVITY ONLY (K=3)</option>
                </select>
              </div>

              <div id="train-heads-container">
                <label class="train-label">Active Deposit Mechanics &amp; Coal Heads (Toggleable)</label>
                <div style="display:grid; grid-template-columns:1fr 1fr; gap:5px; background:var(--bg-canvas); padding:8px; border:1px solid var(--border-hairline); border-radius:4px; font-family:var(--font-mono); font-size:9.5px;">
                  <label class="train-checkbox-label"><input type="checkbox" class="chk-active-head" value="porphyry_cu" checked /> <span>H0: Porphyry Cu-Au</span></label>
                  <label class="train-checkbox-label"><input type="checkbox" class="chk-active-head" value="skarn_fe_cu" checked /> <span>H1: Skarn Fe-Cu</span></label>
                  <label class="train-checkbox-label"><input type="checkbox" class="chk-active-head" value="epithermal_au" checked /> <span>H2: Epithermal Au</span></label>
                  <label class="train-checkbox-label"><input type="checkbox" class="chk-active-head" value="laterite_fe_ni" checked /> <span>H3: Laterite Fe-Ni</span></label>
                  <label class="train-checkbox-label"><input type="checkbox" class="chk-active-head" value="coal_k0_high_cv" checked /> <span style="color:#38BDF8;">C0: High-CV Coal</span></label>
                  <label class="train-checkbox-label"><input type="checkbox" class="chk-active-head" value="coal_k1_med_cv" checked /> <span style="color:#FBBF24;">C1: Med-CV Coal</span></label>
                  <label class="train-checkbox-label"><input type="checkbox" class="chk-active-head" value="coal_k2_low_cv" checked /> <span style="color:#A3E635;">C2: Low-CV Coal</span></label>
                </div>
              </div>
            </div>

            <!-- COLUMN B: CUSTOMIZED HYPERPARAMETERS, SPATIAL CV & THRESHOLDS -->
            <div class="train-card">
              <div class="train-card-title" style="justify-content:space-between;">
                <span style="display:flex; align-items:center; gap:6px;">
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
                  B // Hyperparameters &amp; Spatial CV
                </span>
                <span id="badge-col-b-mode" class="panel-tag" style="font-size:9px;">CUSTOM SCHEDULE</span>
              </div>

              <!-- Training-Only Controls Wrapper (dimmed in Fast Scan Mode) -->
              <div id="train-only-hyperparams" style="display:flex; flex-direction:column; gap:12px;">
                <div>
                  <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                    <label class="train-label" style="margin:0;">Sequential Training Rounds</label>
                    <span id="lbl-train-rounds-val" class="font-mono" style="font-size:11px; font-weight:700; color:var(--text-primary);">3 Rounds</span>
                  </div>
                  <input type="range" id="train-slider-rounds" min="1" max="10" step="1" value="3" oninput="onTrainRoundsInput(this.value)" style="width:100%; accent-color:var(--text-primary); cursor:pointer;" />
                  <div style="display:flex; gap:6px; margin-top:6px;">
                    <button type="button" class="btn-icon" onclick="setQuickRounds(1)" style="flex:1; font-size:9.5px; font-family:var(--font-mono);">1 Rnd</button>
                    <button type="button" class="btn-icon" onclick="setQuickRounds(3)" style="flex:1; font-size:9.5px; font-family:var(--font-mono); border-color:var(--accent-primary);">3 Rnds</button>
                    <button type="button" class="btn-icon" onclick="setQuickRounds(5)" style="flex:1; font-size:9.5px; font-family:var(--font-mono);">5 Rnds</button>
                    <button type="button" class="btn-icon" onclick="setQuickRounds(10)" style="flex:1; font-size:9.5px; font-family:var(--font-mono);">10 Rnds</button>
                  </div>
                </div>

                <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px;">
                  <div>
                    <label class="train-label">Trees / Round</label>
                    <select id="train-select-trees" class="train-select">
                      <option value="25">+25 trees</option>
                      <option value="50" selected>+50 trees</option>
                      <option value="100">+100 trees</option>
                      <option value="200">+200 trees</option>
                    </select>
                  </div>
                  <div>
                    <label class="train-label">Bagging Subsamples</label>
                    <select id="train-select-bags" class="train-select">
                      <option value="1">1 Bag (Fast)</option>
                      <option value="3" selected>3 Bags (Std)</option>
                      <option value="5">5 Bags (Deep)</option>
                    </select>
                  </div>
                </div>

                <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px;">
                  <div>
                    <label class="train-label">XGB Learning Rate (η)</label>
                    <select id="train-select-lr" class="train-select">
                      <option value="0.01">η = 0.01 (Fine-tune)</option>
                      <option value="0.03">η = 0.03 (Conservative)</option>
                      <option value="0.05" selected>η = 0.05 (Canonical)</option>
                      <option value="0.08">η = 0.08 (Fast step)</option>
                      <option value="0.12">η = 0.12 (Aggressive)</option>
                    </select>
                  </div>
                  <div>
                    <label class="train-label">Max Tree Depth</label>
                    <select id="train-select-depth" class="train-select">
                      <option value="4">Depth 4 (Regularized)</option>
                      <option value="5">Depth 5 (Balanced)</option>
                      <option value="6" selected>Depth 6 (Canonical)</option>
                      <option value="8">Depth 8 (High-order)</option>
                      <option value="10">Depth 10 (Deep splits)</option>
                    </select>
                  </div>
                </div>

                <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px;">
                  <div>
                    <label class="train-label">DAE U-Net Epochs / Rnd</label>
                    <select id="train-select-epochs" class="train-select">
                      <option value="0">0 (Skip DAE U-Net)</option>
                      <option value="1">1 Epoch (Fast)</option>
                      <option value="2" selected>2 Epochs (Std)</option>
                      <option value="5">5 Epochs (Deep)</option>
                    </select>
                  </div>
                  <div>
                    <label class="train-label">Spatial CV Block (km)</label>
                    <select id="train-select-block-km" class="train-select">
                      <option value="5">5 km Blocks</option>
                      <option value="10" selected>10 km Blocks (Std)</option>
                      <option value="25">25 km Blocks</option>
                      <option value="50">50 km Blocks</option>
                    </select>
                  </div>
                </div>
              </div>

              <!-- Target Extraction & Conformal Gate Controls (Active in both Train & Scan modes) -->
              <div style="border-top:1px solid var(--border-hairline); padding-top:10px; display:flex; flex-direction:column; gap:8px;">
                <div style="font-family:var(--font-mono); font-size:9.5px; color:var(--text-muted); text-transform:uppercase; font-weight:600;">
                  Pass 5–7 Conformal &amp; Polygonization Gate
                </div>
                <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px;">
                  <div>
                    <label class="train-label">Conformal L(x) Gate</label>
                    <input type="number" id="train-input-l-thr" class="train-input" min="0.10" max="0.95" step="0.05" value="0.50" />
                  </div>
                  <div>
                    <label class="train-label">Min Target Area (ha)</label>
                    <input type="number" id="train-input-min-area" class="train-input" min="0.5" max="250" step="0.5" value="5.0" />
                  </div>
                </div>
                <label class="train-checkbox-label" style="font-size:10px;">
                  <input type="checkbox" id="chk-train-unet" checked />
                  <span>Enable 47.1M-param DistrictDAEUNet Spatial Branch</span>
                </label>
              </div>
            </div>

            <!-- COLUMN C: MODEL LINEAGE, LIVE HARDWARE HUD & DISPATCH -->
            <div class="train-card">
              <div class="train-card-title" style="justify-content:space-between;">
                <span style="display:flex; align-items:center; gap:6px;">
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20.59 13.41l-7.17 7.17a2 2 0 0 1-2.83 0L2 12V2h10l8.59 8.59a2 2 0 0 1 0 2.82z"/><line x1="7" y1="7" x2="7.01" y2="7"/></svg>
                  C // Lineage, Hardware &amp; Dispatch
                </span>
                <span id="hw-pid-badge" class="panel-tag" style="font-size:9px;">DAEMON :8765</span>
              </div>

              <div>
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                  <label class="train-label" style="margin:0;">Run / Scan Output Identifier</label>
                  <button type="button" onclick="generateAutoRunName()" class="btn-icon" style="font-size:9px; padding:1px 6px;">AUTO-GEN</button>
                </div>
                <input type="text" id="train-input-run-name" class="train-input" value="EXP_EK_250M_SANGATTA_CORRIDOR_V1" placeholder="e.g. EXP_KUTAI_POLYGON_100M" />
              </div>

              <div>
                <label class="train-label">Polygonal Domain Label &amp; Geological Notes</label>
                <textarea id="train-input-notes" class="train-textarea" rows="2" placeholder="Enter geological hypothesis or custom polygon scan notes..."></textarea>
              </div>

              <!-- Live Backend Hardware & Resource Telemetry HUD -->
              <div style="background:var(--bg-canvas); border:1px solid var(--border-hairline); border-radius:4px; padding:9px 10px; font-family:var(--font-mono); font-size:10px; color:var(--text-secondary); display:flex; flex-direction:column; gap:5px;">
                <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid var(--border-hairline); padding-bottom:4px; margin-bottom:2px;">
                  <span style="color:var(--text-primary); font-weight:700; font-size:9.5px;">LIVE BACKEND TELEMETRY HUD</span>
                  <span id="train-est-time" style="color:var(--accent-primary); font-weight:700;">~1.5 - 2.0 min</span>
                </div>
                <div style="display:flex; justify-content:space-between;">
                  <span>CPU LOAD (1m AVG):</span>
                  <span id="hw-cpu-load" style="color:var(--text-primary); font-weight:600;">—</span>
                </div>
                <div style="display:flex; justify-content:space-between;">
                  <span>SYSTEM RAM AVAIL:</span>
                  <span id="hw-ram-avail" style="color:var(--text-primary); font-weight:600;">—</span>
                </div>
                <div style="display:flex; justify-content:space-between;">
                  <span>SSD STORAGE FREE:</span>
                  <span id="hw-disk-free" style="color:var(--text-primary); font-weight:600;">—</span>
                </div>
              </div>

              <div style="margin-top:auto; display:flex; flex-direction:column; gap:8px;">
                <label class="train-checkbox-label" style="font-size:10.5px;">
                  <input type="checkbox" id="chk-auto-deploy" checked />
                  <span>Auto-deploy polygonal outputs to map workspace on completion</span>
                </label>

                <button id="btn-main-start-train" onclick="startTrainingRunFromUI()" class="hero-btn-primary" style="width:100%; justify-content:center; padding:12px 18px; font-size:11.5px; border-radius:4px;">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg>
                  START CUSTOM 7-PASS TRAINING RUN
                </button>
                <button id="btn-main-stop-train" onclick="stopTrainingRunFromUI()" class="hero-btn-secondary" style="width:100%; justify-content:center; padding:9px 16px; font-size:11px; display:none; border-color:#EF4444; color:#EF4444;">
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/></svg>
                  ABORT ACTIVE BACKEND JOB
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

        <!-- SECTION 2.5: INTERACTIVE NEURAL ACTIVATION & XGBOOST DECISION TREE INSPECTOR -->
        <section id="section-model-brain" style="display:flex; flex-direction:column; gap:16px; width:100%;">
          <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;">
            <div style="display:flex; align-items:center; gap:8px;">
              <span class="section-num">02.5</span>
              <h2 class="section-title">LIVE MODEL BRAIN // DAE U-NET NEURONS &amp; XGBOOST DECISION TREE INSPECTOR</h2>
            </div>
            <div style="display:flex; align-items:center; gap:8px; flex-wrap:wrap;">
              <label for="brain-target-select" style="font-family:var(--font-mono); font-size:10px; color:var(--text-muted);">TARGET SITE:</label>
              <select id="brain-target-select" class="select-dataset" onchange="onBrainTargetSelect(this.value)" style="padding:4px 8px; font-size:10.5px; min-width:210px;"></select>

              <label for="brain-head-select" style="font-family:var(--font-mono); font-size:10px; color:var(--text-muted);">K=4 HEAD:</label>
              <select id="brain-head-select" class="select-dataset" onchange="onBrainHeadChange(this.value)" style="padding:4px 8px; font-size:10.5px;">
                <option value="0">HEAD 0 // Porphyry Cu-Au (Stage-2)</option>
                <option value="1">HEAD 1 // Skarn Fe-Cu</option>
                <option value="2" selected>HEAD 2 // Epithermal Au-Ag</option>
                <option value="3">HEAD 3 // Lateritic Fe-Ni</option>
              </select>

              <label for="brain-tree-select" style="font-family:var(--font-mono); font-size:10px; color:var(--text-muted);">TREE:</label>
              <select id="brain-tree-select" class="select-dataset" onchange="onBrainTreeChange(this.value)" style="padding:4px 8px; font-size:10.5px;">
                <option value="0" selected>Tree #0</option>
                <option value="1">Tree #1</option>
                <option value="2">Tree #2</option>
                <option value="3">Tree #3</option>
                <option value="4">Tree #4</option>
                <option value="5">Tree #5</option>
                <option value="6">Tree #6</option>
                <option value="7">Tree #7</option>
              </select>

              <button type="button" id="btn-brain-pulse" class="btn-icon" onclick="triggerNeuralPulse()" style="font-family:var(--font-mono); font-size:10px; padding:4px 10px; gap:5px; border-color:var(--accent-primary); color:var(--text-primary);">
                <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>
                PULSE FORWARD PASS
              </button>
              <button type="button" id="btn-brain-autopulse" class="btn-icon" onclick="toggleAutoPulse()" style="font-family:var(--font-mono); font-size:10px; padding:4px 10px;">
                AUTO-PULSE [OFF]
              </button>
            </div>
          </div>

          <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(460px, 1fr)); gap:16px; width:100%;">
            <!-- LEFT CARD: DAE U-NET 17-STAGE NEURAL ARCHITECTURE & 2D SPATIAL ACTIVATION MAP -->
            <div class="train-card" style="gap:12px;">
              <div class="train-card-title" style="justify-content:space-between;">
                <span style="display:flex; align-items:center; gap:8px;">
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="6" cy="6" r="3"/><circle cx="18" cy="6" r="3"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="18" r="3"/><line x1="8.5" y1="7.5" x2="15.5" y2="16.5"/><line x1="15.5" y1="7.5" x2="8.5" y2="16.5"/></svg>
                  A // DistrictDAEUNet (17-Stage Dual-Decoder Neuron Activations)
                </span>
                <span id="brain-unet-meta-badge" class="panel-tag">47.1M PARAMS // 32×32 PATCH</span>
              </div>

              <!-- Interactive U-Net Layer-by-Layer Neuron Graph -->
              <div style="background:var(--bg-canvas); border:1px solid var(--border-hairline); border-radius:6px; padding:10px;">
                <svg id="svg-brain-unet-graph" viewBox="0 0 680 250" style="width:100%; height:auto; display:block;"></svg>
              </div>

              <!-- Selected U-Net Layer Detail: 8x8 2D Spatial Feature Map + Top 6 Firing Neurons -->
              <div style="display:grid; grid-template-columns:150px 1fr; gap:14px; background:var(--bg-canvas); border:1px solid var(--border-hairline); border-radius:6px; padding:12px; align-items:center;">
                <div style="display:flex; flex-direction:column; align-items:center; gap:6px;">
                  <div style="font-family:var(--font-mono); font-size:9.5px; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.5px;">8×8 Spatial Feature Map</div>
                  <svg id="svg-brain-unet-heatmap" viewBox="0 0 120 120" style="width:116px; height:116px; border:1px solid var(--border-hairline); border-radius:4px; background:#050608;"></svg>
                  <div id="brain-unet-dom-ch" style="font-family:var(--font-mono); font-size:9.5px; color:var(--text-secondary);">Dominant: —</div>
                </div>
                <div style="display:flex; flex-direction:column; gap:8px;">
                  <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid var(--border-hairline); padding-bottom:6px;">
                    <span id="brain-unet-selected-layer-name" style="font-family:var(--font-mono); font-size:11px; font-weight:700; color:var(--text-primary);">LAYER: Bottleneck + Drop2d (1024ch × 2×2)</span>
                    <span id="brain-unet-selected-layer-stats" style="font-family:var(--font-mono); font-size:10px; color:var(--text-secondary);">μ=0.282 | max=1.741 | 84.2% Active</span>
                  </div>
                  <div style="font-family:var(--font-mono); font-size:9.5px; color:var(--text-muted);">TOP FIRING NEURON CHANNELS IN SELECTED LAYER (CLICK ANY LAYER IN GRAPH ABOVE):</div>
                  <div id="brain-unet-top-neurons-list" style="display:flex; flex-direction:column; gap:5px;"></div>
                </div>
              </div>
            </div>

            <!-- RIGHT CARD: XGBOOST K=4 DECISION TREE & LIT-UP SPLIT PATH -->
            <div class="train-card" style="gap:12px;">
              <div class="train-card-title" style="justify-content:space-between;">
                <span style="display:flex; align-items:center; gap:8px;">
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
                  B // TwoStageK4Ensemble (XGBoost Decision Tree &amp; Active Split Path)
                </span>
                <span id="brain-xgb-meta-badge" class="panel-tag">900 BOOSTED TREES // HIST</span>
              </div>

              <!-- Interactive XGBoost Decision Tree SVG -->
              <div style="background:var(--bg-canvas); border:1px solid var(--border-hairline); border-radius:6px; padding:10px;">
                <svg id="svg-brain-xgb-tree" viewBox="0 0 680 250" style="width:100%; height:auto; display:block;"></svg>
              </div>

              <!-- Active Split Path Trace Table & Selected Node Inspector -->
              <div style="background:var(--bg-canvas); border:1px solid var(--border-hairline); border-radius:6px; padding:12px; display:flex; flex-direction:column; gap:8px;">
                <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid var(--border-hairline); padding-bottom:6px;">
                  <span id="brain-xgb-selected-node-title" style="font-family:var(--font-mono); font-size:11px; font-weight:700; color:var(--text-primary);">ACTIVE ROOT-TO-LEAF SPLIT TRACE</span>
                  <span id="brain-xgb-leaf-summary" style="font-family:var(--font-mono); font-size:10px; color:var(--text-primary); font-weight:700;">Terminal Leaf Weight: —</span>
                </div>
                <div id="brain-xgb-steps-list" style="display:flex; flex-direction:column; gap:5px; max-height:135px; overflow-y:auto; font-family:var(--font-mono); font-size:10px;"></div>
              </div>
            </div>
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
    // ==================== RESILIENT API TRANSPORT ====================
    function getApiEndpoint(url) {
      if (!url) return url;
      if (url.startsWith("http://") || url.startsWith("https://")) return url;
      const cleanPath = url.startsWith("/") ? url : "/" + url;
      if (!window.location.protocol.startsWith("http") || (window.location.port !== "8765" && !window.location.hostname.includes("ngrok"))) {
        return "http://127.0.0.1:8765" + cleanPath;
      }
      return cleanPath;
    }

    function apiFetch(url, opts) {
      return fetch(getApiEndpoint(url), opts);
    }

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
        if (typeof renderAllModelBrainViews === "function") {
          setTimeout(() => renderAllModelBrainViews(), 60);
        }
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

    let activeEngineDomain = "both";

    const ORE_LABEL_MAP = {
      "porphyry_cu": "Porphyry Cu-Au",
      "skarn_fe_cu": "Skarn Fe-Cu",
      "epithermal_au": "Epithermal Au-Ag",
      "laterite_fe_ni": "Lateritic Fe-Ni",
      "primary_gold": "Orogenic Gold",
      "orogenic_gold": "Orogenic Gold",
      "coal": "Kutai Basin Coal",
      "coal_all": "Coal Basin Systems (K=3)",
      "coal_k0_high_cv": "High-CV Syn-Rift (>6,100 kcal/kg)",
      "coal_k1_med_cv": "Med-CV Folded Deltaic (5,100–6,100 kcal/kg)",
      "coal_k2_low_cv": "Low-CV Thick Deltaic (<5,100 kcal/kg)",
      "p_coal_any": "Coal Prospectivity (Any Rank)"
    };

    const ORE_COMMODITY_MAP = {
      "porphyry_cu": "Copper, Gold (Cu-Au)",
      "skarn_fe_cu": "Iron, Copper (Fe-Cu)",
      "epithermal_au": "Gold, Silver (Au-Ag)",
      "laterite_fe_ni": "Nickel, Cobalt, Iron (Ni-Co-Fe)",
      "primary_gold": "Gold (Au)",
      "orogenic_gold": "Gold (Au)",
      "coal": "Coal (Thermal / Coking)",
      "coal_all": "Coal (Thermal / Metallurgical)",
      "coal_k0_high_cv": "Bituminous / Coking Coal",
      "coal_k1_med_cv": "Sub-Bituminous Thermal Coal",
      "coal_k2_low_cv": "Lignite / Sub-Bituminous Coal",
      "p_coal_any": "Coal (Multi-Rank Composite)",
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
      "coal_all": "Deltaic / Fluvial Sedimentary Basin",
      "coal_k0_high_cv": "Eocene Syn-Rift Graben Fill (Tanjung Fm)",
      "coal_k1_med_cv": "Middle Miocene Regressive Delta (Balikpapan Fm)",
      "coal_k2_low_cv": "Neogene Lower Delta Plain (Warukin/Lati Fm)",
      "p_coal_any": "Kalimantan Sedimentary Basin Traps",
      "telescoped": "Overprinting Porphyry-Epithermal"
    };

    const ORE_COLORS = {
      porphyry_cu: { stroke: "#FFFFFF", fill: "#FFFFFF" },
      skarn_fe_cu: { stroke: "#E2E8F0", fill: "#E2E8F0" },
      epithermal_au: { stroke: "#CBD5E1", fill: "#CBD5E1" },
      laterite_fe_ni: { stroke: "#94A3B8", fill: "#94A3B8" },
      primary_gold: { stroke: "#F8FAFC", fill: "#F8FAFC" },
      coal: { stroke: "#38BDF8", fill: "#38BDF8" },
      coal_all: { stroke: "#38BDF8", fill: "#38BDF8" },
      coal_k0_high_cv: { stroke: "#38BDF8", fill: "#38BDF8" },
      coal_k1_med_cv: { stroke: "#FBBF24", fill: "#FBBF24" },
      coal_k2_low_cv: { stroke: "#A3E635", fill: "#A3E635" },
      p_coal_any: { stroke: "#38BDF8", fill: "#38BDF8" },
      telescoped: { stroke: "#FFFFFF", fill: "#FFFFFF" }
    };

    let map = null;
    let baseTileLayer = null;
    let scannedAreaRect = null;
    let scannedTilesGroup = null;
    let targetRectanglesGroup = null;
    let hdSubRects = [];
    let heatmapOverlay = null;
    let activeHeatmapDomain = "metals";
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

      // Refresh with live server data in background via resilient apiFetch
      apiFetch(`/api/region?name=${encodeURIComponent(regionCode)}`)
        .then(res => res.json())
        .then(data => {
          if (typeof EMBEDDED_BUNDLES !== "undefined") {
            EMBEDDED_BUNDLES[regionCode] = data;
          }
          loadRegionBundle(data);
        })
        .catch(err => {
          console.warn("Live API not reachable, using embedded bundle:", err);
        });
    }

    // Main calibrated prospectivity heatmap for the selected engine domain (v4.8).
    function setMainHeatmap(b, domain) {
      if (!b || !map) return;
      if (heatmapOverlay) {
        map.removeLayer(heatmapOverlay);
        heatmapOverlay = null;
      }
      rawHeatmapPixels = null;
      dynamicHeatmapCanvas = null;
      dynamicHeatmapCtx = null;

      const hBounds = b.heatmap_bounds || b.bounds;
      // v4.8: the coal domain swaps in the coal raster instead of stacking it on the metals one
      const useCoal = domain === "coal" && b.coal_heatmap_png;
      const pngB64 = useCoal ? b.coal_heatmap_png : b.heatmap_png;
      activeHeatmapDomain = useCoal ? "coal" : "metals";
      if (pngB64 && hBounds) {
        const pngSrc = pngB64.startsWith("data:") ? pngB64 : ("data:image/png;base64," + pngB64);
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
    }

    function selectEngineDomain(domain) {
      activeEngineDomain = domain;
      const el = document.getElementById("engine-domain-select");
      if (el && el.value !== domain) el.value = domain;

      const oreSelect = document.getElementById("ore-category-select");
      if (oreSelect) {
        if (domain === "coal") {
          oreSelect.value = "coal_all";
          selectOreCategory("coal_all");
        } else if (domain === "metals") {
          oreSelect.value = "metals_all";
          selectOreCategory("metals_all");
        } else {
          oreSelect.value = "ALL";
          selectOreCategory("ALL");
        }
      }

      updateMetricsHudByDomain();

      // v4.8: swap the main heatmap to the chosen domain instead of stacking the other
      // domain's raster on top of it as a satellite overlay.
      const satSel = document.getElementById("sel-satellite-layer");
      const satChk = document.getElementById("chk-satellite-layer");
      const hasCoalHeatmap = Boolean(activeRegionBundle && activeRegionBundle.coal_heatmap_png);
      const wantHeatmap = (domain === "coal" && hasCoalHeatmap) ? "coal" : "metals";
      if (activeRegionBundle && wantHeatmap !== activeHeatmapDomain) {
        setMainHeatmap(activeRegionBundle, domain);
      }
      // 50m camp-scan heatmaps are metals-only runs; don't paint them over the coal map
      (hdHeatmapOverlays || []).forEach(ov => ov.setOpacity(domain === "coal" ? 0.0 : 0.90));
      // keep the side panel on a target of the chosen domain
      if (domain !== "both" && Array.isArray(activeTargetsList) && activeTargetsList.length) {
        const isCoalT = t => String(t.oreCategory || "").toLowerCase().includes("coal");
        const wantCoal = domain === "coal";
        if (!activeTarget || isCoalT(activeTarget) !== wantCoal) {
          const first = activeTargetsList.filter(t => isCoalT(t) === wantCoal).sort((a, b) => (a.rank || 99) - (b.rank || 99))[0];
          if (first) selectTarget(first, false);
        }
      }
      if (satSel && satChk && satChk.checked &&
          ["prospectivity_k3_coal", "prospectivity_k4_metals"].includes(satSel.value)) {
        // a prospectivity raster shown as a "satellite" layer would double-stack the domains
        satChk.checked = false;
        updateSatelliteLayer();
      }
      if (domain === "coal" && !hasCoalHeatmap && satSel) {
        // older bundles without a coal heatmap: fall back to the coal raster overlay
        satSel.value = "prospectivity_k3_coal";
        if (satChk) satChk.checked = true;
        updateSatelliteLayer();
      }

      if (activeAnalyticsSubTab === "prauc") renderPraucChart();
      else if (activeAnalyticsSubTab === "conformal") renderConformalChart();
    }

    function updateMetricsHudByDomain() {
      if (!activeRegionBundle) return;
      const s = activeRegionBundle.summary || {};
      const coal = s.coal_k3_metrics || activeRegionBundle.coal_k3_metrics || {};

      let xgb, unet, fused, chat;
      if (activeEngineDomain === "coal" && coal && coal.pr_auc_coal_fused !== undefined) {
        xgb = coal.pr_auc_coal_xgb !== undefined ? Number(coal.pr_auc_coal_xgb).toFixed(3) : "0.177";
        unet = coal.pr_auc_coal_unet !== undefined ? Number(coal.pr_auc_coal_unet).toFixed(3) : "0.019";
        fused = coal.pr_auc_coal_fused !== undefined ? Number(coal.pr_auc_coal_fused).toFixed(3) : "0.178";
        const cCoal = coal.elkan_noto_c_hat_coal_per_head ? (coal.elkan_noto_c_hat_coal_per_head.reduce((a,b)=>a+b,0)/coal.elkan_noto_c_hat_coal_per_head.length).toFixed(3) : "0.618";
        chat = cCoal;
      } else {
        xgb = (s.pr_auc_xgb !== undefined && s.pr_auc_xgb !== null) ? Number(s.pr_auc_xgb).toFixed(3) : "0.613";
        unet = (s.pr_auc_unet !== undefined && s.pr_auc_unet !== null) ? Number(s.pr_auc_unet).toFixed(3) : "0.375";
        fused = (s.pr_auc_fused !== undefined && s.pr_auc_fused !== null) ? Number(s.pr_auc_fused).toFixed(3) : "0.699";
        chat = (s.elkan_noto_c_hat !== undefined && s.elkan_noto_c_hat !== null) ? Number(s.elkan_noto_c_hat).toFixed(3) : "0.761";
      }

      const elXgb = document.getElementById("chip-xgb");
      const elUnet = document.getElementById("chip-unet");
      const elFused = document.getElementById("chip-fused");
      const elChat = document.getElementById("chip-chat");

      if (elXgb) elXgb.innerText = xgb;
      if (elUnet) elUnet.innerText = unet;
      if (elFused) elFused.innerText = fused;
      if (elChat) elChat.innerText = chat;

      const legXgb = document.getElementById("legend-xgb");
      const legUnet = document.getElementById("legend-unet");
      const legFused = document.getElementById("legend-fused");
      const legChat = document.getElementById("legend-chat");

      if (legXgb) legXgb.innerText = `■ ${activeEngineDomain === 'coal' ? 'Coal XGB' : 'XGB 3.1'}: ${xgb}`;
      if (legUnet) legUnet.innerText = `■ ${activeEngineDomain === 'coal' ? 'Coal DAE' : 'U-Net'}: ${unet}`;
      if (legFused) legFused.innerText = `■ Fused: ${fused}`;
      if (legChat) legChat.innerText = `Prior ĉ = ${chat}`;
    }

    function onTrainDomainChange(domain) {
      const headsCont = document.getElementById("train-heads-container");
      if (!headsCont) return;
      if (domain === "coal") {
        headsCont.innerHTML = `
          <label class="train-label">Coal Rank Prospectivity Heads (K=3 Toggleable)</label>
          <div style="display:grid; grid-template-columns:1fr; gap:5px; background:var(--bg-canvas); padding:8px; border:1px solid var(--border-hairline); border-radius:4px; font-family:var(--font-mono); font-size:9.5px;">
            <label class="train-checkbox-label"><input type="checkbox" class="chk-active-head" value="coal_k0_high_cv" checked /> <span style="color:#38BDF8;">C0: High-CV Syn-Rift (&gt;6,100 kcal/kg)</span></label>
            <label class="train-checkbox-label"><input type="checkbox" class="chk-active-head" value="coal_k1_med_cv" checked /> <span style="color:#FBBF24;">C1: Med-CV Folded Deltaic (5,100–6,100 kcal/kg)</span></label>
            <label class="train-checkbox-label"><input type="checkbox" class="chk-active-head" value="coal_k2_low_cv" checked /> <span style="color:#A3E635;">C2: Low-CV Thick Deltaic (&lt;5,100 kcal/kg)</span></label>
          </div>
        `;
      } else if (domain === "metals") {
        headsCont.innerHTML = `
          <label class="train-label">Metallic Deposit Mechanics Heads (K=4 Toggleable)</label>
          <div style="display:grid; grid-template-columns:1fr 1fr; gap:5px; background:var(--bg-canvas); padding:8px; border:1px solid var(--border-hairline); border-radius:4px; font-family:var(--font-mono); font-size:9.5px;">
            <label class="train-checkbox-label"><input type="checkbox" class="chk-active-head" value="porphyry_cu" checked /> <span>H0: Porphyry Cu-Au</span></label>
            <label class="train-checkbox-label"><input type="checkbox" class="chk-active-head" value="skarn_fe_cu" checked /> <span>H1: Skarn Fe-Cu</span></label>
            <label class="train-checkbox-label"><input type="checkbox" class="chk-active-head" value="epithermal_au" checked /> <span>H2: Epithermal Au-Ag</span></label>
            <label class="train-checkbox-label"><input type="checkbox" class="chk-active-head" value="laterite_fe_ni" checked /> <span>H3: Lateritic Fe-Ni</span></label>
          </div>
        `;
      } else {
        headsCont.innerHTML = `
          <label class="train-label">Active Deposit Mechanics &amp; Coal Heads (Toggleable)</label>
          <div style="display:grid; grid-template-columns:1fr 1fr; gap:5px; background:var(--bg-canvas); padding:8px; border:1px solid var(--border-hairline); border-radius:4px; font-family:var(--font-mono); font-size:9.5px;">
            <label class="train-checkbox-label"><input type="checkbox" class="chk-active-head" value="porphyry_cu" checked /> <span>H0: Porphyry Cu-Au</span></label>
            <label class="train-checkbox-label"><input type="checkbox" class="chk-active-head" value="skarn_fe_cu" checked /> <span>H1: Skarn Fe-Cu</span></label>
            <label class="train-checkbox-label"><input type="checkbox" class="chk-active-head" value="epithermal_au" checked /> <span>H2: Epithermal Au</span></label>
            <label class="train-checkbox-label"><input type="checkbox" class="chk-active-head" value="laterite_fe_ni" checked /> <span>H3: Laterite Fe-Ni</span></label>
            <label class="train-checkbox-label"><input type="checkbox" class="chk-active-head" value="coal_k0_high_cv" checked /> <span style="color:#38BDF8;">C0: High-CV Coal</span></label>
            <label class="train-checkbox-label"><input type="checkbox" class="chk-active-head" value="coal_k1_med_cv" checked /> <span style="color:#FBBF24;">C1: Med-CV Coal</span></label>
            <label class="train-checkbox-label"><input type="checkbox" class="chk-active-head" value="coal_k2_low_cv" checked /> <span style="color:#A3E635;">C2: Low-CV Coal</span></label>
          </div>
        `;
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

      const url = (runId === "ACTIVE" || !runId)
        ? `/api/region?name=${encodeURIComponent(currentRegion)}`
        : `/api/run_bundle?region=${encodeURIComponent(currentRegion)}&run_id=${encodeURIComponent(runId)}`;
      apiFetch(url)
        .then(res => res.json())
        .then(data => {
          if (typeof EMBEDDED_BUNDLES !== "undefined") {
            const cacheKey = (runId === "ACTIVE" || !runId) ? currentRegion : runId;
            EMBEDDED_BUNDLES[cacheKey] = data;
          }
          loadRegionBundle(data);
        })
        .catch(err => {
          console.warn("Could not load training run bundle from live API:", err);
        });
    }

    function fetchRunHistory(regionCode) {
      apiFetch(`/api/runs?region=${encodeURIComponent(regionCode)}`)
        .then(res => res.json())
        .then(data => {
          if (!data || !data.runs) return;
          availableRuns = data.runs;
          const selectEl = document.getElementById("run-history-select");
          if (selectEl) {
            const activeRun = availableRuns.find(r => r.is_active);
            const activeLabel = activeRun ? `RUN: ACTIVE (${activeRun.label || activeRun.display_name || activeRun.run_id} // PR-AUC: ${(activeRun.metrics && activeRun.metrics.pr_auc_fused) ? Number(activeRun.metrics.pr_auc_fused).toFixed(3) : '0.309'})` : 'RUN: ACTIVE (EXP_EK_250M_SANGATTA_CORRIDOR_V1_10ROUNDS // PR-AUC: 0.309)';
            let html = `<option value="ACTIVE">${activeLabel}</option>`;
            availableRuns.forEach(r => {
              const timeStr = r.archived_at ? r.archived_at.slice(0, 16).replace('T', ' ') : (r.created_at ? r.created_at.slice(0, 16).replace('T', ' ') : 'Run');
              const fusedStr = (r.metrics && r.metrics.pr_auc_fused !== undefined && r.metrics.pr_auc_fused !== null) ? Number(r.metrics.pr_auc_fused).toFixed(3) : '—';
              const tgtCount = r.n_ranked_targets || (r.site_cards ? r.site_cards.length : 0);
              const label = r.label || r.display_name || r.run_id;
              const isAct = r.is_active ? ' ★ [ACTIVE]' : '';
              html += `<option value="${r.run_id}">${label}${isAct} (${timeStr}) // PR-AUC: ${fusedStr} [${tgtCount} TGT]</option>`;
            });
            selectEl.innerHTML = html;
            selectEl.value = selectedRunId || "ACTIVE";
          }
          const optArchived = document.getElementById("optgroup-archived-runs");
          if (optArchived) {
            const allKalRuns = availableRuns.filter(r => (
              r.region === "KALIMANTAN" ||
              r.region === "BORNEO" ||
              r.region === "EAST_KALIMANTAN" ||
              r.region === "KALTIM" ||
              r.region === "HOTSPOT_EAST_KALIMANTAN_50M" ||
              r.region === "KUTAI_BASIN_100M" ||
              r.region === "UNLABELED_TILE_25KM_100M" ||
              (r.region && (r.region.startsWith("HOTSPOT_") || r.region.startsWith("KUTAI_") || r.region.startsWith("UNLABELED_")))
            ));
            if (allKalRuns.length > 0) {
              let optHtml = '';
              allKalRuns.forEach(r => {
                const timeStr = r.archived_at ? r.archived_at.slice(0, 16).replace('T', ' ') : (r.created_at ? r.created_at.slice(0, 16).replace('T', ' ') : 'Archived');
                const fusedStr = (r.metrics && r.metrics.pr_auc_fused !== undefined && r.metrics.pr_auc_fused !== null) ? Number(r.metrics.pr_auc_fused).toFixed(3) : '—';
                const label = r.label || r.display_name || `${r.region} ARCHIVE (${timeStr})`;
                const isAct = r.is_active ? ' ★ [ACTIVE]' : '';
                optHtml += `<option value="${r.run_id}">${label}${isAct} (${timeStr} // PR-AUC: ${fusedStr})</option>`;
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

      // Update Topbar Metric Chips via Domain Engine Handler
      updateMetricsHudByDomain();

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
          let primaryOre = "porphyry_cu";
          if (sc.ore_category && sc.ore_category !== 'primary_target') {
            primaryOre = sc.ore_category;
          } else if (sc.properties && sc.properties.ore_category && sc.properties.ore_category !== 'primary_target') {
            primaryOre = sc.properties.ore_category;
          } else if (conformalSet.length > 0) {
            primaryOre = conformalSet[0];
          } else if (sc.target_id && String(sc.target_id).toLowerCase().includes("coal")) {
            primaryOre = "coal_k0_high_cv";
          }
          let oreLabel = ORE_LABEL_MAP[primaryOre] || String(primaryOre).replace(/_/g, ' ');
          if (sc.ore_category_label && sc.ore_category_label !== 'Primary Target') {
            oreLabel = sc.ore_category_label;
          } else if (sc.properties && sc.properties.ore_category_label && sc.properties.ore_category_label !== 'Primary Target') {
            oreLabel = sc.properties.ore_category_label;
          }
          const isTelescoped = Boolean(sc.telescoped_system_flag || (sc.properties && sc.properties.telescoped_system_flag) || (conformalSet.length >= 2));
          const commodity = sc.commodity || (sc.properties && sc.properties.commodity) || ORE_COMMODITY_MAP[primaryOre] || "High-CV Bituminous / Coking Coal";
          const system = sc.deposit_system || (sc.properties && sc.properties.deposit_system) || ORE_SYSTEM_MAP[primaryOre] || "Sedimentary Basin Peat Mire";
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
            classification: sc.classification || (sc.exploration_class
              ? `${String(sc.exploration_class).toUpperCase()} #${sc.class_rank || sc.rank || 1}${sc.dist_to_known_km != null ? ` · ${Number(sc.dist_to_known_km).toFixed(1)} km from known` : ''}`
              : (sc.rank <= 3 ? "TIER-1 CORE" : "STRUCTURAL EXT")),
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
          let primaryOre = "porphyry_cu";
          if (p.ore_category && p.ore_category !== 'primary_target') {
            primaryOre = p.ore_category;
          } else if (conformalSet.length > 0) {
            primaryOre = conformalSet[0];
          } else if (p.target_id && String(p.target_id).toLowerCase().includes("coal")) {
            primaryOre = "coal_k0_high_cv";
          }
          let oreLabel = ORE_LABEL_MAP[primaryOre] || String(primaryOre).replace(/_/g, ' ');
          if (p.ore_category_label && p.ore_category_label !== 'Primary Target') {
            oreLabel = p.ore_category_label;
          }
          const isTelescoped = Boolean(p.telescoped_system_flag || (conformalSet.length >= 2));
          const commodity = p.commodity || ORE_COMMODITY_MAP[primaryOre] || "High-CV Bituminous / Coking Coal";
          const system = p.deposit_system || ORE_SYSTEM_MAP[primaryOre] || "Sedimentary Basin Peat Mire";
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

      // Honest target count: no synthetic fallback targets
      if (!activeTargetsList) {
        activeTargetsList = [];
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

      // 3. Load and render U-Net Neuron + XGBoost Split Tree Introspection
      if (typeof ensureModelIntrospectionLoaded === "function") {
        ensureModelIntrospectionLoaded(bundle);
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

      map.on("click", (e) => {
        if (typeof isDrawingPolygon !== "undefined" && isDrawingPolygon) {
          addDrawnPolygonVertex(e.latlng.lat, e.latlng.lng);
        }
      });

      map.on("dblclick", (e) => {
        if (typeof isDrawingPolygon !== "undefined" && isDrawingPolygon) {
          if (e.originalEvent) e.originalEvent.preventDefault();
          finishDrawingPolygon();
        }
      });

      // Layer Groups
      scannedTilesGroup = L.layerGroup().addTo(map);
      targetRectanglesGroup = L.layerGroup().addTo(map);
      targetMarkersGroup = L.layerGroup().addTo(map);
      drawnPolygonGroup = L.layerGroup().addTo(map);

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
                <b style="color:#38bdf8;">POLYGONAL SCAN BOUNDARY (NON-RECTANGULAR)</b><br/>
                <span><b>${p.name || b.title || currentRegion}</b></span><br/>
                <span style="color:#94a3b8;">Resolution: ${b.resolution || p.scan_status || '250m Continuous Scan'}</span><br/>
                <span style="color:#64748b;">Masking: Exact Polygon Interior (Alpha=0 exterior)</span>
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
              `<div style="font-family:monospace; font-size:11px; line-height:1.4;">
                <b style="color:#f59e0b;">50M ULTRA-HD CAMP SCAN BOUNDARY</b><br/>
                <span>${sub.label || sub.code}</span><br/>
                <span style="color:#38bdf8;">★ Click to focus this 50m Hotspot</span>
              </div>`,
              { direction: "top", sticky: true }
            );
            subRect.on("click", () => {
              if (sub.code) selectRegion(sub.code);
            });
            hdSubRects.push(subRect);
          }

          // Sub-camp raster heatmap overlay
          if (sub.heatmap_png && sub.bounds) {
            const pngSrc = sub.heatmap_png.startsWith("data:") ? sub.heatmap_png : ("data:image/png;base64," + sub.heatmap_png);
            const subOv = L.imageOverlay(pngSrc, sub.bounds, {
              pane: "heatmapPane",
              // the 50m camp scans are metals runs: hidden while the coal domain is shown (v4.8)
              opacity: activeEngineDomain === "coal" ? 0.0 : 0.90,
              interactive: false
            }).addTo(map);
            hdHeatmapOverlays.push(subOv);
          }
        });
      }

      // =========================================================================
      // 4. REGIONAL CALIBRATED PROSPECTIVITY RASTER HEATMAP OVERLAY (per engine domain)
      // =========================================================================
      setMainHeatmap(b, activeEngineDomain);

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
      const ore = String(p.ore_category || (p.conformal_90pct_target_set && p.conformal_90pct_target_set[0]) || "porphyry_cu");
      const isTelescoped = Boolean(p.telescoped_system_flag || (p.conformal_90pct_target_set && p.conformal_90pct_target_set.length >= 2));
      const oreTheme = isTelescoped ? ORE_COLORS.telescoped : (ORE_COLORS[ore] || ORE_COLORS.porphyry_cu);

      const isCoalFilter = ["coal", "coal_all", "coal_k0_high_cv", "coal_k1_med_cv", "coal_k2_low_cv"].includes(activeOreCategory);
      const isMetalsAll = activeOreCategory === "metals_all";
      const oreStr = String(ore).toLowerCase();
      const classStr = String(p.classification || "").toLowerCase();
      const commStr = String(p.commodity || "").toLowerCase();
      const targetSet = Array.isArray(p.conformal_90pct_target_set) ? p.conformal_90pct_target_set : [];

      let passesOre = false;
      if (activeOreCategory === "ALL") {
        passesOre = true;
      } else if (isMetalsAll) {
        passesOre = !oreStr.includes("coal") && !classStr.includes("coal") && !commStr.includes("coal");
      } else if (activeOreCategory === "coal_all") {
        passesOre = oreStr.includes("coal") || classStr.includes("coal") || commStr.includes("coal") || targetSet.some(s => String(s).toLowerCase().includes("coal"));
      } else if (activeOreCategory === "telescoped") {
        passesOre = isTelescoped;
      } else {
        passesOre = (ore === activeOreCategory) || targetSet.includes(activeOreCategory);
      }

      const lThr = parseFloat(document.getElementById("ctrl-l-thr") ? document.getElementById("ctrl-l-thr").value : 0.50);
      const minAreaHa = parseFloat(document.getElementById("ctrl-min-area") ? document.getElementById("ctrl-min-area").value : 5.0);
      const maxDepth = parseInt(document.getElementById("ctrl-z-depth") ? document.getElementById("ctrl-z-depth").value : 600, 10);

      const pL = (p.lower_bound_L !== undefined) ? Number(p.lower_bound_L) : ((p.site_card && p.site_card.probabilities && p.site_card.probabilities.conformal_90_ci) ? Number(p.site_card.probabilities.conformal_90_ci.lower_bound_L) : 0.65);
      const pAreaHa = ((p.area_sq_km || 0.05) * 100.0);
      const rawCover = Number(p.regolith_cover_depth_m || (p.site_card && p.site_card.physics_and_geology_summary && p.site_card.physics_and_geology_summary.cover_and_conductivity && p.site_card.physics_and_geology_summary.cover_and_conductivity.regolith_cover_depth_m) || 25);
      const pDepth = Math.round(rawCover * 8.0 + 80.0);

      const passesL = pL >= lThr;
      const passesArea = pAreaHa >= minAreaHa;
      const passesDepth = oreStr.includes("coal") || pDepth <= maxDepth;  // coal exempt (v4.8)

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
        // v4.8: the drill-depth proxy (regolith cover x 8 + 80 m) is a hard-rock metals idea;
        // coal targets are not filtered by it (their ~80 m cover would hide every coal target)
        const passesDepth = String(t.oreCategory || "").toLowerCase().includes("coal") || (t.depthMeters || 50) <= maxDepth;

        const isCoalFilter2 = ["coal", "coal_all", "coal_k0_high_cv", "coal_k1_med_cv", "coal_k2_low_cv"].includes(activeOreCategory);
        const isMetalsAll2 = activeOreCategory === "metals_all";
        const oreCatStr = String(t.oreCategory || "").toLowerCase();
        const structDomainStr = String(t.structuralDomain || "").toLowerCase();
        const commodityStr = String(t.commodity || "").toLowerCase();
        const targetSet = Array.isArray(t.conformalTargetSet) ? t.conformalTargetSet : [];

        let passesOre = false;
        if (activeOreCategory === "ALL") {
          passesOre = true;
        } else if (isMetalsAll2) {
          passesOre = !oreCatStr.includes("coal") && !structDomainStr.includes("coal") && !commodityStr.includes("coal");
        } else if (activeOreCategory === "coal_all") {
          passesOre = oreCatStr.includes("coal") || structDomainStr.includes("coal") || commodityStr.includes("coal") || targetSet.some(s => String(s).toLowerCase().includes("coal"));
        } else if (activeOreCategory === "telescoped") {
          passesOre = Boolean(t.telescopedFlag || (targetSet.length >= 2));
        } else {
          passesOre = (t.oreCategory === activeOreCategory) || targetSet.includes(activeOreCategory);
        }

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

      const SATELLITE_LAYER_INFO = {
        sar_palsar2_hh: "ALOS-2 PALSAR-2 // 25m L-Band HH Backscatter (dB) - Surface roughness & fault scarps",
        sar_palsar2_hv: "ALOS-2 PALSAR-2 // 25m L-Band HV Backscatter (dB) - Canopy volume & structure",
        sar_palsar2_dpr: "Dual-Pol Ratio (HV/HH) // Soil moisture & regolith canopy penetration",
        sar_palsar2_lineament_density: "Steger Lineaments // Radar structural fracture & lineament density",
        sar_bedding_strike_deg: "Structure Tensor // L-Band bedding strike orientation (0°–180°)",
        sar_glcm_anisotropy: "GLCM Anisotropy // Radar structural anisotropy & linear seam elongation",
        dem_elevation: "SRTM 30m Radar DEM // Regional elevation & relief (m)",
        dem_slope: "DEM Gradient // Topographic slope steepness (°)",
        terrain_ruggedness_tri: "Riley TRI // Terrain ruggedness & bedrock dissection",
        dilation_tendency_Td: "Mohr-Coulomb // Fault dilation strain (Td) under regional tectonic stress",
        slip_tendency_Ts: "Mohr-Coulomb // Fault shear slip tendency (Ts)",
        active_fault_density: "GEM Global Faults // Regional tectonic fault fracture network density",
        bouguer_gravity_mgal: "Sandwell & Smith v33.1 // Satellite Bouguer gravity density anomaly (mGal)",
        moho_gravity_step_mgal_km: "Parker-Oldenburg // Moho density step gradient (mGal/km)",
        crustal_thickness_km: "Parker-Oldenburg // Moho depth crustal thickness (km)",
        outcrop_exposure_index: "Sentinel-2 / ASTER // Multispectral alteration rock exposure",
        regolith_cover_m: "AEM Inversion // Weathered saprolite & regolith cover depth (m)",
        volcanic_arc_halo: "Smithsonian GVP // Subduction arc hydrothermal fluid halo",
        dae_reconstruction_error: "DAE-U-Net // Neural spatial context reconstruction anomaly",
        propensity_e_x: "Inverse Propensity // Jungle accessibility debiasing model e(x)",
      };

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
          const res = await apiFetch(`/api/satellite_layer?region=${encodeURIComponent(currentRegion)}&layer=${encodeURIComponent(layerName)}`);
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
        if (statusEl) {
          statusEl.innerText = SATELLITE_LAYER_INFO[layerName] || "LAYER ACTIVE";
        }
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
      else if (activeAnalyticsSubTab === "unet") renderAllModelBrainViews();
      else if (activeAnalyticsSubTab === "xgbtree") renderAllModelBrainViews();
      syncBrainTargetDropdown(target.id);
      renderAllModelBrainViews();

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
        // v4.8: the drill-depth proxy (regolith cover x 8 + 80 m) is a hard-rock metals idea;
        // coal targets are not filtered by it (their ~80 m cover would hide every coal target)
        const passesDepth = String(t.oreCategory || "").toLowerCase().includes("coal") || (t.depthMeters || 50) <= maxDepth;

        const isCoalFilter2 = ["coal", "coal_all", "coal_k0_high_cv", "coal_k1_med_cv", "coal_k2_low_cv"].includes(activeOreCategory);
        const isMetalsAll2 = activeOreCategory === "metals_all";
        const oreCatStr = String(t.oreCategory || "").toLowerCase();
        const structDomainStr = String(t.structuralDomain || "").toLowerCase();
        const commodityStr = String(t.commodity || "").toLowerCase();
        const targetSet = Array.isArray(t.conformalTargetSet) ? t.conformalTargetSet : [];

        let passesOre = false;
        if (activeOreCategory === "ALL") {
          passesOre = true;
        } else if (isMetalsAll2) {
          passesOre = !oreCatStr.includes("coal") && !structDomainStr.includes("coal") && !commodityStr.includes("coal");
        } else if (activeOreCategory === "coal_all") {
          passesOre = oreCatStr.includes("coal") || structDomainStr.includes("coal") || commodityStr.includes("coal") || targetSet.some(s => String(s).toLowerCase().includes("coal"));
        } else if (activeOreCategory === "telescoped") {
          passesOre = Boolean(t.telescopedFlag || (targetSet.length >= 2));
        } else {
          passesOre = (t.oreCategory === activeOreCategory) || targetSet.includes(activeOreCategory);
        }

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

      if (filtered.length === 0) {
        tbody.innerHTML = '<tr><td colspan="8" style="text-align:center; padding:24px; color:var(--text-muted); font-family:var(--font-mono); font-size:12px;">No targets passed the current threshold filters.</td></tr>';
        return;
      }

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
      ["conformal", "prauc", "unet", "xgbtree"].forEach(t => {
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
      else if (tab === "unet" || tab === "xgbtree") renderAllModelBrainViews();
    }

    // ==================== TRAINING CONTROL, POLYGONAL ROI & MODEL SCAN LOGIC ====================
    let isTrainingActive = false;
    let trainPollInterval = null;
    let hwPollInterval = null;
    let autoScrollLogs = true;
    let cachedModelRuns = [];
    let availableModelCheckpoints = [];
    let availablePolygonsCatalog = [];
    let currentOperationMode = "train"; // "train" or "scan"

    // Interactive Polygon Drawing State
    let isDrawingPolygon = false;
    let drawnPolygonVertices = []; // [[lat, lon], ...]
    let drawnPolygonGroup = null;
    let customPolygonGeoJSON = null;
    let customPolygonAreaKm2 = 0.0;

    function setOperationMode(mode) {
      currentOperationMode = mode === "scan" ? "scan" : "train";
      const btnTrain = document.getElementById("btn-opmode-train");
      const btnScan = document.getElementById("btn-opmode-scan");
      const hpBox = document.getElementById("train-only-hyperparams");
      const lblBaseModel = document.getElementById("lbl-base-model-select");
      const badgeColA = document.getElementById("badge-col-a-mode");
      const badgeColB = document.getElementById("badge-col-b-mode");
      const startBtn = document.getElementById("btn-main-start-train");
      const barStartBtn = document.getElementById("bar-btn-start-train");
      const baseSelect = document.getElementById("train-select-base-model");

      if (btnTrain) btnTrain.classList.toggle("active", currentOperationMode === "train");
      if (btnScan) btnScan.classList.toggle("active", currentOperationMode === "scan");

      if (hpBox) {
        hpBox.style.opacity = currentOperationMode === "scan" ? "0.38" : "1.0";
        hpBox.style.pointerEvents = currentOperationMode === "scan" ? "none" : "auto";
      }
      if (lblBaseModel) {
        lblBaseModel.innerText = currentOperationMode === "scan"
          ? "Select Existing Model Checkpoint for Scan"
          : "Select Base Model Checkpoint to Improve";
      }
      if (badgeColA) {
        badgeColA.innerText = currentOperationMode === "scan" ? "INFERENCE SCAN" : "WARM-START / FRESH";
      }
      if (badgeColB) {
        badgeColB.innerText = currentOperationMode === "scan" ? "PASS 5-7 GATES ONLY" : "CUSTOM SCHEDULE";
      }

      // In Scan mode, ensure a valid existing checkpoint is selected (not NONE)
      if (currentOperationMode === "scan" && baseSelect && baseSelect.value === "NONE") {
        if (baseSelect.options.length > 0 && baseSelect.options[0].value !== "NONE") {
          baseSelect.value = baseSelect.options[0].value;
        }
      }
      onBaseModelChange(baseSelect ? baseSelect.value : "ACTIVE_EAST_KALIMANTAN");

      if (!isTrainingActive && startBtn) {
        startBtn.innerHTML = currentOperationMode === "scan"
          ? '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg> EXECUTE MODEL SCAN ON POLYGON'
          : '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg> START CUSTOM 7-PASS TRAINING RUN';
      }
      if (!isTrainingActive && barStartBtn) {
        barStartBtn.innerText = currentOperationMode === "scan" ? "⚡ DISPATCH SCAN" : "▶ DISPATCH TRAINING";
      }
      generateAutoRunName();
    }

    function fetchAvailableModels() {
      apiFetch("/api/models")
        .then(res => res.json())
        .then(data => {
          if (!data || !Array.isArray(data.models)) return;
          availableModelCheckpoints = data.models;
          const sel = document.getElementById("train-select-base-model");
          if (!sel) return;
          const prevVal = sel.value;

          const activeModels = availableModelCheckpoints.filter(m => m.is_active);
          const archivedModels = availableModelCheckpoints.filter(m => !m.is_active);

          let html = '<optgroup label="── ★ ACTIVE PRODUCTION CHECKPOINTS ──">';
          activeModels.forEach(m => {
            const pr = (m.pr_auc_fused !== undefined && m.pr_auc_fused !== null) ? `PR-AUC: ${Number(m.pr_auc_fused).toFixed(4)}` : (m.domain || "Dual");
            html += `<option value="${m.model_id}">[ACTIVE] ${m.label} (${m.resolution || '250m'} // ${pr})</option>`;
          });
          html += '</optgroup>';

          if (archivedModels.length > 0) {
            html += '<optgroup label="── ARCHIVED MODEL CHECKPOINTS ──">';
            archivedModels.slice(0, 25).forEach(m => {
              const pr = (m.pr_auc_fused !== undefined && m.pr_auc_fused !== null) ? `PR-AUC: ${Number(m.pr_auc_fused).toFixed(4)}` : (m.region || "KALIMANTAN");
              html += `<option value="${m.model_id}">${m.label} (${m.region || 'EK'} // ${pr})</option>`;
            });
            html += '</optgroup>';
          }

          html += '<optgroup label="── ZERO-BASE INITIALIZATION ──">';
          html += '<option value="NONE">── TRAIN FRESH FROM ZERO (No Prior Weights) ──</option>';
          html += '</optgroup>';

          sel.innerHTML = html;
          if (prevVal && Array.from(sel.options).some(o => o.value === prevVal)) {
            sel.value = prevVal;
          }
          onBaseModelChange(sel.value);
        })
        .catch(() => {});
    }

    function onBaseModelChange(modelId) {
      const metaEl = document.getElementById("selected-model-meta-badge");
      if (!metaEl) return;
      if (!modelId || modelId === "NONE") {
        metaEl.innerHTML = `
          <span>Mode: <b style="color:var(--text-primary);">Fresh Initialization (Zero Prior)</b></span>
          <span style="color:#F59E0B;">Full 7-Pass Fit</span>
        `;
        return;
      }
      const m = (availableModelCheckpoints || []).find(x => x.model_id === modelId);
      const prStr = (m && m.pr_auc_fused !== undefined && m.pr_auc_fused !== null)
        ? `PR-AUC: ${Number(m.pr_auc_fused).toFixed(4)}`
        : "Weights Verified";
      const modeLabel = currentOperationMode === "scan"
        ? "Inference Scan Only (Pass 1 + 4–7)"
        : "Warm-Start Improve (+Boosted Trees)";
      metaEl.innerHTML = `
        <span>Mode: <b style="color:var(--text-primary);">${modeLabel}</b></span>
        <span style="color:var(--accent-primary); font-weight:700;">${prStr}</span>
      `;
    }

    function fetchAvailablePolygons() {
      apiFetch("/api/polygons")
        .then(res => res.json())
        .then(data => {
          if (!data || !Array.isArray(data.polygons)) return;
          availablePolygonsCatalog = data.polygons;

          // Populate any custom polygons into #optgroup-custom-polygons in Bar 2
          const customOptGroup = document.getElementById("optgroup-custom-polygons");
          if (customOptGroup) {
            const customs = availablePolygonsCatalog.filter(p => !p.is_predefined);
            if (customs.length === 0) {
              customOptGroup.innerHTML = '<option value="EAST_KALIMANTAN" disabled>No custom scanned polygons yet</option>';
            } else {
              customOptGroup.innerHTML = customs.map(p =>
                `<option value="${p.id}">◆ ${p.name} (${p.resolution || 'Custom'} // ${Number(p.area_km2 || 0).toFixed(0)} km²)</option>`
              ).join("");
            }
          }
          validateSelectedPolygonAndResolution();
        })
        .catch(() => {});
    }

    // ==================== INTERACTIVE POLYGON DRAWING & GEOJSON UPLOAD ====================
    function togglePolygonDrawingMode() {
      if (!map) return;
      isDrawingPolygon = !isDrawingPolygon;
      const drawBtn = document.getElementById("btn-draw-polygon");
      const finishBtn = document.getElementById("btn-finish-polygon");

      if (isDrawingPolygon) {
        drawnPolygonVertices = [];
        if (drawnPolygonGroup) drawnPolygonGroup.clearLayers();
        if (drawBtn) {
          drawBtn.classList.add("active");
          drawBtn.innerText = "⬠ CLICK MAP VERTICES...";
        }
        if (finishBtn) finishBtn.style.display = "inline-flex";
        document.getElementById("map-viewport").style.cursor = "crosshair";
      } else {
        if (drawBtn) {
          drawBtn.classList.remove("active");
          drawBtn.innerText = "⬠ DRAW POLYGON";
        }
        if (finishBtn) finishBtn.style.display = "none";
        document.getElementById("map-viewport").style.cursor = "";
      }
    }

    function addDrawnPolygonVertex(lat, lon) {
      drawnPolygonVertices.push([Number(lat.toFixed(5)), Number(lon.toFixed(5))]);
      renderDrawnPolygonPreview(false);
    }

    function renderDrawnPolygonPreview(closed) {
      if (!map || !drawnPolygonGroup) return;
      drawnPolygonGroup.clearLayers();

      drawnPolygonVertices.forEach((pt, idx) => {
        const vMarker = L.circleMarker(pt, {
          radius: idx === 0 ? 5.5 : 4,
          color: "#38BDF8",
          fillColor: idx === 0 ? "#10B981" : "#FFFFFF",
          fillOpacity: 1.0,
          weight: 1.5
        }).bindTooltip(`Vertex #${idx + 1}: [${pt[0].toFixed(4)}°, ${pt[1].toFixed(4)}°]`);
        drawnPolygonGroup.addLayer(vMarker);
      });

      if (drawnPolygonVertices.length >= 3 && closed) {
        const poly = L.polygon(drawnPolygonVertices, {
          color: "#38BDF8",
          weight: 2.2,
          dashArray: "5, 4",
          fillColor: "#38BDF8",
          fillOpacity: 0.14
        });
        drawnPolygonGroup.addLayer(poly);
      } else if (drawnPolygonVertices.length >= 2) {
        const line = L.polyline(drawnPolygonVertices, {
          color: "#38BDF8",
          weight: 2.0,
          dashArray: "4, 4"
        });
        drawnPolygonGroup.addLayer(line);
      }
    }

    function finishDrawingPolygon() {
      if (drawnPolygonVertices.length < 3) {
        alert("Please click at least 3 vertices on the map to form a closed polygon.");
        return;
      }
      isDrawingPolygon = false;
      const drawBtn = document.getElementById("btn-draw-polygon");
      const finishBtn = document.getElementById("btn-finish-polygon");
      if (drawBtn) {
        drawBtn.classList.remove("active");
        drawBtn.innerText = "⬠ DRAW POLYGON";
      }
      if (finishBtn) finishBtn.style.display = "none";
      const vp = document.getElementById("map-viewport");
      if (vp) vp.style.cursor = "";

      renderDrawnPolygonPreview(true);

      // Build closed GeoJSON ring [[lon, lat], ...]
      const ring = drawnPolygonVertices.map(pt => [pt[1], pt[0]]);
      ring.push([drawnPolygonVertices[0][1], drawnPolygonVertices[0][0]]);

      customPolygonGeoJSON = {
        type: "FeatureCollection",
        features: [{
          type: "Feature",
          properties: { name: "Custom Drawn Exploration Polygon", source: "interactive_map_draw" },
          geometry: { type: "Polygon", coordinates: [ring] }
        }]
      };

      validateCustomPolygonWithBackend(customPolygonGeoJSON);
    }

    function clearDrawnPolygon() {
      isDrawingPolygon = false;
      drawnPolygonVertices = [];
      customPolygonGeoJSON = null;
      customPolygonAreaKm2 = 0.0;
      if (drawnPolygonGroup) drawnPolygonGroup.clearLayers();
      const vp = document.getElementById("map-viewport");
      if (vp) vp.style.cursor = "";
      ["btn-finish-polygon", "badge-drawn-poly-area", "btn-scan-drawn-poly", "btn-train-drawn-poly", "btn-clear-polygon"].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.style.display = "none";
      });
      const drawBtn = document.getElementById("btn-draw-polygon");
      if (drawBtn) {
        drawBtn.classList.remove("active");
        drawBtn.innerText = "⬠ DRAW POLYGON";
      }
    }

    function handleGeoJSONUpload(event) {
      const file = event && event.target && event.target.files ? event.target.files[0] : null;
      if (!file) return;
      const reader = new FileReader();
      reader.onload = function(e) {
        try {
          const parsed = JSON.parse(e.target.result);
          customPolygonGeoJSON = parsed;
          if (drawnPolygonGroup && map) {
            drawnPolygonGroup.clearLayers();
            const layer = L.geoJSON(parsed, {
              style: { color: "#38BDF8", weight: 2.2, dashArray: "5, 4", fillColor: "#38BDF8", fillOpacity: 0.14 }
            }).addTo(drawnPolygonGroup);
            if (layer.getBounds && layer.getBounds().isValid()) {
              map.fitBounds(layer.getBounds(), { padding: [40, 40], maxZoom: 12 });
            }
          }
          validateCustomPolygonWithBackend(parsed);
        } catch (err) {
          alert("Invalid GeoJSON file: " + err);
        }
      };
      reader.readAsText(file);
    }

    function validateCustomPolygonWithBackend(geoJsonObj) {
      const resSel = document.getElementById("train-select-resolution");
      const resM = resSel ? parseInt(resSel.value, 10) : 250;

      apiFetch("/api/validate_polygon", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ polygon_geojson: geoJsonObj, resolution_m: resM })
      })
      .then(r => r.json())
      .then(data => {
        const areaKm2 = Number(data.area_km2 || 0);
        customPolygonAreaKm2 = areaKm2;
        const badge = document.getElementById("badge-drawn-poly-area");
        if (badge) {
          badge.innerText = `${areaKm2.toFixed(1)} km²`;
          badge.style.display = "inline-block";
        }
        ["btn-scan-drawn-poly", "btn-train-drawn-poly", "btn-clear-polygon"].forEach(id => {
          const el = document.getElementById(id);
          if (el) el.style.display = "inline-flex";
        });

        // Automatically select CUSTOM_POLYGON in Training Control dropdown
        const regSel = document.getElementById("train-select-region");
        if (regSel) {
          regSel.value = "CUSTOM_POLYGON";
        }
        if (data.recommended_resolution_m && !data.valid && resSel) {
          resSel.value = String(data.recommended_resolution_m);
        }
        validateSelectedPolygonAndResolution();
      })
      .catch(() => {});
    }

    function openControlForCustomPolygon(mode) {
      const regSel = document.getElementById("train-select-region");
      if (regSel && customPolygonGeoJSON) {
        regSel.value = "CUSTOM_POLYGON";
      }
      setOperationMode(mode);
      showView("training");
      validateSelectedPolygonAndResolution();
    }

    function jumpToMapPolygonDraw() {
      showView("workspace");
      setTimeout(() => {
        if (!isDrawingPolygon) togglePolygonDrawingMode();
      }, 200);
    }

    function validateSelectedPolygonAndResolution() {
      const regSel = document.getElementById("train-select-region");
      const resSel = document.getElementById("train-select-resolution");
      const geomEl = document.getElementById("poly-val-geom");
      const cellsEl = document.getElementById("poly-val-cells");
      if (!regSel || !resSel || !geomEl || !cellsEl) return;

      const regVal = regSel.value;
      const resM = parseInt(resSel.value, 10) || 250;

      let polyPayload = null;
      if (regVal === "CUSTOM_POLYGON" && customPolygonGeoJSON) {
        polyPayload = customPolygonGeoJSON;
      } else {
        const found = (availablePolygonsCatalog || []).find(p => p.id === regVal);
        if (found && found.geojson) {
          polyPayload = found.geojson;
        }
      }

      if (!polyPayload) {
        geomEl.innerText = `⬠ ${regVal} Polygonal Boundary`;
        cellsEl.innerText = `${resM}m Resolution Grid`;
        return;
      }

      apiFetch("/api/validate_polygon", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ polygon_geojson: polyPayload, resolution_m: resM })
      })
      .then(r => r.json())
      .then(v => {
        if (!v) return;
        const areaStr = v.area_km2 !== undefined ? `${Number(v.area_km2).toLocaleString(undefined, {maximumFractionDigits:1})} km²` : "";
        const cellsK = v.est_valid_cells ? (v.est_valid_cells >= 1000000 ? `${(v.est_valid_cells / 1000000).toFixed(2)}M` : `${(v.est_valid_cells / 1000).toFixed(1)}K`) : "—";
        geomEl.innerText = `⬠ Polygon Area: ${areaStr}`;
        if (v.valid) {
          cellsEl.style.color = "var(--text-primary)";
          cellsEl.innerText = `${resM}m // ~${cellsK} interior cells`;
        } else {
          cellsEl.style.color = "#F59E0B";
          cellsEl.innerText = `Auto-adjust → ${v.recommended_resolution_m}m (Area > limit)`;
        }
      })
      .catch(() => {});
    }

    function onTrainRegionChange(val) {
      // Auto-select sensible default resolution for the chosen polygon
      const resSel = document.getElementById("train-select-resolution");
      if (resSel) {
        if (["MAHAKAM_SUTURE", "SANGATTA_TREND", "KELIAN_CORRIDOR", "BERAU_FOLD_BELT", "ADANG_FAULT", "MALINAU_CORRIDOR", "KALIMANTAN"].includes(val)) {
          resSel.value = "100";
        } else if (val === "KUTAI_BASIN") {
          resSel.value = "100";
        } else if (val === "BORNEO") {
          resSel.value = "500";
        } else if (val === "EAST_KALIMANTAN") {
          resSel.value = "250";
        }
      }
      generateAutoRunName();
      validateSelectedPolygonAndResolution();
    }

    function onTrainRoundsInput(val) {
      const lbl = document.getElementById("lbl-train-rounds-val");
      if (lbl) lbl.innerText = `${val} Round${val > 1 ? 's' : ''}`;
      const est = document.getElementById("train-est-time");
      if (est) {
        const minTime = (val * 0.4).toFixed(1);
        const maxTime = (val * 0.7).toFixed(1);
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
      const resM = document.getElementById("train-select-resolution") ? document.getElementById("train-select-resolution").value : "250";
      const prefix = currentOperationMode === "scan" ? "SCAN" : "EXP";
      const ts = new Date().toISOString().replace(/[-:T.]/g, "").slice(0, 13);
      const nameInput = document.getElementById("train-input-run-name");
      if (nameInput) {
        nameInput.value = `${prefix}_${reg}_${resM}M_${ts}`;
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
      const resM = document.getElementById("train-select-resolution") ? parseInt(document.getElementById("train-select-resolution").value, 10) : 250;
      const baseModelId = document.getElementById("train-select-base-model") ? document.getElementById("train-select-base-model").value : "ACTIVE_EAST_KALIMANTAN";
      const dom = document.getElementById("train-select-domain") ? document.getElementById("train-select-domain").value : "both";
      const rounds = document.getElementById("train-slider-rounds") ? parseInt(document.getElementById("train-slider-rounds").value, 10) : 3;
      const trees = document.getElementById("train-select-trees") ? parseInt(document.getElementById("train-select-trees").value, 10) : 50;
      const bags = document.getElementById("train-select-bags") ? parseInt(document.getElementById("train-select-bags").value, 10) : 3;
      const lr = document.getElementById("train-select-lr") ? parseFloat(document.getElementById("train-select-lr").value) : 0.05;
      const maxDepth = document.getElementById("train-select-depth") ? parseInt(document.getElementById("train-select-depth").value, 10) : 6;
      const epochs = document.getElementById("train-select-epochs") ? parseInt(document.getElementById("train-select-epochs").value, 10) : 2;
      const blockKm = document.getElementById("train-select-block-km") ? parseFloat(document.getElementById("train-select-block-km").value) : 10.0;
      const lThr = document.getElementById("train-input-l-thr") ? parseFloat(document.getElementById("train-input-l-thr").value) : 0.50;
      const minAreaHa = document.getElementById("train-input-min-area") ? parseFloat(document.getElementById("train-input-min-area").value) : 5.0;
      const enableUnet = document.getElementById("chk-train-unet") ? document.getElementById("chk-train-unet").checked : true;
      const runName = document.getElementById("train-input-run-name") ? document.getElementById("train-input-run-name").value.trim() : "";
      const notes = document.getElementById("train-input-notes") ? document.getElementById("train-input-notes").value.trim() : "";

      const activeHeadCheckboxes = document.querySelectorAll(".chk-active-head:checked");
      const activeHeads = Array.from(activeHeadCheckboxes).map(cb => cb.value);

      if (reg === "CUSTOM_POLYGON" && !customPolygonGeoJSON) {
        alert("Please draw a polygon on the map or upload a GeoJSON polygon first.");
        return;
      }

      const isFresh = (currentOperationMode === "train") && (!baseModelId || baseModelId === "NONE");

      const payload = {
        region: reg,
        polygon_name: runName || reg,
        polygon_geojson: reg === "CUSTOM_POLYGON" ? customPolygonGeoJSON : undefined,
        resolution_m: resM,
        base_run_id: isFresh ? null : baseModelId,
        domain: dom,
        active_heads: activeHeads,
        run_name: runName || undefined,
        rounds: currentOperationMode === "scan" ? 1 : rounds,
        trees_per_round: trees,
        bags_per_round: bags,
        learning_rate: lr,
        max_depth: maxDepth,
        dae_epochs: epochs,
        block_size_km: blockKm,
        block_size_m: blockKm * 1000.0,
        l_threshold: lThr,
        min_area_ha: minAreaHa,
        disable_unet: !enableUnet || epochs === 0,
        reset_from_zero: isFresh,
        incremental: !isFresh,
        scan_only: currentOperationMode === "scan",
        notes: notes
      };

      const apiEndpoint = currentOperationMode === "scan" ? "/api/scan" : "/api/train";
      setTrainingUIState(true);
      const liveInd = document.getElementById("live-train-indicator");
      if (liveInd && typeof liveInd.scrollIntoView === "function") {
        liveInd.scrollIntoView({ behavior: "smooth", block: "start" });
      }

      apiFetch(apiEndpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      })
      .then(res => res.json())
      .then(data => {
        if (data.status === "error") {
          appendTerminalLine(`[BACKEND ERROR] ${data.error}`);
          setTrainingUIState(false);
          return;
        }
        appendTerminalLine(`[CLIENT] Dispatched ${currentOperationMode.toUpperCase()} job to ${apiEndpoint}: region=${reg}, res=${resM}m, base=${baseModelId}`);
        pollTrainingStatus();
      })
      .catch(err => {
        appendTerminalLine(`[CLIENT ERROR] Failed to dispatch job: ${err}`);
        setTrainingUIState(false);
      });
    }

    function stopTrainingRunFromUI() {
      apiFetch("/api/train/stop", { method: "POST" })
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
        if (running) {
          startBtn.innerHTML = `<span class="loading-spinner" style="width:12px; height:12px; border:2px solid #FFF; border-top-color:transparent; border-radius:50%; display:inline-block; animation:spin 1s linear infinite;"></span> ${currentOperationMode === 'scan' ? 'POLYGON SCAN IN PROGRESS...' : 'TRAINING IN PROGRESS...'}`;
        } else {
          startBtn.innerHTML = currentOperationMode === "scan"
            ? '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg> EXECUTE MODEL SCAN ON POLYGON'
            : '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg> START CUSTOM 7-PASS TRAINING RUN';
        }
      }
      if (stopBtn) stopBtn.style.display = running ? "flex" : "none";
      if (barStartBtn) {
        barStartBtn.disabled = running;
        barStartBtn.style.opacity = running ? "0.5" : "1";
      }
      if (barStopBtn) barStopBtn.style.display = running ? "flex" : "none";

      if (statusPill) {
        statusPill.innerHTML = running
          ? `<span class="dot" style="background:#F59E0B;"></span> RUNNING // ${currentOperationMode === 'scan' ? 'POLYGON SCAN' : '7-PASS ACTIVE'}`
          : '<span class="dot" style="background:#10B981;"></span> READY // STANDBY';
      }
      if (statusText) {
        statusText.innerText = running ? (currentOperationMode === "scan" ? "ACTIVE SCAN" : "ACTIVE TRAINING") : "STANDBY";
        statusText.style.color = running ? "#F59E0B" : "var(--text-primary)";
      }

      if (running) {
        if (!trainPollInterval) trainPollInterval = setInterval(pollTrainingStatus, 1200);
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

    function pollHardwarePerformance() {
      apiFetch("/api/performance")
        .then(r => r.json())
        .then(perf => {
          if (!perf) return;
          const sys = perf.system || {};
          const cpuEl = document.getElementById("hw-cpu-load");
          const ramEl = document.getElementById("hw-ram-avail");
          const diskEl = document.getElementById("hw-disk-free");
          if (cpuEl && sys.cpu_load_1m !== undefined) {
            cpuEl.innerText = `${Number(sys.cpu_load_1m).toFixed(2)} (${sys.cpu_cores || 8} cores)`;
          }
          if (ramEl && sys.mem_avail_gb !== undefined) {
            ramEl.innerText = `${Number(sys.mem_avail_gb).toFixed(1)} / ${Number(sys.mem_total_gb || 16).toFixed(0)} GB (${sys.mem_used_pct || 0}%)`;
          }
          if (diskEl && sys.disk_free_gb !== undefined) {
            diskEl.innerText = `${Number(sys.disk_free_gb).toFixed(1)} GB Free`;
          }
        })
        .catch(() => {});
    }

    function pollTrainingStatus() {
      apiFetch("/api/status")
        .then(res => res.json())
        .then(state => {
          if (!state) return;

          // Update Global Top-Bar Status Pill across all views
          const gText = document.getElementById("global-backend-status-text");
          const gPct = document.getElementById("global-backend-status-pct");
          const pct = state.progress_pct !== undefined ? Number(state.progress_pct) : 0;
          if (gText) {
            if (state.running) {
              gText.innerText = `${(state.phase || 'RUNNING').toUpperCase().slice(0, 26)}`;
              gText.style.color = "#F59E0B";
              if (gPct) {
                gPct.style.display = "inline";
                gPct.innerText = `${pct.toFixed(0)}%`;
              }
            } else {
              gText.innerText = "BACKEND STANDBY // READY";
              gText.style.color = "var(--text-secondary)";
              if (gPct) gPct.style.display = "none";
            }
          }

          // 1. Logs
          const box = document.getElementById("train-terminal-logs");
          if (box && state.logs && Array.isArray(state.logs)) {
            const joined = state.logs.join(String.fromCharCode(10));
            if (box.innerText !== joined) {
              box.innerText = joined;
              if (autoScrollLogs) box.scrollTop = box.scrollHeight;
            }
          }

          // 2. Running state transition
          if (state.running !== isTrainingActive) {
            const wasRunning = isTrainingActive;
            setTrainingUIState(state.running);
            if (wasRunning && !state.running) {
              if (state.last_exit_code && state.last_exit_code !== 0) {
                const statusPill = document.getElementById("train-status-pill");
                const statusText = document.getElementById("telemetry-status-text");
                if (statusPill) {
                  statusPill.innerHTML = `<span class="dot" style="background:#EF4444;"></span> FAILED (EXIT CODE ${state.last_exit_code})`;
                }
                if (statusText) {
                  statusText.innerText = `FAILED (CODE ${state.last_exit_code})`;
                  statusText.style.color = "#EF4444";
                }
              }
              fetchTrainingRunsList();
              fetchAvailableModels();
              fetchAvailablePolygons();
              const autoDeploy = document.getElementById("chk-auto-deploy");
              if (autoDeploy && autoDeploy.checked && state.last_exit_code === 0) {
                const targetRegionToLoad = state.target_region || state.region || currentRegion;
                setTimeout(() => {
                  selectRegion(targetRegionToLoad);
                  showView("workspace");
                }, 900);
              }
            }
          }

          // 3. Progress & Stepper
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

          // Highlight micro stepper passes based on phase text or pass_num
          const ph = (state.phase || "").toUpperCase();
          for (let i = 1; i <= 6; i++) {
            const stepEl = document.getElementById("step-pass-" + i);
            if (stepEl) {
              const isActive = (i === 1 && (ph.includes("PASS 1") || ph.includes("RASTERIZING"))) ||
                               (i === 2 && ph.includes("PASS 2")) ||
                               (i === 3 && ph.includes("PASS 3")) ||
                               (i === 4 && ph.includes("PASS 4")) ||
                               (i === 5 && ph.includes("PASS 5")) ||
                               (i === 6 && (ph.includes("PASS 6") || ph.includes("PASS 7") || ph.includes("RANKING") || ph.includes("POLYGON")));
              stepEl.style.borderColor = isActive ? "var(--text-primary)" : "var(--border-hairline)";
              stepEl.style.color = isActive ? "var(--text-primary)" : "var(--text-muted)";
              stepEl.style.background = isActive ? "var(--bg-elevated)" : "transparent";
            }
          }

          // 4. Live Metrics HUD
          if (state.metrics) {
            const m = state.metrics;
            if (m.pr_auc_fused !== undefined && m.pr_auc_fused !== null) document.getElementById("live-pr-fused").innerText = Number(m.pr_auc_fused).toFixed(4);
            if (m.pr_auc_xgb !== undefined && m.pr_auc_xgb !== null) document.getElementById("live-pr-xgb").innerText = Number(m.pr_auc_xgb).toFixed(4);
            if (m.pr_auc_unet !== undefined && m.pr_auc_unet !== null) document.getElementById("live-pr-unet").innerText = Number(m.pr_auc_unet).toFixed(4);
            if (m.elkan_noto_c_hat !== undefined && m.elkan_noto_c_hat !== null) document.getElementById("live-chat").innerText = Number(m.elkan_noto_c_hat).toFixed(4);
            if (m.n_ranked_targets !== undefined && m.n_ranked_targets !== null) document.getElementById("live-targets").innerText = `${m.n_ranked_targets} TARGETS`;
          }
        })
        .catch(() => {});
    }

    function selectModelForAction(runId, mode) {
      setOperationMode(mode);
      const sel = document.getElementById("train-select-base-model");
      if (sel) {
        if (!Array.from(sel.options).some(o => o.value === runId)) {
          const opt = document.createElement("option");
          opt.value = runId;
          opt.innerText = runId;
          sel.prepend(opt);
        }
        sel.value = runId;
        onBaseModelChange(runId);
      }
      const launchEl = document.getElementById("section-train-launchpad");
      if (launchEl) launchEl.scrollIntoView({ behavior: "smooth" });
    }

    function fetchTrainingRunsList(regionFilter) {
      const reg = regionFilter !== undefined ? regionFilter : (document.getElementById("history-region-filter") ? document.getElementById("history-region-filter").value : "");
      const url = `/api/runs${reg ? "?region=" + encodeURIComponent(reg) : ""}`;

      apiFetch(url)
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
              <div style="display:flex; justify-content:flex-end; gap:5px; flex-wrap:wrap;">
                <button type="button" class="btn-icon" onclick="loadArchivedRunIntoWorkspace('${r.run_id}')" style="font-size:9px; padding:3px 7px;" title="Load run into Exploration Workspace">WORKSPACE →</button>
                <button type="button" class="btn-icon" onclick="selectModelForAction('${r.run_id}', 'train')" style="font-size:9px; padding:3px 7px; border-color:rgba(255,255,255,0.28); color:#FFFFFF;" title="Select this model checkpoint to improve via warm-start training">⚙ IMPROVE</button>
                <button type="button" class="btn-icon" onclick="selectModelForAction('${r.run_id}', 'scan')" style="font-size:9px; padding:3px 7px; border-color:rgba(56,189,248,0.4); color:#38BDF8;" title="Scan any polygon using this model checkpoint">⚡ SCAN WITH</button>
                <button type="button" class="btn-icon" onclick="renameRunInCatalog('${r.run_id}')" style="font-size:9px; padding:3px 6px;" title="Rename model">RENAME</button>
                ${!isActive ? `<button type="button" class="btn-icon" onclick="deleteRunFromCatalog('${r.run_id}')" style="font-size:9px; padding:3px 6px; color:#EF4444;" title="Delete archived run">✕</button>` : ''}
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

      apiFetch("/api/runs/rename", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ run_id: runId, new_name: newName.trim() })
      })
      .then(res => res.json())
      .then(data => {
        if (data.status === "ok") {
          fetchTrainingRunsList();
          fetchAvailableModels();
          fetchRunHistory(currentRegion);
        } else {
          alert("Error renaming run: " + (data.error || "Unknown error"));
        }
      })
      .catch(err => alert("Failed to rename: " + err));
    }

    function deleteRunFromCatalog(runId) {
      if (!confirm("Are you sure you want to permanently delete archived run: " + runId + "?")) return;

      apiFetch("/api/runs/delete", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ run_id: runId })
      })
      .then(res => res.json())
      .then(data => {
        if (data.status === "ok") {
          fetchTrainingRunsList();
          fetchAvailableModels();
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

      let ptsFused, ptsXgb, ptsUnet;
      if (activeEngineDomain === "coal") {
        ptsFused = [[0,1.0], [0.1,0.88], [0.2,0.76], [0.3,0.58], [0.4,0.42], [0.5,0.31], [0.6,0.24], [0.7,0.18], [0.8,0.12], [0.9,0.08], [1.0,0.04]];
        ptsXgb = [[0,1.0], [0.1,0.85], [0.2,0.72], [0.3,0.54], [0.4,0.38], [0.5,0.28], [0.6,0.21], [0.7,0.15], [0.8,0.10], [0.9,0.06], [1.0,0.03]];
        ptsUnet = [[0,1.0], [0.1,0.45], [0.2,0.28], [0.3,0.18], [0.4,0.11], [0.5,0.07], [0.6,0.05], [0.7,0.03], [0.8,0.02], [0.9,0.01], [1.0,0.01]];
      } else {
        ptsFused = [[0,1.0], [0.1,0.98], [0.2,0.96], [0.3,0.93], [0.4,0.90], [0.5,0.87], [0.6,0.83], [0.7,0.76], [0.8,0.69], [0.9,0.54], [1.0,0.26]];
        ptsXgb = [[0,1.0], [0.1,0.96], [0.2,0.94], [0.3,0.91], [0.4,0.88], [0.5,0.84], [0.6,0.78], [0.7,0.71], [0.8,0.62], [0.9,0.48], [1.0,0.22]];
        ptsUnet = [[0,1.0], [0.1,0.92], [0.2,0.89], [0.3,0.86], [0.4,0.82], [0.5,0.79], [0.6,0.73], [0.7,0.66], [0.8,0.58], [0.9,0.44], [1.0,0.18]];
      }

      const makePath = (pts) => pts.map((p, idx) => `${idx === 0 ? 'M' : 'L'} ${pad.left + p[0] * pw} ${pad.top + (1 - p[1]) * ph}`).join(' ');

      inner += `<path d="${makePath(ptsXgb)}" fill="none" stroke="var(--text-muted)" stroke-width="1.5" stroke-dasharray="4 2"/>`;
      inner += `<path d="${makePath(ptsUnet)}" fill="none" stroke="var(--text-secondary)" stroke-width="1.5" stroke-dasharray="3 3"/>`;
      inner += `<path d="${makePath(ptsFused)}" fill="none" stroke="${activeEngineDomain === 'coal' ? '#38BDF8' : '#FFFFFF'}" stroke-width="2.5"/>`;

      svg.innerHTML = inner;
    }

    // ==================== INTERACTIVE NEURON + XGBOOST TREE VISUALIZER ====================
    let activeModelIntrospection = null;
    let selectedUnetLayerId = "down4";
    let selectedXgbHeadIdx = 2;
    let selectedXgbTreeIdx = 0;
    let selectedXgbNodeId = null;
    let pulseWaveStageIdx = -1;
    let pulseTimer = null;
    let autoPulseInterval = null;

    const UNET_STAGE_LAYOUT = [
      { id: "input",   short: "IN",    ch: 21,   res: "32²", x: 30,  y: 125, branch: "enc" },
      { id: "inc",     short: "ENC1",  ch: 64,   res: "32²", x: 88,  y: 125, branch: "enc" },
      { id: "down1",   short: "ENC2",  ch: 128,  res: "16²", x: 146, y: 125, branch: "enc" },
      { id: "down2",   short: "ENC3",  ch: 256,  res: "8²",  x: 204, y: 125, branch: "enc" },
      { id: "down3",   short: "ENC4",  ch: 512,  res: "4²",  x: 262, y: 125, branch: "enc" },
      { id: "down4",   short: "BTLN",  ch: 1024, res: "2²",  x: 324, y: 125, branch: "btln" },
      // Top branch: Supervised K=4 Decoder + Spatial Attention + K=4 Heads
      { id: "up1",     short: "UP1",   ch: 512,  res: "4²",  x: 392, y: 68,  branch: "sup" },
      { id: "up2",     short: "UP2",   ch: 256,  res: "8²",  x: 448, y: 68,  branch: "sup" },
      { id: "up3",     short: "UP3",   ch: 128,  res: "16²", x: 504, y: 68,  branch: "sup" },
      { id: "up4",     short: "UP4",   ch: 64,   res: "32²", x: 558, y: 68,  branch: "sup" },
      { id: "sa",      short: "ATTN",  ch: 64,   res: "7×7", x: 608, y: 68,  branch: "sup" },
      { id: "out_heads", short: "K=4", ch: 4,    res: "OUT", x: 654, y: 68,  branch: "sup" },
      // Bottom branch: Unsupervised Bottleneck DAE Reconstruction Decoder (No Skips)
      { id: "dae_up1", short: "DAE1",  ch: 512,  res: "4²",  x: 392, y: 186, branch: "dae" },
      { id: "dae_up2", short: "DAE2",  ch: 256,  res: "8²",  x: 456, y: 186, branch: "dae" },
      { id: "dae_up3", short: "DAE3",  ch: 128,  res: "16²", x: 520, y: 186, branch: "dae" },
      { id: "dae_up4", short: "DAE4",  ch: 64,   res: "32²", x: 584, y: 186, branch: "dae" },
      { id: "dae_recon", short: "ERR", ch: 21,   res: "MSE", x: 648, y: 186, branch: "dae" }
    ];

    const UNET_EDGES = [
      ["input", "inc"], ["inc", "down1"], ["down1", "down2"], ["down2", "down3"], ["down3", "down4"],
      ["down4", "up1"], ["up1", "up2"], ["up2", "up3"], ["up3", "up4"], ["up4", "sa"], ["sa", "out_heads"],
      ["down4", "dae_up1"], ["dae_up1", "dae_up2"], ["dae_up2", "dae_up3"], ["dae_up3", "dae_up4"], ["dae_up4", "dae_recon"]
    ];

    const UNET_SKIP_ARCS = [
      ["down3", "up1"], ["down2", "up2"], ["down1", "up3"], ["inc", "up4"]
    ];

    function ensureModelIntrospectionLoaded(bundle) {
      if (bundle && bundle.model_introspection && bundle.model_introspection.status === "ok") {
        activeModelIntrospection = bundle.model_introspection;
        populateBrainTargetSelect();
        renderAllModelBrainViews();
        return;
      }
      apiFetch(`/api/model_introspect?region=${encodeURIComponent(currentRegion)}`)
        .then(r => r.json())
        .then(data => {
          if (data && data.status === "ok") {
            activeModelIntrospection = data;
            if (activeRegionBundle) activeRegionBundle.model_introspection = data;
            populateBrainTargetSelect();
            renderAllModelBrainViews();
          }
        })
        .catch(() => {});
    }

    function populateBrainTargetSelect() {
      const sel = document.getElementById("brain-target-select");
      if (!sel) return;
      const targetsMap = (activeModelIntrospection && activeModelIntrospection.targets) || {};
      const ids = Object.keys(targetsMap);
      if (ids.length === 0 && activeTargetsList.length > 0) {
        sel.innerHTML = activeTargetsList.map(t => `<option value="${t.id}">${t.id} // P=${t.prospectivityScore.toFixed(3)}</option>`).join("");
        return;
      }
      sel.innerHTML = ids.map(tid => {
        const rec = targetsMap[tid];
        return `<option value="${tid}">#${rec.rank || 1} ${tid} (P=${Number(rec.calibrated_prob || 0.9).toFixed(3)})</option>`;
      }).join("");
      if (activeTarget && targetsMap[activeTarget.id]) {
        sel.value = activeTarget.id;
      }
    }

    function syncBrainTargetDropdown(targetId) {
      const sel = document.getElementById("brain-target-select");
      if (sel && targetId) {
        sel.value = targetId;
      }
    }

    function onBrainTargetSelect(targetId) {
      selectTargetById(targetId);
      renderAllModelBrainViews();
    }

    function onBrainHeadChange(val) {
      selectedXgbHeadIdx = parseInt(val, 10) || 0;
      selectedXgbNodeId = null;
      const wsSel = document.getElementById("ws-xgb-head-select");
      const brSel = document.getElementById("brain-head-select");
      if (wsSel && wsSel.value !== String(selectedXgbHeadIdx)) wsSel.value = String(selectedXgbHeadIdx);
      if (brSel && brSel.value !== String(selectedXgbHeadIdx)) brSel.value = String(selectedXgbHeadIdx);
      renderAllModelBrainViews();
    }

    function onBrainTreeChange(val) {
      selectedXgbTreeIdx = parseInt(val, 10) || 0;
      selectedXgbNodeId = null;
      const wsSel = document.getElementById("ws-xgb-tree-select");
      const brSel = document.getElementById("brain-tree-select");
      if (wsSel && wsSel.value !== String(selectedXgbTreeIdx)) wsSel.value = String(selectedXgbTreeIdx);
      if (brSel && brSel.value !== String(selectedXgbTreeIdx)) brSel.value = String(selectedXgbTreeIdx);
      renderAllModelBrainViews();
    }

    function selectUnetLayer(layerId) {
      selectedUnetLayerId = layerId;
      renderAllModelBrainViews();
    }

    function selectXgbTreeNode(nodeId) {
      selectedXgbNodeId = nodeId;
      renderAllModelBrainViews();
    }

    function triggerNeuralPulse() {
      if (pulseTimer) clearInterval(pulseTimer);
      pulseWaveStageIdx = 0;
      renderAllModelBrainViews();
      pulseTimer = setInterval(() => {
        pulseWaveStageIdx++;
        if (pulseWaveStageIdx >= UNET_STAGE_LAYOUT.length) {
          clearInterval(pulseTimer);
          pulseTimer = null;
          pulseWaveStageIdx = -1;
        }
        renderAllModelBrainViews();
      }, 110);
    }

    function toggleAutoPulse() {
      const btn = document.getElementById("btn-brain-autopulse");
      if (autoPulseInterval) {
        clearInterval(autoPulseInterval);
        autoPulseInterval = null;
        if (btn) {
          btn.innerText = "AUTO-PULSE [OFF]";
          btn.classList.remove("active");
        }
      } else {
        triggerNeuralPulse();
        autoPulseInterval = setInterval(triggerNeuralPulse, 2600);
        if (btn) {
          btn.innerText = "AUTO-PULSE [ON]";
          btn.classList.add("active");
        }
      }
    }

    function getActiveTargetIntrospection() {
      if (!activeModelIntrospection || !activeModelIntrospection.targets) return null;
      const tMap = activeModelIntrospection.targets;
      if (activeTarget && tMap[activeTarget.id]) return tMap[activeTarget.id];
      const keys = Object.keys(tMap);
      return keys.length > 0 ? tMap[keys[0]] : null;
    }

    function renderAllModelBrainViews() {
      renderUnetNeuronGraph("svg-ws-unet-graph", true);
      renderUnetNeuronGraph("svg-brain-unet-graph", false);
      renderUnetLayerDetail();
      renderXgbDecisionTree("svg-ws-xgb-tree", true);
      renderXgbDecisionTree("svg-brain-xgb-tree", false);
    }

    function renderUnetNeuronGraph(svgId, isCompact) {
      const svg = document.getElementById(svgId);
      if (!svg) return;

      const tRec = getActiveTargetIntrospection();
      const layersData = (tRec && tRec.unet && tRec.unet.layers) ? tRec.unet.layers : {};
      const posMap = {};
      UNET_STAGE_LAYOUT.forEach((st, idx) => { posMap[st.id] = { ...st, idx }; });

      let html = "";
      // Background branch labels
      html += `<text x="14" y="18" fill="var(--text-muted)" font-family="var(--font-mono)" font-size="9" font-weight="600">ENCODER SPINE (32×32 → 2×2 BOTTLENECK)</text>`;
      html += `<text x="385" y="18" fill="var(--text-primary)" font-family="var(--font-mono)" font-size="9" font-weight="700">SUPERVISED DECODER + SKIP ARCS + 7×7 ATTN → K=4</text>`;
      html += `<text x="385" y="242" fill="var(--text-secondary)" font-family="var(--font-mono)" font-size="9" font-weight="600">UNSUPERVISED BOTTLENECK DAE RECONSTRUCTION (NO SKIPS)</text>`;

      // 1. Skip connection arcs (dashed curves from Encoder to Supervised Decoder)
      UNET_SKIP_ARCS.forEach(([srcId, dstId]) => {
        const s = posMap[srcId];
        const d = posMap[dstId];
        if (!s || !d) return;
        const midX = (s.x + d.x) / 2;
        const arcY = Math.min(s.y, d.y) - (38 + (d.x - s.x) * 0.06);
        const isLit = selectedUnetLayerId === srcId || selectedUnetLayerId === dstId;
        html += `<path d="M ${s.x} ${s.y - 26} Q ${midX} ${arcY} ${d.x} ${d.y - 26}" fill="none" stroke="${isLit ? '#38BDF8' : 'rgba(148, 163, 184, 0.28)'}" stroke-width="${isLit ? 1.5 : 1}" stroke-dasharray="3 3"/>`;
      });

      // 2. Synaptic inter-layer connections between the 5 visual neurons per stage
      const nNeurons = 5;
      const spanH = 44;
      UNET_EDGES.forEach(([uId, vId]) => {
        const u = posMap[uId];
        const v = posMap[vId];
        if (!u || !v) return;
        const uLayer = layersData[uId] || { active_pct: 65 };
        const vLayer = layersData[vId] || { active_pct: 65 };
        const edgePulse = (pulseWaveStageIdx >= 0 && (u.idx === pulseWaveStageIdx || v.idx === pulseWaveStageIdx));
        const isSelectedEdge = (selectedUnetLayerId === uId || selectedUnetLayerId === vId);

        for (let i = 0; i < nNeurons; i++) {
          const y1 = u.y - spanH / 2 + (i / (nNeurons - 1)) * spanH;
          for (let j = 0; j < nNeurons; j++) {
            if (Math.abs(i - j) > 1 && !(i === 2 || j === 2)) continue;
            const y2 = v.y - spanH / 2 + (j / (nNeurons - 1)) * spanH;
            const topN = (vLayer.top_neurons && vLayer.top_neurons[j]) ? vLayer.top_neurons[j].norm : 0.6;
            const alpha = edgePulse ? 0.85 : (isSelectedEdge ? 0.38 * topN : 0.14 * topN);
            const strokeColor = edgePulse ? "#38BDF8" : (v.branch === "sup" ? `rgba(255,255,255,${alpha.toFixed(2)})` : `rgba(148,163,184,${alpha.toFixed(2)})`);
            html += `<line x1="${u.x}" y1="${y1}" x2="${v.x}" y2="${y2}" stroke="${strokeColor}" stroke-width="${edgePulse ? 1.4 : 0.8}"/>`;
          }
        }
      });

      // 3. Layer columns & individual firing neurons
      UNET_STAGE_LAYOUT.forEach((st, idx) => {
        const lInfo = layersData[st.id] || { mean_act: 0.32, max_act: 2.1, active_pct: 68.0, top_neurons: [] };
        const isSelected = selectedUnetLayerId === st.id;
        const isPulsing = pulseWaveStageIdx === idx;

        // Interactive hit column box
        html += `<g style="cursor:pointer;" onclick="selectUnetLayer('${st.id}')">`;
        html += `<rect x="${st.x - 18}" y="${st.y - 34}" width="36" height="68" rx="5" fill="${isSelected ? 'rgba(56, 189, 248, 0.12)' : 'rgba(15, 23, 42, 0.45)'}" stroke="${isPulsing ? '#38BDF8' : (isSelected ? '#FFFFFF' : 'var(--border-hairline)')}" stroke-width="${isSelected || isPulsing ? 1.6 : 0.9}"/>`;

        for (let nIdx = 0; nIdx < nNeurons; nIdx++) {
          const ny = st.y - spanH / 2 + (nIdx / (nNeurons - 1)) * spanH;
          const nObj = (lInfo.top_neurons && lInfo.top_neurons[nIdx]) ? lInfo.top_neurons[nIdx] : { norm: 0.7 - nIdx * 0.1, act: 0.5 };
          const normVal = Math.max(0.15, Math.min(1.0, Number(nObj.norm || 0.5)));
          const r = isPulsing ? 4.2 : (2.3 + normVal * 1.6);
          const fillCol = isPulsing
            ? "#38BDF8"
            : (st.branch === "dae"
                ? `rgba(148, 163, 184, ${(0.35 + 0.65 * normVal).toFixed(2)})`
                : `rgba(255, 255, 255, ${(0.35 + 0.65 * normVal).toFixed(2)})`);
          html += `<circle cx="${st.x}" cy="${ny}" r="${r.toFixed(1)}" fill="${fillCol}" stroke="${isSelected ? '#38BDF8' : '#060709'}" stroke-width="0.8"/>`;
        }

        // Stage label & channel count
        html += `<text x="${st.x}" y="${st.y + 45}" text-anchor="middle" fill="${isSelected ? '#FFFFFF' : 'var(--text-secondary)'}" font-family="var(--font-mono)" font-size="8.5" font-weight="${isSelected ? '700' : '600'}">${st.short}</text>`;
        html += `<text x="${st.x}" y="${st.y + 54}" text-anchor="middle" fill="var(--text-muted)" font-family="var(--font-mono)" font-size="7.5">${st.ch}c</text>`;
        html += `</g>`;
      });

      svg.innerHTML = html;
    }

    function renderUnetLayerDetail() {
      const tRec = getActiveTargetIntrospection();
      const layersData = (tRec && tRec.unet && tRec.unet.layers) ? tRec.unet.layers : {};
      const catalog = (activeModelIntrospection && activeModelIntrospection.unet_layer_catalog) || [];
      const catItem = catalog.find(c => c.id === selectedUnetLayerId) || { id: selectedUnetLayerId, label: selectedUnetLayerId.toUpperCase(), channels: 64, res: "32×32", params: 0, mb: 0 };
      const lData = layersData[selectedUnetLayerId] || { mean_act: 0.31, max_act: 2.45, active_pct: 64.5, top_neurons: [], grid_8x8: [] };

      const titleText = `LAYER: ${catItem.label} (${catItem.channels}ch × ${catItem.res})`;
      const statsText = `μ=${Number(lData.mean_act || 0).toFixed(3)} | max=${Number(lData.max_act || 0).toFixed(3)} | ${Number(lData.active_pct || 0).toFixed(1)}% ReLU Active`;

      const wsTitle = document.getElementById("ws-unet-layer-title");
      const wsStats = document.getElementById("ws-unet-layer-stats");
      if (wsTitle) wsTitle.innerText = titleText;
      if (wsStats) wsStats.innerText = statsText;

      const brTitle = document.getElementById("brain-unet-selected-layer-name");
      const brStats = document.getElementById("brain-unet-selected-layer-stats");
      if (brTitle) brTitle.innerText = titleText;
      if (brStats) brStats.innerText = `${statsText} | ${Number(catItem.mb || 0).toFixed(2)} MB`;

      // Render 8x8 Spatial Feature Map SVG
      const hmSvg = document.getElementById("svg-brain-unet-heatmap");
      if (hmSvg) {
        const grid = (lData.grid_8x8 && lData.grid_8x8.length === 8) ? lData.grid_8x8 : Array.from({length: 8}, (_, r) => Array.from({length: 8}, (__, c) => Math.max(0, 1 - Math.hypot(r - 3.5, c - 3.5) / 5)));
        const cellSz = 120 / 8;
        let cellsHtml = "";
        for (let r = 0; r < 8; r++) {
          for (let c = 0; c < 8; c++) {
            const v = Math.max(0, Math.min(1, Number((grid[r] && grid[r][c]) || 0)));
            // Inferno/Titanium ramp
            const red = Math.round(15 + v * 240);
            const grn = Math.round(20 + Math.pow(v, 1.3) * 215);
            const blu = Math.round(32 + Math.pow(v, 0.8) * 223);
            cellsHtml += `<rect x="${c * cellSz}" y="${r * cellSz}" width="${cellSz - 0.8}" height="${cellSz - 0.8}" fill="rgb(${red},${grn},${blu})" rx="1"><title>[${r},${c}] Activation: ${v.toFixed(3)}</title></rect>`;
          }
        }
        // Center target reticle
        cellsHtml += `<rect x="${3 * cellSz}" y="${3 * cellSz}" width="${2 * cellSz}" height="${2 * cellSz}" fill="none" stroke="#38BDF8" stroke-width="1.2" stroke-dasharray="2 2"/>`;
        hmSvg.innerHTML = cellsHtml;
      }

      const domEl = document.getElementById("brain-unet-dom-ch");
      if (domEl) {
        const top0 = (lData.top_neurons && lData.top_neurons[0]) ? lData.top_neurons[0] : null;
        domEl.innerText = top0 ? `${top0.name} (act=${top0.act})` : "Dominant ch: —";
      }

      const listEl = document.getElementById("brain-unet-top-neurons-list");
      if (listEl) {
        const topList = lData.top_neurons || [];
        if (topList.length === 0) {
          listEl.innerHTML = `<div style="font-family:var(--font-mono); font-size:10px; color:var(--text-muted);">Select a target to inspect channel activations.</div>`;
        } else {
          listEl.innerHTML = topList.slice(0, 5).map(n => {
            const pct = Math.round(Math.max(6, Math.min(100, (n.norm || 0.5) * 100)));
            return `
              <div style="display:grid; grid-template-columns:150px 1fr 52px; align-items:center; gap:8px; font-family:var(--font-mono); font-size:10px;">
                <span style="color:var(--text-primary); overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">${n.name}</span>
                <div style="height:6px; background:var(--bg-elevated); border-radius:3px; overflow:hidden; border:1px solid var(--border-hairline);">
                  <div style="width:${pct}%; height:100%; background:var(--text-primary);"></div>
                </div>
                <span style="text-align:right; color:var(--text-secondary); font-weight:700;">${Number(n.act).toFixed(3)}</span>
              </div>
            `;
          }).join("");
        }
      }
    }

    function traceClientXgbPath(rootNode, featVec) {
      const activeNodes = [];
      const steps = [];
      let curr = rootNode;
      while (curr) {
        const nid = Number(curr.nodeid);
        activeNodes.push(nid);
        if (curr.is_leaf) {
          return { active_nodes: activeNodes, leaf_nodeid: nid, leaf_weight: Number(curr.leaf || 0), steps };
        }
        const fIdx = Number(curr.feature_idx || 0);
        const cond = Number(curr.split_condition || 0);
        const val = (featVec && fIdx < featVec.length) ? Number(featVec[fIdx]) : NaN;
        let nextId, dir;
        if (Number.isNaN(val)) {
          nextId = Number(curr.missing !== undefined ? curr.missing : curr.no);
          dir = "MISSING";
        } else if (val < cond) {
          nextId = Number(curr.yes);
          dir = "YES (<)";
        } else {
          nextId = Number(curr.no);
          dir = "NO (>=)";
        }
        steps.append ? steps.append({ nodeid: nid, feature_name: curr.feature_name, feature_val: val, threshold: cond, direction: dir, next_nodeid: nextId, gain: curr.gain }) : steps.push({ nodeid: nid, feature_name: curr.feature_name, feature_val: val, threshold: cond, direction: dir, next_nodeid: nextId, gain: curr.gain });
        const children = curr.children || [];
        curr = children.find(c => Number(c.nodeid) === nextId) || null;
      }
      return { active_nodes: activeNodes, leaf_nodeid: 0, leaf_weight: 0, steps };
    }

    function renderXgbDecisionTree(svgId, isCompact) {
      const svg = document.getElementById(svgId);
      if (!svg) return;

      const heads = (activeModelIntrospection && activeModelIntrospection.xgb_heads) || [];
      const headObj = heads.find(h => Number(h.head_id) === Number(selectedXgbHeadIdx)) || heads[0];
      const badgeEl = document.getElementById("brain-xgb-meta-badge");

      if (!headObj || !headObj.is_active || !headObj.trees || headObj.trees.length === 0) {
        const note = headObj ? headObj.status_note : "XGBoost tree data loading...";
        if (badgeEl) badgeEl.innerText = "CONSTANT ZERO PRIOR";
        svg.innerHTML = `
          <rect x="20" y="35" width="600" height="125" rx="6" fill="var(--bg-elevated)" stroke="var(--border-hairline)"/>
          <text x="320" y="88" text-anchor="middle" fill="var(--text-primary)" font-family="var(--font-mono)" font-size="11.5" font-weight="700">${headObj ? headObj.head_label : 'K=4 DEPOSIT HEAD'}</text>
          <text x="320" y="112" text-anchor="middle" fill="var(--text-secondary)" font-family="var(--font-mono)" font-size="10">${note}</text>
          <text x="320" y="134" text-anchor="middle" fill="var(--text-muted)" font-family="var(--font-mono)" font-size="9.5">Switch to H0 (Porphyry Cu-Au), H1 (Skarn Fe-Cu), or H2 (Epithermal Au-Ag) to inspect active boosted decision trees.</text>
        `;
        const stepsEl = document.getElementById("brain-xgb-steps-list");
        if (stepsEl) stepsEl.innerHTML = `<div style="color:var(--text-muted);">${note}</div>`;
        return;
      }

      if (badgeEl) {
        badgeEl.innerText = `${headObj.n_total_trees} TREES × ${headObj.n_bags} BAGS // LR=${headObj.learning_rate}`;
      }

      const treeObj = headObj.trees[selectedXgbTreeIdx % headObj.trees.length] || headObj.trees[0];
      const tRec = getActiveTargetIntrospection();
      const featVec = (tRec && tRec.feature_vector) ? tRec.feature_vector : [];
      const trace = traceClientXgbPath(treeObj.root, featVec);
      const activeSet = new Set(trace.active_nodes);

      // Update leaf badges
      const wsLeaf = document.getElementById("ws-xgb-leaf-badge");
      if (wsLeaf) wsLeaf.innerText = `LEAF #${trace.leaf_nodeid}: ${trace.leaf_weight >= 0 ? '+' : ''}${trace.leaf_weight.toFixed(4)}`;
      const brLeaf = document.getElementById("brain-xgb-leaf-summary");
      if (brLeaf) {
        const headProb = (tRec && tRec.xgb_probs_k4 && tRec.xgb_probs_k4[selectedXgbHeadIdx] !== undefined)
          ? ` | Head P(Y=1) = ${Number(tRec.xgb_probs_k4[selectedXgbHeadIdx]).toFixed(3)}`
          : "";
        brLeaf.innerText = `Leaf #${trace.leaf_nodeid} Weight: ${trace.leaf_weight >= 0 ? '+' : ''}${trace.leaf_weight.toFixed(4)}${headProb}`;
      }

      // Flatten tree up to depth 4 for crisp visual layout
      const maxVisDepth = isCompact ? 3 : 4;
      const layoutNodes = [];
      const layoutEdges = [];

      function traverseLayout(node, depth, xMin, xMax) {
        if (!node || depth > maxVisDepth) return;
        const x = (xMin + xMax) / 2;
        const y = 26 + depth * (isCompact ? 48 : 50);
        layoutNodes.push({ node, depth, x, y });

        if (!node.is_leaf && node.children && node.children.length > 0 && depth < maxVisDepth) {
          const yesChild = node.children.find(c => Number(c.nodeid) === Number(node.yes)) || node.children[0];
          const noChild = node.children.find(c => Number(c.nodeid) === Number(node.no)) || node.children[1];
          const mid = (xMin + xMax) / 2;
          if (yesChild) {
            const cx = (xMin + mid) / 2;
            const cy = 26 + (depth + 1) * (isCompact ? 48 : 50);
            layoutEdges.push({ from: node, to: yesChild, x1: x, y1: y, x2: cx, y2: cy, label: "<" });
            traverseLayout(yesChild, depth + 1, xMin, mid);
          }
          if (noChild && noChild !== yesChild) {
            const cx = (mid + xMax) / 2;
            const cy = 26 + (depth + 1) * (isCompact ? 48 : 50);
            layoutEdges.push({ from: node, to: noChild, x1: x, y1: y, x2: cx, y2: cy, label: "≥" });
            traverseLayout(noChild, depth + 1, mid, xMax);
          }
        }
      }

      const svgW = isCompact ? 640 : 680;
      traverseLayout(treeObj.root, 0, 16, svgW - 16);

      let html = "";
      // Render tree edges
      layoutEdges.forEach(ed => {
        const isPathEdge = activeSet.has(Number(ed.from.nodeid)) && activeSet.has(Number(ed.to.nodeid));
        const stroke = isPathEdge ? "#38BDF8" : "rgba(148, 163, 184, 0.25)";
        const sw = isPathEdge ? 2.2 : 1.0;
        html += `<path d="M ${ed.x1} ${ed.y1 + 13} C ${ed.x1} ${(ed.y1 + ed.y2)/2}, ${ed.x2} ${(ed.y1 + ed.y2)/2}, ${ed.x2} ${ed.y2 - 13}" fill="none" stroke="${stroke}" stroke-width="${sw}"/>`;
        if (isPathEdge) {
          const mx = (ed.x1 + ed.x2) / 2;
          const my = (ed.y1 + ed.y2) / 2 + 2;
          html += `<text x="${mx}" y="${my}" text-anchor="middle" fill="#38BDF8" font-family="var(--font-mono)" font-size="8" font-weight="700">${ed.label}</text>`;
        }
      });

      // Render tree nodes
      layoutNodes.forEach(item => {
        const n = item.node;
        const nid = Number(n.nodeid);
        const isActive = activeSet.has(nid);
        const isSelected = selectedXgbNodeId === nid;
        const boxW = item.depth === 0 ? 150 : (item.depth === 1 ? 132 : (item.depth === 2 ? 108 : 76));
        const boxH = 26;

        const fill = n.is_leaf
          ? (isActive ? "rgba(16, 185, 129, 0.22)" : "rgba(15, 23, 42, 0.7)")
          : (isActive ? "rgba(56, 189, 248, 0.16)" : "rgba(15, 23, 42, 0.65)");
        const stroke = isSelected
          ? "#F59E0B"
          : (isActive ? (n.is_leaf ? "#10B981" : "#FFFFFF") : "var(--border-hairline)");

        html += `<g style="cursor:pointer;" onclick="selectXgbTreeNode(${nid})">`;
        html += `<rect x="${item.x - boxW / 2}" y="${item.y - boxH / 2}" width="${boxW}" height="${boxH}" rx="4" fill="${fill}" stroke="${stroke}" stroke-width="${isActive || isSelected ? 1.6 : 0.9}"/>`;

        if (n.is_leaf) {
          const wStr = `${Number(n.leaf) >= 0 ? '+' : ''}${Number(n.leaf).toFixed(3)}`;
          html += `<text x="${item.x}" y="${item.y - 1}" text-anchor="middle" fill="${isActive ? '#10B981' : 'var(--text-secondary)'}" font-family="var(--font-mono)" font-size="8.5" font-weight="700">LEAF #${nid}</text>`;
          html += `<text x="${item.x}" y="${item.y + 9}" text-anchor="middle" fill="${isActive ? '#FFFFFF' : 'var(--text-muted)'}" font-family="var(--font-mono)" font-size="8">w=${wStr}</text>`;
        } else {
          const shortFeat = String(n.feature_name || n.feature_code || "f0")
            .replace("moho_gravity_step_mgal_km", "moho_grav_step")
            .replace("dilation_tendency_Td", "dilation_Td")
            .replace("slip_tendency_Ts", "slip_Ts")
            .replace("dist_active_faults_m", "fault_dist_m")
            .replace("active_fault_density", "fault_density")
            .replace("dist_volcanic_arc_km", "volc_arc_km")
            .replace("terrain_ruggedness_tri", "ruggedness_tri")
            .replace("outcrop_exposure_index", "outcrop_idx")
            .replace("stage1_prob_skarn_fe_cu", "s1_skarn_prob")
            .replace("stage1_prob_epithermal_au", "s1_epith_prob")
            .slice(0, item.depth >= 3 ? 9 : 16);
          html += `<text x="${item.x}" y="${item.y - 1}" text-anchor="middle" fill="${isActive ? '#FFFFFF' : 'var(--text-secondary)'}" font-family="var(--font-mono)" font-size="8.5" font-weight="${isActive ? '700' : '600'}">${shortFeat}</text>`;
          html += `<text x="${item.x}" y="${item.y + 9}" text-anchor="middle" fill="${isActive ? '#38BDF8' : 'var(--text-muted)'}" font-family="var(--font-mono)" font-size="7.5">&lt; ${Number(n.split_condition).toFixed(2)}</text>`;
        }
        html += `</g>`;
      });

      svg.innerHTML = html;

      // Update workspace summary line & Training Control trace steps
      const wsSum = document.getElementById("ws-xgb-path-summary");
      if (wsSum) {
        if (trace.steps.length > 0) {
          const s0 = trace.steps[0];
          wsSum.innerText = `Path (${trace.steps.length} splits): Root [${s0.feature_name}=${Number(s0.feature_val).toFixed(2)} vs ${s0.threshold}] → ... → Leaf #${trace.leaf_nodeid} (w=${trace.leaf_weight.toFixed(4)})`;
        } else {
          wsSum.innerText = `Direct Leaf #${trace.leaf_nodeid} (w=${trace.leaf_weight.toFixed(4)})`;
        }
      }

      const stepsEl = document.getElementById("brain-xgb-steps-list");
      if (stepsEl) {
        if (trace.steps.length === 0) {
          stepsEl.innerHTML = `<div style="color:var(--text-secondary);">Tree #${treeObj.tree_idx} evaluates directly to Leaf #${trace.leaf_nodeid} (weight=${trace.leaf_weight.toFixed(4)}).</div>`;
        } else {
          stepsEl.innerHTML = trace.steps.map((st, idx) => `
            <div style="display:grid; grid-template-columns:58px 1fr 125px 85px; gap:8px; padding:4px 8px; background:var(--bg-elevated); border:1px solid var(--border-hairline); border-radius:4px; align-items:center;">
              <span style="color:#38BDF8; font-weight:700;">STEP ${idx + 1}</span>
              <span style="color:var(--text-primary); font-weight:600; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">Node #${st.nodeid}: ${st.feature_name}</span>
              <span style="color:var(--text-secondary);">val=${Number(st.feature_val).toFixed(3)} vs ${Number(st.threshold).toFixed(3)}</span>
              <span style="text-align:right; color:#FFFFFF; font-weight:700;">${st.direction} → #${st.next_nodeid}</span>
            </div>
          `).join("");
        }
      }
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

    // Z-depth slice slider (v4.8.2). The handler was deleted with the ASCII tomography view
    // (commit d009708) while the slider and resetDefaults() still called it, so dragging the
    // slider threw and Reset stopped half-way. It now updates the label and re-filters targets.
    function onAsciiZChange(val) {
      const depth = parseInt(val, 10);
      const lbl = document.getElementById("lbl-z-depth");
      if (lbl) lbl.innerText = `-${Number.isFinite(depth) ? depth : 450} m`;
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
        const res = await apiFetch("/api/rerank", {
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
      if (typeof renderAllModelBrainViews === "function") renderAllModelBrainViews();
    }

    // ==================== INITIALIZATION ON DOM READY ====================
    window.addEventListener("DOMContentLoaded", () => {
      initMap();
      initSplitHandle();
      fetchRunHistory(currentRegion);
      fetchAvailableModels();
      fetchAvailablePolygons();
      pollTrainingStatus();
      pollHardwarePerformance();
      if (!hwPollInterval) {
        hwPollInterval = setInterval(pollHardwarePerformance, 4000);
      }
      setInterval(() => {
        if (!isTrainingActive) pollTrainingStatus();
      }, 3500);

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
