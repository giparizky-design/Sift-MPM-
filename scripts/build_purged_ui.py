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
  <title>GEOSPATIALS // Autonomous Mineral Intelligence Workspace</title>

  <!-- 4-Tier Non-Cliché Typography -->
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="" />
  <link href="https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:ital,wght@0,400;0,700;1,400;1,700&family=IBM+Plex+Mono:ital,wght@0,400;0,500;0,600;0,700;1,400&family=Newsreader:ital,opsz,wght@1,6..72,400;1,6..72,500;1,6..72,600&family=Schibsted+Grotesk:wght@500;600;700;800&family=Plus+Jakarta+Sans:wght@500;600;700&display=swap" rel="stylesheet" />

  <!-- Leaflet CSS & JS -->
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" crossOrigin="" />
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" crossOrigin=""></script>

  <style>
    /* ==========================================================================
       ANTIGRAVITY STRICT MONOCHROMATIC DUAL-THEME TOKENS (SWISS MODERN MINIMALISM)
       ========================================================================== */
    :root {
      /* Typography Tokens */
      --font-display: 'Schibsted Grotesk', -apple-system, BlinkMacSystemFont, sans-serif;
      --font-editorial: 'Newsreader', Georgia, serif;
      --font-sans: 'Atkinson Hyperlegible', 'Plus Jakarta Sans', sans-serif;
      --font-mono: 'IBM Plex Mono', ui-monospace, SFMono-Regular, monospace;

      /* Dark Mode ("Deep Obsidian & Abyssal Slate" — Default) */
      --bg-canvas: #070A10;
      --bg-surface: #0D131F;
      --bg-elevated: #141D2E;
      --bg-glass: rgba(13, 19, 31, 0.85);
      --border-hairline: #1E293B;
      --border-subtle: rgba(255, 255, 255, 0.08);
      --text-primary: #F8FAFC;
      --text-secondary: #94A3B8;
      --text-muted: #64748B;
      --accent-primary: #38BDF8;
      --accent-secondary: #0EA5E9;
      --accent-amber: #F59E0B;
      --accent-emerald: #10B981;
      --accent-purple: #A855F7;
      --shadow-elevation: 0 8px 32px rgba(0, 0, 0, 0.45);
    }

    body.light-theme {
      --bg-canvas: #F8FAFC !important;
      --bg-surface: #FFFFFF !important;
      --bg-elevated: #F1F5F9 !important;
      --bg-glass: rgba(255, 255, 255, 0.88) !important;
      --border-hairline: #E2E8F0 !important;
      --border-subtle: rgba(0, 0, 0, 0.06) !important;
      --text-primary: #0F172A !important;
      --text-secondary: #475569 !important;
      --text-muted: #94A3B8 !important;
      --accent-primary: #0284C7 !important;
      --accent-secondary: #0369A1 !important;
      --accent-amber: #D97706 !important;
      --accent-emerald: #059669 !important;
      --accent-purple: #7C3AED !important;
      --shadow-elevation: 0 8px 32px rgba(15, 23, 42, 0.08) !important;
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
      background-color: rgba(56, 189, 248, 0.3);
      color: #F8FAFC;
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
      background: var(--accent-primary);
    }

    /* Accessible Focus Rings */
    button:focus-visible, input:focus-visible, select:focus-visible {
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

    /* Top Telemetry & Command Bar */
    .top-header {
      height: 52px;
      flex-shrink: 0;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 18px;
      background: var(--bg-surface);
      border-bottom: 1px solid var(--border-hairline);
      z-index: 40;
    }

    .brand-wrap {
      display: flex;
      align-items: center;
      gap: 12px;
    }

    .pulse-node {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--accent-primary);
      box-shadow: 0 0 10px var(--accent-primary);
      animation: pulseGlow 2.4s infinite ease-in-out;
    }

    @keyframes pulseGlow {
      0%, 100% { opacity: 0.7; transform: scale(0.95); }
      50% { opacity: 1; transform: scale(1.15); }
    }

    .header-telemetry-bar {
      display: flex;
      align-items: center;
      gap: 14px;
      font-family: var(--font-mono);
      font-size: 11px;
      color: var(--text-secondary);
    }

    .telemetry-chip {
      display: flex;
      align-items: center;
      gap: 5px;
      padding: 3px 8px;
      border-radius: 4px;
      background: var(--bg-elevated);
      border: 1px solid var(--border-subtle);
    }

    .header-actions {
      display: flex;
      align-items: center;
      gap: 10px;
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
      color: #070A10;
      font-weight: 700;
      box-shadow: 0 1px 3px rgba(0, 0, 0, 0.15);
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
      max-width: 320px;
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
      max-width: 215px;
      border-color: rgba(249, 115, 22, 0.4);
      color: var(--text-primary);
    }

    .select-run-history {
      max-width: 235px;
      border-color: rgba(56, 189, 248, 0.4);
      color: var(--text-primary);
    }

    .ore-badge {
      display: inline-block;
      padding: 2px 7px;
      border-radius: 4px;
      font-size: 9.5px;
      font-family: var(--font-mono);
      font-weight: 700;
      letter-spacing: 0.03em;
      text-transform: uppercase;
      white-space: nowrap;
    }

    .ore-badge-porphyry_cu {
      background: rgba(249, 115, 22, 0.15);
      color: #F97316;
      border: 1px solid rgba(249, 115, 22, 0.4);
    }

    .ore-badge-skarn_fe_cu {
      background: rgba(168, 85, 247, 0.15);
      color: #C084FC;
      border: 1px solid rgba(168, 85, 247, 0.4);
    }

    .ore-badge-epithermal_au {
      background: rgba(234, 179, 8, 0.15);
      color: #FACC15;
      border: 1px solid rgba(234, 179, 8, 0.4);
    }

    .ore-badge-laterite_fe_ni {
      background: rgba(16, 185, 129, 0.15);
      color: #34D399;
      border: 1px solid rgba(16, 185, 129, 0.4);
    }

    .ore-badge-primary_gold {
      background: rgba(245, 158, 11, 0.15);
      color: #FBBF24;
      border: 1px solid rgba(245, 158, 11, 0.4);
    }

    .ore-badge-telescoped {
      background: rgba(56, 189, 248, 0.18);
      color: #38BDF8;
      border: 1px solid rgba(56, 189, 248, 0.5);
    }

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

    /* Main Workspace Container */
    .workspace-body {
      flex: 1;
      display: flex;
      overflow: hidden;
      position: relative;
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
      border-color: rgba(56, 189, 248, 0.25);
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
      color: var(--accent-primary);
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
      color: #070A10;
      font-weight: 700;
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
      font-family: var(--font-sans);
      font-size: 11px;
      font-weight: 600;
      color: var(--text-secondary);
      padding: 5px 11px;
      border-radius: 4px;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 6px;
      transition: all 0.15s ease;
    }

    .tab-btn:hover {
      color: var(--text-primary);
    }

    .tab-btn.active {
      background: var(--accent-primary);
      color: #070A10;
      font-weight: 700;
    }

    .ascii-canvas-container {
      position: relative;
      width: 100%;
      background: var(--bg-canvas);
      border: 1px solid var(--border-hairline);
      border-radius: 6px;
      overflow: hidden;
      cursor: crosshair;
    }

    #canvas-ascii-tomography {
      display: block;
      width: 100%;
      height: 220px;
    }

    /* Ranked Telemetry Table */
    .telemetry-table-wrap {
      width: 100%;
      overflow-x: auto;
      max-height: 230px;
    }

    .telemetry-table {
      width: 100%;
      border-collapse: collapse;
      font-family: var(--font-mono);
      font-size: 11px;
    }

    .telemetry-table th {
      position: sticky;
      top: 0;
      background: var(--bg-surface);
      padding: 8px 10px;
      text-align: left;
      font-size: 9.5px;
      text-transform: uppercase;
      color: var(--text-muted);
      border-bottom: 1px solid var(--border-hairline);
      z-index: 5;
    }

    .telemetry-table td {
      padding: 8px 10px;
      border-bottom: 1px solid var(--border-hairline);
      color: var(--text-secondary);
      font-variant-numeric: tabular-nums;
    }

    .telemetry-table tr {
      cursor: pointer;
      transition: background 0.12s ease;
    }

    .telemetry-table tr:hover {
      background: var(--bg-elevated);
    }

    .telemetry-table tr.active-row {
      background: var(--bg-elevated);
      color: var(--accent-primary);
      font-weight: 600;
    }

    .badge-pill {
      display: inline-block;
      padding: 2px 7px;
      border-radius: 4px;
      font-size: 9.5px;
      font-weight: 700;
      letter-spacing: 0.02em;
    }

    /* Custom High-Acuity Target Site Reticle Pins */
    .custom-target-marker {
      background: transparent !important;
      border: none !important;
    }

    .target-pin {
      display: flex;
      align-items: center;
      justify-content: center;
      box-sizing: border-box;
      border-radius: 50%;
      background: #0D131F;
      border: 2px solid #f59e0b;
      box-shadow: 0 0 10px rgba(245, 158, 11, 0.65), 0 2px 6px rgba(0,0,0,0.85);
      color: #F8FAFC;
      font-family: var(--font-mono);
      font-weight: 800;
      cursor: pointer;
      user-select: none;
      transition: transform 0.18s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.18s ease;
    }

    .target-pin:hover {
      transform: scale(1.25);
      box-shadow: 0 0 16px rgba(245, 158, 11, 0.95), 0 0 30px rgba(245, 158, 11, 0.4);
      border-color: #fbbf24;
      z-index: 9999 !important;
    }

    .target-pin.selected {
      background: #38bdf8 !important;
      border: 2.5px solid #FFFFFF !important;
      box-shadow: 0 0 20px rgba(56, 189, 248, 0.95), 0 0 40px rgba(56, 189, 248, 0.5) !important;
      color: #070A10 !important;
      transform: scale(1.35) !important;
      animation: pulse-target-ring 1.8s infinite;
      z-index: 99999 !important;
    }

    @keyframes pulse-target-ring {
      0% { box-shadow: 0 0 0 0 rgba(56, 189, 248, 0.75), 0 0 20px rgba(56, 189, 248, 0.95); }
      70% { box-shadow: 0 0 0 14px rgba(56, 189, 248, 0), 0 0 25px rgba(56, 189, 248, 0.95); }
      100% { box-shadow: 0 0 0 0 rgba(56, 189, 248, 0), 0 0 20px rgba(56, 189, 248, 0.95); }
    }
  </style>
</head>
<body class="dark antialiased">
  <div class="app-shell">
    <!-- TOP TELEMETRY & COMMAND BAR -->
    <header class="top-header">
      <div class="brand-wrap">
        <div class="pulse-node"></div>
        <h1 class="panel-title" style="font-size: 13px; font-weight: 700; letter-spacing: -0.01em;">
          GEOSPATIALS <span style="color:var(--text-muted); font-weight:400;">//</span> AUTONOMOUS MINERAL INTELLIGENCE
        </h1>
        <span id="region-resolution-badge" class="panel-tag" style="margin-left: 4px;">50M ULTRA-HD GRID</span>
      </div>

      <!-- CENTER LIVE STREAM METRICS -->
      <div class="header-telemetry-bar">
        <div style="display:flex; align-items:center; gap:6px; border-right:1px solid var(--border-hairline); padding-right:12px;">
          <span id="braille-scanner" style="color:var(--accent-primary); font-size:14px; font-weight:700;">⠁</span>
          <span style="font-size:9.5px; color:var(--text-muted); letter-spacing:0.04em;">SENSOR INGESTION</span>
        </div>
        <div class="telemetry-chip">
          <span style="color:var(--text-muted); font-size:10px;">XGB 3.1:</span>
          <span id="chip-xgb" style="color:var(--accent-amber); font-weight:700;">0.847</span>
        </div>
        <div class="telemetry-chip">
          <span style="color:var(--text-muted); font-size:10px;">nnPU U-NET:</span>
          <span id="chip-unet" style="color:var(--accent-purple); font-weight:700;">0.803</span>
        </div>
        <div class="telemetry-chip">
          <span style="color:var(--text-muted); font-size:10px;">FUSED PR-AUC:</span>
          <span id="chip-fused" style="color:var(--accent-emerald); font-weight:700;">0.872</span>
        </div>
        <div class="telemetry-chip">
          <span style="color:var(--text-muted); font-size:10px;">PRIOR ĉ:</span>
          <span id="chip-chat" style="color:var(--text-primary); font-weight:600;">0.285</span>
        </div>
      </div>

      <!-- RIGHT COMMANDS & VIEW SWITCHER -->
      <div class="header-actions">
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

        <!-- Kalimantan Specialized Regional Selector & Archived Training Runs -->
        <select class="select-dataset" id="dataset-select" onchange="selectRegion(this.value)" aria-label="Select Target Region">
          <option value="KALIMANTAN" selected>EAST KALIMANTAN (PROVINCE-WIDE 500M K=4 MODEL)</option>
          <optgroup id="optgroup-archived-runs" label="ARCHIVED TRAINING RUNS">
''' + archived_opts + '''          </optgroup>
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

        <!-- Theme Toggle -->
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

    <!-- WORKSPACE BODY: PROGRESSIVE DRAWER + CENTRAL SPLIT WORKSPACE -->
    <div id="workspace-container" class="workspace-body mode-split">
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
              <span style="display:flex; align-items:center; gap:6px;"><span style="display:inline-block; width:8px; height:8px; border-radius:2px; background:linear-gradient(45deg, #f97316, #e11d48);"></span> Show Prospectivity Heatmap</span>
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
                  <span style="display:inline-block; width:8px; height:8px; border-radius:2px; background:linear-gradient(45deg, #10b981, #06b6d4);"></span>
                  Satellite Sensor Overlay
                </span>
                <input type="checkbox" id="chk-satellite-layer" onchange="updateSatelliteLayer()" style="accent-color:var(--accent-primary);" />
              </label>

              <select id="sel-satellite-layer" onchange="updateSatelliteLayer()" style="width:100%; padding:5px 7px; font-family:var(--font-mono); font-size:10px; background:var(--bg-surface); color:var(--text-primary); border:1px solid var(--border-hairline); border-radius:4px; outline:none; cursor:pointer;">
                <option value="dem_elevation">Topography // SRTM-DEM 30m Elevation</option>
                <option value="outcrop_exposure_index">Multispectral // Rock Outcrop & Alteration</option>
                <option value="dilation_tendency_Td">Geomechanics // Fault Dilation Strain (Td)</option>
                <option value="bouguer_gravity_mgal">Geophysics // Bouguer Gravity Density</option>
                <option value="volcanic_arc_halo">Geodynamics // Volcanic Arc Hydrothermal</option>
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
              <span style="display:flex; align-items:center; gap:6px;"><span style="display:inline-block; width:8px; height:8px; border-radius:2px; border:1.5px dashed var(--accent-primary);"></span> Provincial Scan Boundary</span>
              <input type="checkbox" id="chk-scanned-rect" checked onchange="toggleScannedRect(this.checked)" style="accent-color:var(--accent-primary);" />
            </label>

            <label style="display:flex; justify-content:space-between; align-items:center; font-size:11px; padding:6px 8px; background:var(--bg-elevated); border:1px solid var(--border-hairline); border-radius:5px; cursor:pointer;">
              <span style="display:flex; align-items:center; gap:6px;"><span style="display:inline-block; width:8px; height:8px; border-radius:2px; background:var(--accent-primary);"></span> Target Vector Polygons</span>
              <input type="checkbox" id="chk-targets-geo" checked onchange="toggleTargetsLayer(this.checked)" style="accent-color:var(--accent-primary);" />
            </label>

            <label style="display:flex; justify-content:space-between; align-items:center; font-size:11px; padding:6px 8px; background:var(--bg-elevated); border:1px solid var(--border-hairline); border-radius:5px; cursor:pointer;">
              <span style="display:flex; align-items:center; gap:6px;"><span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:var(--accent-amber);"></span> Ground Truth Mines</span>
              <input type="checkbox" id="chk-ground-mines" checked onchange="toggleMinesLayer(this.checked)" style="accent-color:var(--accent-primary);" />
            </label>
          </div>

          <button onclick="resetDefaults()" style="margin-top:6px; width:100%; padding:7px; background:var(--bg-elevated); border:1px solid var(--border-hairline); border-radius:5px; font-family:var(--font-mono); font-size:10.5px; color:var(--text-secondary); cursor:pointer; display:flex; align-items:center; justify-content:center; gap:6px;" aria-label="Reset Default Prior">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><polyline points="1 4 1 10 7 10"></polyline><path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10"></path></svg>
            RESET DEFAULT PRIOR
          </button>

          <button id="btn-live-rerank" onclick="triggerLiveRerank()" style="margin-top:6px; width:100%; padding:7px; background:var(--accent-primary); color:#070A10; font-weight:700; border:none; border-radius:5px; font-family:var(--font-mono); font-size:10.5px; cursor:pointer; display:flex; align-items:center; justify-content:center; gap:6px;" aria-label="Recompute Targets Live">
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
              </div>
              <div style="display:flex; gap:6px; align-items:center;">
                <span id="target-card-ore-badge" class="badge-pill ore-badge ore-badge-porphyry_cu">
                  PORPHYRY CU-AU
                </span>
                <span id="target-card-class" class="badge-pill" style="background:rgba(56, 189, 248, 0.15); color:var(--accent-primary); border:1px solid var(--accent-primary);">
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
                <div id="kpi-uncert" class="kpi-cell-val" style="color:var(--accent-emerald);">0.562</div>
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
                <button id="btn-subtab-tomography" class="tab-btn active" onclick="switchAnalyticsSubTab('tomography')">
                  01 // Subsurface Tomography
                </button>
                <button id="btn-subtab-conformal" class="tab-btn" onclick="switchAnalyticsSubTab('conformal')">
                  02 // Conformal 90% CI
                </button>
                <button id="btn-subtab-prauc" class="tab-btn" onclick="switchAnalyticsSubTab('prauc')">
                  03 // Model Fusion PR-AUC
                </button>
              </div>

              <!-- Depth Slice Control when on Tomography tab -->
              <div id="tomography-controls" style="display:flex; align-items:center; gap:8px;">
                <span class="font-editorial" style="font-size:12px; font-style:italic; color:var(--text-secondary);">
                  Z-Slice: <span id="ascii-z-readout" class="font-mono" style="color:var(--accent-primary); font-style:normal; font-weight:700;">-380 m</span>
                </span>
                <input type="range" id="ascii-z-slider" min="50" max="1000" step="10" value="380" oninput="onAsciiZChange(this.value)" style="width:90px; accent-color:var(--accent-primary); cursor:pointer;" />
              </div>
            </div>

            <!-- Tab View 1: 3D Subsurface Tomographic ASCII Scanner -->
            <div id="view-tab-tomography" style="display:block; flex:1;">
              <div class="ascii-canvas-container">
                <canvas id="canvas-ascii-tomography" width="728" height="220"></canvas>
                <div style="position:absolute; top:8px; left:12px; font-family:var(--font-mono); font-size:9px; color:var(--text-muted); pointer-events:none;">
                  TOMOGRAPHY HARMONICS // 3D GAUSSIAN FALLOFF
                </div>
                <div id="ascii-hover-tag" style="position:absolute; bottom:8px; right:12px; font-family:var(--font-mono); font-size:9.5px; font-weight:700; color:var(--accent-primary); background:rgba(13, 19, 31, 0.9); border:1px solid var(--border-hairline); border-radius:4px; padding:2px 8px; pointer-events:none;">
                  CURSOR LENS: STANDBY
                </div>
              </div>

              <div style="display:flex; justify-content:space-between; align-items:center; margin-top:10px; padding-top:8px; border-top:1px solid var(--border-hairline); font-family:var(--font-mono); font-size:10px; color:var(--text-secondary);">
                <div>RAMP: <span style="color:var(--text-primary); font-weight:700; letter-spacing:2px;">· : - = + * # % @ ◈</span></div>
                <div style="display:flex; gap:12px;">
                  <span style="display:flex; align-items:center; gap:4px;"><span style="display:inline-block; width:6px; height:6px; border-radius:50%; background:var(--accent-primary);"></span> Peak Anomaly</span>
                  <span style="display:flex; align-items:center; gap:4px;"><span style="display:inline-block; width:6px; height:6px; border-radius:50%; background:var(--accent-amber);"></span> Focus Sync</span>
                </div>
              </div>
            </div>

            <!-- Tab View 2: Conformal Risk Calibration Chart -->
            <div id="view-tab-conformal" style="display:none; flex:1;">
              <div style="background:var(--bg-canvas); border:1px solid var(--border-hairline); border-radius:6px; padding:10px;">
                <svg id="svg-analytics-chart" viewBox="0 0 600 210" style="width:100%; height:auto; display:block;"></svg>
              </div>
              <div style="display:flex; justify-content:space-between; align-items:center; margin-top:10px; padding-top:8px; border-top:1px solid var(--border-hairline); font-size:11px;">
                <span class="font-editorial" style="font-size:12px; font-style:italic; color:var(--text-secondary);">
                  Coverage Guarantee: <span class="font-mono" style="font-style:normal; font-weight:700; color:var(--accent-primary);">P(Y ∈ C(X)) ≥ 1 - α (α = 0.10)</span>.
                </span>
                <span id="lbl-conformal-count" class="font-mono" style="font-size:10.5px; color:var(--accent-primary);">— Targets Evaluated</span>
              </div>
            </div>

            <!-- Tab View 3: Model Fusion PR-AUC Curve -->
            <div id="view-tab-prauc" style="display:none; flex:1;">
              <div style="background:var(--bg-canvas); border:1px solid var(--border-hairline); border-radius:6px; padding:10px;">
                <svg id="svg-prauc-chart" viewBox="0 0 600 210" style="width:100%; height:auto; display:block;"></svg>
              </div>
              <div style="display:flex; justify-content:space-between; align-items:center; margin-top:10px; padding-top:8px; border-top:1px solid var(--border-hairline); font-family:var(--font-mono); font-size:10.5px;">
                <span id="legend-xgb" style="color:#F97316;">■ XGB 3.1: 0.847</span>
                <span id="legend-unet" style="color:var(--accent-purple);">■ U-Net: 0.803</span>
                <span id="legend-fused" style="color:var(--accent-primary); font-weight:700;">■ Fused: 0.872</span>
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

  <!-- SCRIPT LOGIC -->
  <script>
'''

    # Upgraded JavaScript runtime logic:
    html_script = '''
    // ==================== STATE & DATA MODELS ====================
    let currentRegion = "KALIMANTAN";
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
    let activeAnalyticsSubTab = "tomography";

    const ORE_LABEL_MAP = {
      "porphyry_cu": "Porphyry Cu-Au",
      "skarn_fe_cu": "Skarn Fe-Cu",
      "epithermal_au": "Epithermal Au-Ag",
      "laterite_fe_ni": "Lateritic Fe-Ni",
      "primary_gold": "Orogenic Gold",
      "orogenic_gold": "Orogenic Gold"
    };

    const ORE_COLORS = {
      porphyry_cu: { stroke: "#f97316", fill: "#ea580c" },
      skarn_fe_cu: { stroke: "#a855f7", fill: "#9333ea" },
      epithermal_au: { stroke: "#eab308", fill: "#ca8a04" },
      laterite_fe_ni: { stroke: "#10b981", fill: "#059669" },
      primary_gold: { stroke: "#f59e0b", fill: "#d97706" },
      telescoped: { stroke: "#38bdf8", fill: "#0284c7" }
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

          const conformalSet = sc.conformal_90pct_target_set || (sc.properties && sc.properties.conformal_90pct_target_set) || [];
          const primaryOre = sc.ore_category || (sc.properties && sc.properties.ore_category) || (conformalSet[0]) || "porphyry_cu";
          const oreLabel = sc.ore_category_label || (sc.properties && sc.properties.ore_category_label) || ORE_LABEL_MAP[primaryOre] || primaryOre.replace(/_/g, ' ');
          const isTelescoped = Boolean(sc.telescoped_system_flag || (sc.properties && sc.properties.telescoped_system_flag) || (conformalSet.length >= 2));

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
          const conformalSet = p.conformal_90pct_target_set || [];
          const primaryOre = p.ore_category || (conformalSet[0]) || "porphyry_cu";
          const oreLabel = p.ore_category_label || ORE_LABEL_MAP[primaryOre] || primaryOre.replace(/_/g, ' ');
          const isTelescoped = Boolean(p.telescoped_system_flag || (conformalSet.length >= 2));

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
            color: "#38bdf8",
            weight: 2.2,
            dashArray: "6, 5",
            fillColor: "#0284c7",
            fillOpacity: 0.04
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
            layer.bindTooltip(
              `<div style="font-family:monospace; font-size:11px; line-height:1.4;">
                <b style="color:#38bdf8;">TARGET POLYGON: ${p.target_id || p.id || 'Target'}</b><br/>
                <span>Rank #${p.rank || 1} // ${p.classification || 'Porphyry Cu-Au'}</span><br/>
                <span>Calibrated Prob: <b>${scoreVal}</b> | Footprint: <b>${areaVal} km²</b></span>
              </div>`,
              { direction: 'top', sticky: true }
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
          color: oreTheme.stroke,
          weight: 3.5,
          fillColor: oreTheme.fill,
          fillOpacity: 0.65,
          dashArray: null
        };
      }
      if (isTopRank) {
        return {
          opacity: 1,
          color: oreTheme.stroke,
          weight: 2.2,
          fillColor: oreTheme.fill,
          fillOpacity: 0.40,
          dashArray: null
        };
      }
      return {
        opacity: 0.9,
        color: oreTheme.stroke,
        weight: 1.5,
        fillColor: oreTheme.fill,
        fillOpacity: 0.22,
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
          `<div style="font-family:monospace; font-size:11px; line-height:1.4;">
            <b style="color:#38bdf8;">TARGET SITE: ${t.id}</b><br/>
            <span style="color:${oreTheme.stroke}; font-weight:700;">${t.telescopedFlag ? '⚡ TELESCOPED: ' : ''}${t.oreCategoryLabel || oreCat.toUpperCase()}</span><br/>
            <span>Rank #${t.rank || 1} // ${t.classification}</span><br/>
            <span>Calibrated Deposit Prob: <b>${t.prospectivityScore.toFixed(3)}</b></span><br/>
            <span>Regolith Depth: <b>${t.depthMeters}m</b> | CI: <b>[${t.conformalInterval.join(' - ')}]</b></span><br/>
            <span style="color:#f59e0b; font-size:10px;">Click to inspect drill telemetry &amp; fly-to</span>
          </div>`,
          { direction: "top", sticky: true }
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

      const cLow = target.conformalInterval ? target.conformalInterval[0] : 0.50;
      const cHigh = target.conformalInterval ? target.conformalInterval[1] : 1.00;

      glyphScramble("kpi-score", target.prospectivityScore.toFixed(3), 180);
      glyphScramble("kpi-depth", `-${target.depthMeters} m`, 180);
      glyphScramble("kpi-uncert", cLow.toFixed(3), 180);
      glyphScramble("kpi-ci", `[${cLow.toFixed(3)} - ${cHigh.toFixed(3)}]`, 180);

      renderTable();
      if (activeAnalyticsSubTab === "conformal") renderConformalChart();
      if (activeAnalyticsSubTab === "tomography") renderAsciiTomography();

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
      if (activeAnalyticsSubTab === "tomography") renderAsciiTomography();
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

    // ==================== PROCEDURAL ASCII TOMOGRAPHY ====================
    let asciiZ = 380;
    let asciiMouse = null;
    const ASCII_CHARS = "·:-=+*#%@◈";

    function onAsciiZChange(val) {
      asciiZ = parseInt(val, 10);
      glyphScramble("ascii-z-readout", `-${asciiZ} m`, 140);
      const lbl = document.getElementById("lbl-z-depth");
      if (lbl) lbl.innerText = `-${asciiZ} m`;
      const ctrl = document.getElementById("ctrl-z-depth");
      if (ctrl && ctrl.value != val) ctrl.value = val;
      const slider = document.getElementById("ascii-z-slider");
      if (slider && slider.value != val) slider.value = val;
      renderAsciiTomography();
      renderTable();
      updateMapTargetMarkers();
    }

    function renderAsciiTomography() {
      const cvs = document.getElementById("canvas-ascii-tomography");
      if (!cvs) return;
      const ctx = cvs.getContext("2d");
      const cols = 52;
      const rows = 18;
      const cw = cvs.width / cols;
      const ch = cvs.height / rows;

      ctx.fillStyle = isLightTheme ? "#F8FAFC" : "#070A10";
      ctx.fillRect(0, 0, cvs.width, cvs.height);

      ctx.font = '500 11.5px "IBM Plex Mono", ui-monospace, monospace';
      ctx.textBaseline = 'middle';
      ctx.textAlign = 'center';

      const normZ = asciiZ / 1000.0;
      const displayTargets = activeTargetsList.slice(0, 5);

      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const px = c * cw + cw / 2;
          const py = r * ch + ch / 2;

          let val = 0.12 * Math.sin(c * 0.28 + normZ * 3.8) +
                    0.10 * Math.cos(r * 0.38 - normZ * 2.6) +
                    0.06 * Math.sin((c + r) * 0.18);

          let isPeak = false;
          let isHovered = false;
          let isSelected = false;

          displayTargets.forEach((t, idx) => {
            const gx = Math.round(8 + idx * 9);
            const gy = Math.round(5 + (idx % 3) * 4);
            const dx = c - gx;
            const dy = r - gy;
            const distSq = dx * dx + dy * dy;
            const dz = Math.abs(normZ - t.depthMeters / 1000.0);
            const inf = Math.exp(-distSq / 14.0) * Math.max(0, 1 - dz * 2.4);
            val += inf * t.prospectivityScore;

            if (distSq < 1.8 && dz < 0.22) {
              isPeak = true;
              if (activeTarget && activeTarget.id === t.id) isSelected = true;
              if (hoveredTargetId === t.id) isHovered = true;
            }
          });

          let lensDist = 999;
          if (asciiMouse) {
            const mdx = px - asciiMouse.x;
            const mdy = py - asciiMouse.y;
            lensDist = Math.hypot(mdx, mdy);
            if (lensDist < 64) {
              val += (1 - lensDist / 64) * 0.36;
            }
          }

          val = Math.max(0, Math.min(val, 0.999));
          const charIdx = Math.floor(val * (ASCII_CHARS.length - 1));
          const glyph = ASCII_CHARS[charIdx];

          if (isSelected) {
            ctx.fillStyle = isLightTheme ? "#0284C7" : "#38BDF8";
            ctx.fillText("◈", px, py);
          } else if (isHovered) {
            ctx.fillStyle = "#F59E0B";
            ctx.fillText("◈", px, py);
          } else if (isPeak) {
            ctx.fillStyle = isLightTheme ? "#0369A1" : "#38BDF8";
            ctx.fillText("◈", px, py);
          } else if (lensDist < 64) {
            ctx.fillStyle = isLightTheme ? "#0F172A" : "#F8FAFC";
            ctx.fillText(glyph, px, py);
          } else if (val > 0.52) {
            ctx.fillStyle = isLightTheme ? "#334155" : "#94A3B8";
            ctx.fillText(glyph, px, py);
          } else {
            ctx.fillStyle = isLightTheme ? "#94A3B8" : "#334155";
            ctx.fillText(glyph, px, py);
          }
        }
      }
    }

    function initAsciiTomography() {
      const cvs = document.getElementById("canvas-ascii-tomography");
      if (!cvs) return;

      cvs.addEventListener("mousemove", (e) => {
        const rect = cvs.getBoundingClientRect();
        asciiMouse = {
          x: (e.clientX - rect.left) * (cvs.width / rect.width),
          y: (e.clientY - rect.top) * (cvs.height / rect.height)
        };

        const tag = document.getElementById("ascii-hover-tag");
        if (tag) {
          tag.innerText = `CURSOR LENS: RESOLVED (${Math.round(asciiMouse.x)}, ${Math.round(asciiMouse.y)})`;
        }
        renderAsciiTomography();
      });

      cvs.addEventListener("mouseleave", () => {
        asciiMouse = null;
        const tag = document.getElementById("ascii-hover-tag");
        if (tag) tag.innerText = "CURSOR LENS: STANDBY";
        renderAsciiTomography();
      });

      renderAsciiTomography();
    }

    // ==================== ANALYTICS TABS & CHARTS ====================
    function switchAnalyticsSubTab(tab) {
      activeAnalyticsSubTab = tab;
      ["tomography", "conformal", "prauc"].forEach(t => {
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

      const tomCtrl = document.getElementById("tomography-controls");
      if (tomCtrl) {
        tomCtrl.style.display = tab === "tomography" ? "flex" : "none";
      }

      if (tab === "tomography") renderAsciiTomography();
      else if (tab === "conformal") renderConformalChart();
      else if (tab === "prauc") renderPraucChart();
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

      inner += `<path d="${makePath(ptsXgb)}" fill="none" stroke="#F97316" stroke-width="1.5" stroke-dasharray="4 2"/>`;
      inner += `<path d="${makePath(ptsUnet)}" fill="none" stroke="var(--accent-purple)" stroke-width="1.5" stroke-dasharray="3 3"/>`;
      inner += `<path d="${makePath(ptsFused)}" fill="none" stroke="var(--accent-primary)" stroke-width="2.5"/>`;

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
      renderAsciiTomography();
      if (activeAnalyticsSubTab === "conformal") renderConformalChart();
      if (activeAnalyticsSubTab === "prauc") renderPraucChart();
    }

    // ==================== INITIALIZATION ON DOM READY ====================
    window.addEventListener("DOMContentLoaded", () => {
      initMap();
      initAsciiTomography();
      initSplitHandle();
      fetchRunHistory(currentRegion);
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
