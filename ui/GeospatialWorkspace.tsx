'use client';

import React, { useState, useEffect, useCallback } from 'react';
import {
  Sun,
  Moon,
  Sliders,
  Crosshair,
  Download,
  Filter,
  Columns,
  Maximize2,
  BarChart2
} from 'lucide-react';

import { DrillTarget, TARGET_DATASET, BRAILLE_FRAMES, ThemeMode, WorkspaceLayoutMode } from './types';
import { useScrambledValue } from './hooks/useScrambledValue';
import { AsciiTomographicScanner } from './components/AsciiTomographicScanner';
import { StatisticalAnalyticsChart } from './components/StatisticalAnalyticsChart';
import { GeospatialMapViewport } from './components/GeospatialMapViewport';
import { TelemetryDrawer } from './components/TelemetryDrawer';
import { TelemetryTable } from './components/TelemetryTable';

export default function GeospatialWorkspace() {
  // Theme state
  const [theme, setTheme] = useState<ThemeMode>('dark');

  // Selected Scanned Region
  const [selectedRegion, setSelectedRegion] = useState<string>('KALIMANTAN');

  // Progressive Disclosure Drawer state: 270px expanded / 48px collapsed
  const [drawerOpen, setDrawerOpen] = useState<boolean>(true);

  // Workspace layout mode
  const [layoutMode, setLayoutMode] = useState<WorkspaceLayoutMode>('split');

  // Pipeline parameter state
  const [zDepth, setZDepth] = useState<number>(380);
  const [lThreshold, setLThreshold] = useState<number>(0.50);
  const [minAreaHa, setMinAreaHa] = useState<number>(5.0);

  // Cross-pane synchronized selection & hover state
  const [activeTarget, setActiveTarget] = useState<DrillTarget | null>(TARGET_DATASET[0]);
  const [hoveredTargetId, setHoveredTargetId] = useState<string | null>(null);
  const [activeFilter, setActiveFilter] = useState<string>('ALL');
  const [activeAnalyticsTab, setActiveAnalyticsTab] = useState<'tomography' | 'conformal' | 'prauc'>('tomography');

  // Split-pane layout drag ratio (left vs right split in workspace)
  const [splitRatio, setSplitRatio] = useState<number>(54);
  const [isDraggingSplit, setIsDraggingSplit] = useState<boolean>(false);

  // Live ASCII Braille stream sequence
  const [brailleIndex, setBrailleIndex] = useState<number>(0);

  useEffect(() => {
    const timer = setInterval(() => {
      setBrailleIndex(idx => (idx + 1) % BRAILLE_FRAMES.length);
    }, 130);
    return () => clearInterval(timer);
  }, []);

  // Split pane drag listeners
  const handleSplitMouseMove = useCallback((e: MouseEvent) => {
    if (!isDraggingSplit) return;
    const container = document.getElementById('split-workspace-container');
    if (!container) return;
    const rect = container.getBoundingClientRect();
    const ratio = ((e.clientX - rect.left) / rect.width) * 100;
    setSplitRatio(Math.min(75, Math.max(25, ratio)));
  }, [isDraggingSplit]);

  const handleSplitMouseUp = useCallback(() => {
    setIsDraggingSplit(false);
  }, []);

  useEffect(() => {
    if (isDraggingSplit) {
      window.addEventListener('mousemove', handleSplitMouseMove);
      window.addEventListener('mouseup', handleSplitMouseUp);
    } else {
      window.removeEventListener('mousemove', handleSplitMouseMove);
      window.removeEventListener('mouseup', handleSplitMouseUp);
    }
    return () => {
      window.removeEventListener('mousemove', handleSplitMouseMove);
      window.removeEventListener('mouseup', handleSplitMouseUp);
    };
  }, [isDraggingSplit, handleSplitMouseMove, handleSplitMouseUp]);

  // Scrambled KPI transitions on active target change
  const scrambledScore = useScrambledValue(
    activeTarget ? activeTarget.prospectivityScore.toFixed(3) : '0.942',
    180
  );
  const scrambledDepth = useScrambledValue(
    activeTarget ? `-${activeTarget.depthMeters} m` : '-380 m',
    180
  );
  const scrambledUncertainty = useScrambledValue(
    activeTarget ? `±${activeTarget.epistemicUncertainty.toFixed(3)}` : '±0.048',
    180
  );
  const scrambledCoverage = useScrambledValue(
    activeTarget ? `[${activeTarget.conformalInterval.join(' - ')}]` : '[0.894 - 0.978]',
    180
  );

  return (
    <div
      className={`h-screen w-full flex flex-col font-sans transition-colors duration-200 overflow-hidden ${
        theme === 'light' ? 'light-theme' : ''
      }`}
      style={{
        ['--bg-canvas' as any]: theme === 'dark' ? '#070A10' : '#F8FAFC',
        ['--bg-surface' as any]: theme === 'dark' ? '#0D131F' : '#FFFFFF',
        ['--bg-elevated' as any]: theme === 'dark' ? '#141D2E' : '#F1F5F9',
        ['--bg-glass' as any]: theme === 'dark' ? 'rgba(13, 19, 31, 0.85)' : 'rgba(255, 255, 255, 0.88)',
        ['--border-hairline' as any]: theme === 'dark' ? '#1E293B' : '#E2E8F0',
        ['--text-primary' as any]: theme === 'dark' ? '#F8FAFC' : '#0F172A',
        ['--text-secondary' as any]: theme === 'dark' ? '#94A3B8' : '#475569',
        ['--text-muted' as any]: theme === 'dark' ? '#64748B' : '#94A3B8',
        ['--accent-primary' as any]: theme === 'dark' ? '#38BDF8' : '#0284C7',
        ['--accent-secondary' as any]: theme === 'dark' ? '#0EA5E9' : '#0369A1',
        ['--accent-amber' as any]: theme === 'dark' ? '#F59E0B' : '#D97706',
        ['--accent-emerald' as any]: theme === 'dark' ? '#10B981' : '#059669',
        ['--accent-purple' as any]: theme === 'dark' ? '#A855F7' : '#7C3AED',
        backgroundColor: 'var(--bg-canvas)',
        color: 'var(--text-primary)'
      }}
    >
      {/* ================================================================== */}
      {/* TOP HEADER: TELEMETRY STREAM, VIEW SWITCHER & THEME TOGGLE         */}
      {/* ================================================================== */}
      <header className="h-[52px] flex items-center justify-between px-4 border-b border-[var(--border-hairline)] bg-[var(--bg-surface)] shrink-0 z-40 select-none">
        {/* Brand & Acuity Tag */}
        <div className="flex items-center gap-3">
          <div className="w-2 h-2 rounded-full bg-[var(--accent-primary)] animate-pulse" />
          <h1 className="font-display font-bold text-[13.5px] tracking-[-0.01em] text-[var(--text-primary)]">
            GEOSPATIALS <span className="text-[var(--text-muted)] font-normal">//</span> AUTONOMOUS MINERAL INTELLIGENCE
          </h1>
          <span className="font-mono text-[9px] font-bold text-[var(--accent-primary)] uppercase px-2 py-0.5 rounded bg-[var(--bg-elevated)] border border-[var(--border-hairline)] hidden sm:inline-block">
            50M HIGH-ACUITY GRID
          </span>
        </div>

        {/* Center Live Telemetry Bar */}
        <div className="hidden lg:flex items-center gap-3 font-mono text-[10.5px] text-[var(--text-secondary)]">
          <div className="flex items-center gap-2 border-r border-[var(--border-hairline)] pr-3">
            <span className="text-[var(--accent-primary)] font-bold text-[14px]">
              {BRAILLE_FRAMES[brailleIndex]}
            </span>
            <span className="text-[var(--text-muted)] text-[9.5px] tracking-wider uppercase">
              SENSOR STREAM
            </span>
          </div>

          <div className="flex items-center gap-1.5 px-2 py-0.5 rounded bg-[var(--bg-elevated)] border border-[var(--border-hairline)]">
            <span className="text-[var(--text-muted)]">XGB 3.1:</span>
            <span className="text-amber-400 font-bold tabular-nums">0.847</span>
          </div>
          <div className="flex items-center gap-1.5 px-2 py-0.5 rounded bg-[var(--bg-elevated)] border border-[var(--border-hairline)]">
            <span className="text-[var(--text-muted)]">U-NET:</span>
            <span className="text-purple-400 font-bold tabular-nums">0.803</span>
          </div>
          <div className="flex items-center gap-1.5 px-2 py-0.5 rounded bg-[var(--bg-elevated)] border border-[var(--border-hairline)]">
            <span className="text-[var(--text-muted)]">FUSED:</span>
            <span className="text-emerald-400 font-bold tabular-nums">0.872</span>
          </div>
          <div className="flex items-center gap-1.5 px-2 py-0.5 rounded bg-[var(--bg-elevated)] border border-[var(--border-hairline)]">
            <span className="text-[var(--text-muted)]">PRIOR ĉ:</span>
            <span className="text-[var(--text-primary)] font-semibold tabular-nums">0.285</span>
          </div>
        </div>

        {/* Right Actions: Layout Switcher, Region Selector, Theme Toggle */}
        <div className="flex items-center gap-2.5">
          {/* Segmented Layout Mode Pills */}
          <div className="flex items-center gap-1 bg-[var(--bg-elevated)] p-1 rounded-md border border-[var(--border-hairline)]">
            <button
              onClick={() => setLayoutMode('split')}
              className={`flex items-center gap-1 px-2 py-1 text-[10px] font-mono font-medium rounded transition-all cursor-pointer ${
                layoutMode === 'split'
                  ? 'bg-[var(--accent-primary)] text-slate-950 font-bold shadow-sm'
                  : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
              }`}
              title="Split Studio (Map + Analytics)"
            >
              <Columns className="w-3 h-3" />
              <span>SPLIT</span>
            </button>
            <button
              onClick={() => setLayoutMode('map')}
              className={`flex items-center gap-1 px-2 py-1 text-[10px] font-mono font-medium rounded transition-all cursor-pointer ${
                layoutMode === 'map'
                  ? 'bg-[var(--accent-primary)] text-slate-950 font-bold shadow-sm'
                  : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
              }`}
              title="Expansive Map View"
            >
              <Maximize2 className="w-3 h-3" />
              <span>MAP</span>
            </button>
            <button
              onClick={() => setLayoutMode('analytics')}
              className={`flex items-center gap-1 px-2 py-1 text-[10px] font-mono font-medium rounded transition-all cursor-pointer ${
                layoutMode === 'analytics'
                  ? 'bg-[var(--accent-primary)] text-slate-950 font-bold shadow-sm'
                  : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
              }`}
              title="Analytical Deep-Dive"
            >
              <BarChart2 className="w-3 h-3" />
              <span>ANALYTICS</span>
            </button>
          </div>

          <select 
            aria-label="Geospatial Target Region Selection"
            value={selectedRegion}
            onChange={(e) => setSelectedRegion(e.target.value)}
            className="bg-[var(--bg-elevated)] border border-[var(--border-hairline)] rounded-md text-[11px] font-mono px-2.5 py-1 text-[var(--text-primary)] focus:outline-none focus:border-[var(--accent-primary)] cursor-pointer"
          >
            <optgroup label="KALIMANTAN ISLAND &amp; 50M ULTRA-HD CAMPS">
              <option value="KALIMANTAN">KALIMANTAN (ISLAND-WIDE 50M ULTRA-HD // 42 TILES)</option>
              <option value="HOTSPOT_SCHWANER">WEST KALIMANTAN // SCHWANER RANGE (50M BLIND DISCOVERY)</option>
              <option value="HOTSPOT_KAYAN_NORTH">NORTH KALIMANTAN // KAYAN RIVER (50M FRACTURE JOG)</option>
              <option value="HOTSPOT_KAHAYAN">CENTRAL KALIMANTAN // UPPER KAHAYAN (50M SUTURE)</option>
              <option value="BORNEO">EAST KALIMANTAN // KELIAN EPITHERMAL (50M HD AU-AG)</option>
            </optgroup>
            <optgroup label="INDONESIA NATIONAL &amp; ISLAND ARCS">
              <option value="INDONESIA_GEORIMA">NATIONAL INDONESIA (5,000KM GEORIMA ATLAS)</option>
              <option value="PAPUA">PAPUA CENTRAL RANGE (GRASBERG-WABU 200M HD)</option>
              <option value="SUMBAWA">SUMBAWA ARC (BATU HIJAU-ELANG 230M HD)</option>
              <option value="SUMATRA">NORTH SUMATRA (MARTABE-TORU 240M HD)</option>
              <option value="SULAWESI">NORTH SULAWESI (TOKA TINDUNG-MINAHASA 290M HD)</option>
            </optgroup>
            <optgroup label="INTERACTIVE RUNTIME SCANS">
              <option value="CUSTOM_VIEWPORT">MAP VIEWPORT (INTERACTIVE 7-PASS SCAN)</option>
            </optgroup>
            <optgroup label="AUSTRALIA ARCHETYPE BENCHMARKS">
              <option value="YILGARN">YILGARN CRATON (KALGOORLIE 50M GOLDFIELDS)</option>
            </optgroup>
          </select>

          <button
            onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
            className="p-1.5 rounded-md bg-[var(--bg-elevated)] border border-[var(--border-hairline)] text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:border-[var(--accent-primary)] transition-colors cursor-pointer"
            title="Toggle Monochromatic Theme (Dark/Light)"
          >
            {theme === 'dark' ? <Sun className="w-3.5 h-3.5 text-amber-400" /> : <Moon className="w-3.5 h-3.5 text-sky-600" />}
          </button>
        </div>
      </header>

      {/* ================================================================== */}
      {/* MAIN WORKSPACE: PROGRESSIVE DRAWER + CENTRAL STUDIO                */}
      {/* ================================================================== */}
      <div className="flex-1 flex overflow-hidden relative">
        {/* Collapsible Left Parameter Drawer */}
        <TelemetryDrawer
          isOpen={drawerOpen}
          onToggle={() => setDrawerOpen(!drawerOpen)}
          zDepth={zDepth}
          onZDepthChange={setZDepth}
          lThreshold={lThreshold}
          onLThresholdChange={setLThreshold}
          minAreaHa={minAreaHa}
          onMinAreaChange={setMinAreaHa}
          theme={theme}
        />

        {/* Central Workspace Canvas */}
        <main
          id="split-workspace-container"
          className="flex-1 flex overflow-hidden relative bg-[var(--bg-canvas)]"
        >
          {/* Left Hero Pane: Expansive Geospatial Map */}
          {layoutMode !== 'analytics' && (
            <div
              className="h-full overflow-hidden relative"
              style={{ width: layoutMode === 'map' ? '100%' : `${splitRatio}%` }}
            >
              <GeospatialMapViewport
                activeTarget={activeTarget}
                onSelectTarget={setActiveTarget}
                hoveredTargetId={hoveredTargetId}
                onHoverTarget={setHoveredTargetId}
                activeFilter={activeFilter}
                onFilterChange={setActiveFilter}
                theme={theme}
                selectedRegion={selectedRegion}
              />
            </div>
          )}

          {/* Resizable 1px Hairline Drag Handle */}
          {layoutMode === 'split' && (
            <div
              onMouseDown={() => setIsDraggingSplit(true)}
              className={`w-[8px] -ml-[4px] -mr-[4px] cursor-col-resize z-28 transition-all flex items-center justify-center group select-none ${
                isDraggingSplit ? 'bg-transparent' : 'bg-transparent'
              }`}
              title="Drag to resize split panes"
            >
              <div className={`w-[1px] h-full transition-colors ${
                isDraggingSplit 
                  ? 'bg-[var(--accent-primary)] shadow-[0_0_10px_var(--accent-primary)]' 
                  : 'bg-[var(--border-hairline)] group-hover:bg-[var(--accent-primary)]'
              }`} />
            </div>
          )}

          {/* Right Analytical Studio Console */}
          {layoutMode !== 'map' && (
            <div
              className="h-full overflow-y-auto p-4 space-y-4"
              style={{ width: layoutMode === 'analytics' ? '100%' : `${100 - splitRatio}%` }}
            >
              {/* 1. Active Target Dossier Record Card */}
              {activeTarget && (
                <div className="bg-[var(--bg-surface)] border border-[var(--border-hairline)] rounded-lg p-4 transition-all duration-200">
                  <div className="flex items-center justify-between pb-2.5 mb-2.5 border-b border-[var(--border-hairline)]">
                    <span className="font-mono text-[9.5px] uppercase tracking-wider text-[var(--accent-primary)] font-semibold bg-[var(--bg-elevated)] px-2 py-0.5 rounded border border-[var(--border-hairline)]">
                      SELECTED TARGET RECORD
                    </span>
                    <span className="font-mono text-[10.5px] text-[var(--text-muted)]">
                      {activeTarget.id}
                    </span>
                  </div>

                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-3">
                    <div>
                      <h3 className="font-display font-semibold text-[15.5px] tracking-[-0.02em] text-[var(--text-primary)]">
                        {activeTarget.name}
                      </h3>
                      <p className="font-editorial italic text-[12.5px] text-[var(--text-secondary)] mt-0.5">
                        {activeTarget.structuralDomain} — Primary Vector: {activeTarget.primaryDriver}
                      </p>
                    </div>
                    <span className="font-mono text-[9.5px] font-bold px-2 py-0.5 rounded bg-[var(--bg-elevated)] border border-[var(--border-hairline)] text-[var(--accent-primary)] self-start sm:self-auto">
                      {activeTarget.classification}
                    </span>
                  </div>

                  {/* KPI Grid with Scrambled Transitions */}
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5 pt-2.5 border-t border-[var(--border-hairline)] font-mono">
                    <div className="p-2.5 rounded-md bg-[var(--bg-canvas)] border border-[var(--border-hairline)]">
                      <span className="font-sans text-[9px] uppercase tracking-wider text-[var(--text-muted)] block mb-1 font-semibold">
                        PROSPECTIVITY P(Y=1)
                      </span>
                      <span className="font-mono text-[16px] font-bold text-[var(--accent-primary)] tabular-nums slashed-zero">
                        {scrambledScore}
                      </span>
                    </div>

                    <div className="p-2.5 rounded-md bg-[var(--bg-canvas)] border border-[var(--border-hairline)]">
                      <span className="font-sans text-[9px] uppercase tracking-wider text-[var(--text-muted)] block mb-1 font-semibold">
                        Z-DEPTH TARGET
                      </span>
                      <span className="font-mono text-[16px] font-bold text-[var(--text-primary)] tabular-nums slashed-zero">
                        {scrambledDepth}
                      </span>
                    </div>

                    <div className="p-2.5 rounded-md bg-[var(--bg-canvas)] border border-[var(--border-hairline)]">
                      <span className="font-sans text-[9px] uppercase tracking-wider text-[var(--text-muted)] block mb-1 font-semibold">
                        EPISTEMIC UNCERTAINTY
                      </span>
                      <span className="font-mono text-[16px] font-bold text-emerald-400 tabular-nums slashed-zero">
                        {scrambledUncertainty}
                      </span>
                    </div>

                    <div className="p-2.5 rounded-md bg-[var(--bg-canvas)] border border-[var(--border-hairline)]">
                      <span className="font-sans text-[9px] uppercase tracking-wider text-[var(--text-muted)] block mb-1 font-semibold">
                        90% CONFORMAL CI
                      </span>
                      <span className="font-mono text-[12.5px] font-semibold text-[var(--text-secondary)] tabular-nums slashed-zero">
                        {scrambledCoverage}
                      </span>
                    </div>
                  </div>
                </div>
              )}

              {/* 2. Analytical Tabbed Views: Subsurface Tomography vs Statistical Charts */}
              <div className="bg-[var(--bg-surface)] border border-[var(--border-hairline)] rounded-lg p-4">
                <div className="flex items-center justify-between pb-2.5 mb-2.5 border-b border-[var(--border-hairline)]">
                  <div className="flex items-center gap-1 bg-[var(--bg-elevated)] p-1 rounded-md border border-[var(--border-hairline)]">
                    <button
                      onClick={() => setActiveAnalyticsTab('tomography')}
                      className={`px-2.5 py-1 text-[10.5px] font-sans font-semibold rounded transition-all cursor-pointer ${
                        activeAnalyticsTab === 'tomography'
                          ? 'bg-[var(--accent-primary)] text-slate-950 shadow-sm'
                          : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
                      }`}
                    >
                      01 // Subsurface Tomography
                    </button>
                    <button
                      onClick={() => setActiveAnalyticsTab('conformal')}
                      className={`px-2.5 py-1 text-[10.5px] font-sans font-semibold rounded transition-all cursor-pointer ${
                        activeAnalyticsTab === 'conformal'
                          ? 'bg-[var(--accent-primary)] text-slate-950 shadow-sm'
                          : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
                      }`}
                    >
                      02 // Conformal 90% CI
                    </button>
                    <button
                      onClick={() => setActiveAnalyticsTab('prauc')}
                      className={`px-2.5 py-1 text-[10.5px] font-sans font-semibold rounded transition-all cursor-pointer ${
                        activeAnalyticsTab === 'prauc'
                          ? 'bg-[var(--accent-primary)] text-slate-950 shadow-sm'
                          : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
                      }`}
                    >
                      03 // Model Fusion PR-AUC
                    </button>
                  </div>
                </div>

                {activeAnalyticsTab === 'tomography' && (
                  <AsciiTomographicScanner
                    zDepth={zDepth}
                    onZDepthChange={setZDepth}
                    activeTarget={activeTarget}
                    onSelectTarget={setActiveTarget}
                    hoveredTargetId={hoveredTargetId}
                    onHoverTarget={setHoveredTargetId}
                    theme={theme}
                  />
                )}

                {(activeAnalyticsTab === 'conformal' || activeAnalyticsTab === 'prauc') && (
                  <StatisticalAnalyticsChart
                    activeTarget={activeTarget}
                    onSelectTarget={setActiveTarget}
                    hoveredTargetId={hoveredTargetId}
                    onHoverTarget={setHoveredTargetId}
                    theme={theme}
                  />
                )}
              </div>

              {/* 3. High-Density Ranked Telemetry Portfolio Table */}
              <TelemetryTable
                activeTarget={activeTarget}
                onSelectTarget={setActiveTarget}
                hoveredTargetId={hoveredTargetId}
                onHoverTarget={setHoveredTargetId}
                theme={theme}
              />
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
