'use client';

import React, { useState } from 'react';
import { DrillTarget, TARGET_DATASET, ThemeMode, TargetClassification } from '../types';
import { ZoomIn, ZoomOut, Layers, Filter, Compass } from 'lucide-react';
import { useScrambledValue } from '../hooks/useScrambledValue';

interface GeospatialMapViewportProps {
  activeTarget: DrillTarget | null;
  onSelectTarget: (target: DrillTarget) => void;
  hoveredTargetId?: string | null;
  onHoverTarget?: (targetId: string | null) => void;
  activeFilter?: string;
  onFilterChange?: (filter: string) => void;
  theme?: ThemeMode;
  selectedRegion?: string;
}

export const GeospatialMapViewport: React.FC<GeospatialMapViewportProps> = ({
  activeTarget,
  onSelectTarget,
  hoveredTargetId,
  onHoverTarget,
  activeFilter = 'ALL',
  onFilterChange,
  theme = 'dark',
  selectedRegion = 'KALIMANTAN'
}) => {
  const [zoom, setZoom] = useState<number>(12);
  const [basemap, setBasemap] = useState<'dark' | 'satellite' | 'topo'>('dark');
  const [localFilter, setLocalFilter] = useState<string>(activeFilter);
  const isLight = theme === 'light';

  const currentFilter = onFilterChange ? activeFilter : localFilter;
  const handleFilter = (flt: string) => {
    setLocalFilter(flt);
    if (onFilterChange) onFilterChange(flt);
  };

  // Easting / Northing dynamic HUD strings with scramble
  const scrambledEasting = useScrambledValue(
    activeTarget ? activeTarget.easting.toLocaleString() : '356,400',
    180
  );
  const scrambledNorthing = useScrambledValue(
    activeTarget ? activeTarget.northing.toLocaleString() : '6,598,200',
    180
  );

  const filteredTargets = TARGET_DATASET.filter(t => 
    currentFilter === 'ALL' || t.classification === currentFilter
  );

  return (
    <div className="relative w-full h-full flex-1 flex flex-col bg-[var(--bg-canvas)] overflow-hidden select-none">
      {/* Top Floating Frosted Glass Command Bar */}
      <div className="absolute top-3.5 left-3.5 right-3.5 flex items-center justify-between pointer-events-none z-30">
        {/* Basemap Switcher Pill */}
        <div className="pointer-events-auto flex items-center gap-1 bg-[var(--bg-glass)] backdrop-blur-md border border-[var(--border-hairline)] rounded-lg p-1 shadow-lg">
          <span className="font-mono text-[9.5px] text-[var(--text-muted)] px-2 font-medium">BASEMAP:</span>
          {(['dark', 'satellite', 'topo'] as const).map(mode => (
            <button
              key={mode}
              onClick={() => setBasemap(mode)}
              className={`font-mono text-[10px] font-semibold px-2.5 py-1 rounded transition-all ${
                basemap === mode
                  ? 'bg-[var(--accent-primary)] text-slate-950 shadow-sm font-bold'
                  : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-elevated)]'
              }`}
            >
              {mode === 'dark' ? 'Dark Obsidian' : mode === 'satellite' ? 'Satellite' : 'Terrain'}
            </button>
          ))}
        </div>

        {/* Target Quick Filter Pills */}
        <div className="pointer-events-auto flex items-center gap-1 bg-[var(--bg-glass)] backdrop-blur-md border border-[var(--border-hairline)] rounded-lg p-1 shadow-lg">
          {[
            { label: 'ALL (5)', value: 'ALL' },
            { label: 'TIER-1', value: 'TIER-1 CORE' },
            { label: 'EXT', value: 'STRUCTURAL EXT' },
            { label: 'SHEAR', value: 'SHEAR ANOMALY' },
            { label: 'BLIND', value: 'BLIND INFERRED' }
          ].map(flt => (
            <button
              key={flt.value}
              onClick={() => handleFilter(flt.value)}
              className={`font-mono text-[10px] font-semibold px-2.5 py-1 rounded transition-all ${
                currentFilter === flt.value
                  ? 'bg-[var(--accent-primary)] text-slate-950 shadow-sm font-bold'
                  : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-elevated)]'
              }`}
            >
              {flt.label}
            </button>
          ))}
        </div>
      </div>

      {/* Main Interactive Vector Map Viewport */}
      <div className="relative w-full h-full flex-1 overflow-hidden bg-[var(--bg-canvas)]">
        <svg viewBox="0 0 900 560" className="w-full h-full cursor-crosshair">
          <defs>
            <radialGradient id="target-glow" cx="50%" cy="50%" r="50%">
              <stop offset="0%" stopColor="var(--accent-primary)" stopOpacity="0.35" />
              <stop offset="100%" stopColor="var(--accent-primary)" stopOpacity="0" />
            </radialGradient>
          </defs>

          {/* Coordinate Hairline Grids */}
          {[80, 160, 240, 320, 400, 480].map((y) => (
            <line
              key={`h-grid-${y}`}
              x1="0"
              y1={y}
              x2="900"
              y2={y}
              stroke="var(--border-hairline)"
              strokeWidth="0.75"
              strokeDasharray="4 4"
            />
          ))}
          {[120, 240, 360, 480, 600, 720, 840].map((x) => (
            <line
              key={`v-grid-${x}`}
              x1={x}
              y1="0"
              x2={x}
              y2="560"
              stroke="var(--border-hairline)"
              strokeWidth="0.75"
              strokeDasharray="4 4"
            />
          ))}

          {/* Regional Major Shear & Fault Structural Lineaments */}
          <path
            d="M 60,60 Q 300,180 540,280 T 860,460"
            fill="none"
            stroke={isLight ? '#CBD5E1' : '#334155'}
            strokeWidth="1.75"
            strokeDasharray="8 4"
          />
          <path
            d="M 160,490 Q 380,360 620,200 T 840,50"
            fill="none"
            stroke={isLight ? '#E2E8F0' : '#1E293B'}
            strokeWidth="2.5"
          />
          <path
            d="M 280,50 Q 340,240 400,380 T 480,520"
            fill="none"
            stroke={isLight ? '#94A3B8' : '#475569'}
            strokeWidth="1.25"
            strokeDasharray="4 2"
          />

          {/* ================================================================= */}
          {/* TARGET SCANNED RECTANGLE AREA (BOUNDARY EXTENT & CORNER ANCHORS)   */}
          {/* ================================================================= */}
          <g className="transition-opacity duration-300">
            {/* Scanned Area Main Bounding Rectangle */}
            <rect
              x={90}
              y={70}
              width={720}
              height={420}
              fill={isLight ? 'rgba(2, 132, 199, 0.04)' : 'rgba(56, 189, 248, 0.05)'}
              stroke="var(--accent-primary)"
              strokeWidth="1.75"
              strokeDasharray="6 6"
              rx="4"
            />

            {/* Corner Precision Bracket Anchors (NW, NE, SW, SE) */}
            {/* NW Corner */}
            <path d="M 80,95 L 80,70 L 105,70" fill="none" stroke="var(--accent-primary)" strokeWidth="2.5" />
            {/* NE Corner */}
            <path d="M 795,70 L 820,70 L 820,95" fill="none" stroke="var(--accent-primary)" strokeWidth="2.5" />
            {/* SW Corner */}
            <path d="M 80,465 L 80,490 L 105,490" fill="none" stroke="var(--accent-primary)" strokeWidth="2.5" />
            {/* SE Corner */}
            <path d="M 795,490 L 820,490 L 820,465" fill="none" stroke="var(--accent-primary)" strokeWidth="2.5" />

            {/* Scanned Area Metadata Tag Pill */}
            <g transform="translate(100, 88)">
              <rect
                x="0"
                y="-14"
                width="310"
                height="22"
                rx="4"
                fill={isLight ? '#FFFFFF' : '#0D131F'}
                stroke="var(--accent-primary)"
                strokeWidth="1"
              />
              <circle cx="12" cy="-3" r="3.5" fill="var(--accent-primary)" />
              <text
                x="22"
                y="1"
                fill="var(--text-primary)"
                className="font-mono text-[9.5px] font-bold tracking-wider uppercase"
              >
                TARGET SCANNED AREA // {selectedRegion} [50M HD]
              </text>
            </g>

            {/* NW & SE Corner Coordinate Labels */}
            <text
              x="100"
              y="112"
              fill="var(--text-muted)"
              className="font-mono text-[8.5px]"
            >
              NW BOUND: -0°45&apos;12&quot;S, 114°12&apos;30&quot;E
            </text>
            <text
              x="800"
              y="480"
              textAnchor="end"
              fill="var(--text-muted)"
              className="font-mono text-[8.5px]"
            >
              SE BOUND: -3°10&apos;45&quot;S, 116°50&apos;15&quot;E
            </text>

            {/* Sub-Camp High-Acuity Nested Rectangles (e.g. Kelian & Schwaner) */}
            <rect
              x={300}
              y={140}
              width={260}
              height={180}
              fill="rgba(245, 158, 11, 0.05)"
              stroke="#F59E0B"
              strokeWidth="1.25"
              strokeDasharray="4 4"
              rx="2"
            />
            <text
              x={310}
              y={156}
              fill="#F59E0B"
              className="font-mono text-[8.5px] font-semibold uppercase"
            >
              50M ULTRA-HD CORE FOCUS CAMP
            </text>
          </g>

          {/* Mineral Deposit Targets */}
          {filteredTargets.map((target) => {
            const coords: Record<string, [number, number]> = {
              'TGT-YIL-01': [420, 260],
              'TGT-YIL-02': [320, 180],
              'TGT-YIL-03': [640, 130],
              'TGT-YIL-04': [210, 360],
              'TGT-YIL-05': [560, 390],
            };
            const [cx, cy] = coords[target.id] || [450, 260];
            const isSelected = activeTarget?.id === target.id;
            const isHovered = hoveredTargetId === target.id;

            return (
              <g
                key={target.id}
                className="cursor-pointer"
                onClick={() => onSelectTarget(target)}
                onMouseEnter={() => onHoverTarget && onHoverTarget(target.id)}
                onMouseLeave={() => onHoverTarget && onHoverTarget(null)}
              >
                {/* Conformal Uncertainty Buffer Ring */}
                <circle
                  cx={cx}
                  cy={cy}
                  r={isSelected ? 42 : isHovered ? 32 : 24}
                  fill={isSelected ? (isLight ? 'rgba(2, 132, 199, 0.14)' : 'rgba(56, 189, 248, 0.16)') : 'transparent'}
                  stroke={isSelected ? 'var(--accent-primary)' : isHovered ? '#F59E0B' : 'var(--border-hairline)'}
                  strokeWidth={isSelected ? '1.5' : '1'}
                  strokeDasharray={isSelected ? 'none' : '4 3'}
                />

                {/* Outer Glow on Selected */}
                {isSelected && (
                  <circle
                    cx={cx}
                    cy={cy}
                    r={24}
                    fill="url(#target-glow)"
                  />
                )}

                {/* Target Core Center */}
                <circle
                  cx={cx}
                  cy={cy}
                  r={isSelected ? 7 : 5}
                  fill={isSelected ? 'var(--accent-primary)' : isHovered ? '#F59E0B' : 'var(--text-primary)'}
                  stroke="var(--bg-canvas)"
                  strokeWidth="2"
                />

                {/* Target Tag Label */}
                <text
                  x={cx + 14}
                  y={cy - 4}
                  fill={isSelected ? 'var(--accent-primary)' : 'var(--text-primary)'}
                  className="font-mono text-[11px] font-bold"
                >
                  {target.id}
                </text>
                <text
                  x={cx + 14}
                  y={cy + 10}
                  fill="var(--text-secondary)"
                  className="font-sans text-[9.5px] font-medium"
                >
                  Score: {target.prospectivityScore.toFixed(3)}
                </text>
              </g>
            );
          })}
        </svg>

        {/* Floating Precision Telemetry HUD at Bottom */}
        <div className="absolute bottom-3.5 left-3.5 flex items-center gap-4 bg-[var(--bg-glass)] backdrop-blur-md px-3.5 py-2 rounded-lg border border-[var(--border-hairline)] font-mono text-[10.5px] text-[var(--text-secondary)] shadow-lg pointer-events-none z-30">
          <div>
            <span className="text-[var(--text-muted)] mr-1.5 font-medium">EASTING:</span>
            <span className="text-[var(--text-primary)] font-bold tabular-nums">{scrambledEasting} mE</span>
          </div>
          <div className="border-l border-[var(--border-hairline)] pl-3.5">
            <span className="text-[var(--text-muted)] mr-1.5 font-medium">NORTHING:</span>
            <span className="text-[var(--text-primary)] font-bold tabular-nums">{scrambledNorthing} mN</span>
          </div>
          <div className="border-l border-[var(--border-hairline)] pl-3.5">
            <span className="text-[var(--text-muted)] mr-1.5 font-medium">ZOOM:</span>
            <span className="text-[var(--accent-primary)] font-bold tabular-nums">{zoom}x</span>
          </div>
          <div className="border-l border-[var(--border-hairline)] pl-3.5">
            <span className="text-[var(--text-muted)] mr-1.5 font-medium">CRS:</span>
            <span className="text-[var(--text-primary)] font-semibold">EPSG:7851 (MGA51)</span>
          </div>
        </div>

        {/* Zoom Controls at Bottom Right */}
        <div className="absolute bottom-3.5 right-3.5 flex flex-col gap-1.5 z-30">
          <button
            onClick={() => setZoom(z => Math.min(18, z + 1))}
            className="p-2 rounded-lg bg-[var(--bg-glass)] backdrop-blur-md border border-[var(--border-hairline)] hover:border-[var(--accent-primary)] text-[var(--text-secondary)] hover:text-[var(--text-primary)] shadow-md transition-all cursor-pointer"
            title="Zoom In"
            aria-label="Zoom In"
          >
            <ZoomIn className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={() => setZoom(z => Math.max(8, z - 1))}
            className="p-2 rounded-lg bg-[var(--bg-glass)] backdrop-blur-md border border-[var(--border-hairline)] hover:border-[var(--accent-primary)] text-[var(--text-secondary)] hover:text-[var(--text-primary)] shadow-md transition-all cursor-pointer"
            title="Zoom Out"
            aria-label="Zoom Out"
          >
            <ZoomOut className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>
    </div>
  );
};
