'use client';

import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import {
  Activity,
  Layers,
  Compass,
  Cpu,
  ChevronLeft,
  ChevronRight,
  Sun,
  Moon,
  Crosshair,
  Maximize2,
  Database,
  Radio,
  Sliders,
  ShieldAlert,
  Download,
  Terminal,
  Zap,
  Flame,
  Binary,
  RotateCcw
} from 'lucide-react';

// ============================================================================
// 1. DATA MODELS & TELEMETRY CONSTANTS
// ============================================================================

export type ThemeMode = 'dark' | 'light';

export interface DrillTarget {
  id: string;
  name: string;
  easting: number;
  northing: number;
  depthMeters: number;
  prospectivityScore: number;
  epistemicUncertainty: number;
  conformalInterval: [number, number];
  primaryDriver: string;
  structuralDomain: string;
  classification: 'TIER-1 CORE' | 'STRUCTURAL EXT' | 'SHEAR ANOMALY' | 'BLIND INFERRED';
}

export const TARGET_DATASET: DrillTarget[] = [
  {
    id: 'TGT-YIL-01',
    name: 'Boulder-Lefroy Shear Node A',
    easting: 356400,
    northing: 6598200,
    depthMeters: 380,
    prospectivityScore: 0.942,
    epistemicUncertainty: 0.048,
    conformalInterval: [0.894, 0.978],
    primaryDriver: 'Fault Dilation & Mag AS 1VD',
    structuralDomain: 'Kalgoorlie Terrane',
    classification: 'TIER-1 CORE'
  },
  {
    id: 'TGT-YIL-02',
    name: 'Zuleika Shear Dilational Jog',
    easting: 342150,
    northing: 6612800,
    depthMeters: 520,
    prospectivityScore: 0.887,
    epistemicUncertainty: 0.071,
    conformalInterval: [0.816, 0.945],
    primaryDriver: 'Hydrothermal K/Th Radiometrics',
    structuralDomain: 'Kurnalpi Terrane',
    classification: 'STRUCTURAL EXT'
  },
  {
    id: 'TGT-YIL-03',
    name: 'Laverton Tectonic Zone Crest',
    easting: 428900,
    northing: 6834100,
    depthMeters: 290,
    prospectivityScore: 0.915,
    epistemicUncertainty: 0.052,
    conformalInterval: [0.863, 0.957],
    primaryDriver: 'Banded Iron Contact Shear',
    structuralDomain: 'Laverton Domain',
    classification: 'TIER-1 CORE'
  },
  {
    id: 'TGT-YIL-04',
    name: 'Ida Fault Basement Splay',
    easting: 298400,
    northing: 6711500,
    depthMeters: 640,
    prospectivityScore: 0.794,
    epistemicUncertainty: 0.112,
    conformalInterval: [0.682, 0.891],
    primaryDriver: 'Gravity Bouguer HGM Gradient',
    structuralDomain: 'Youanmi Terrane Margin',
    classification: 'SHEAR ANOMALY'
  },
  {
    id: 'TGT-YIL-05',
    name: 'Kurnalpi Subsurface Diatreme',
    easting: 388100,
    northing: 6649700,
    depthMeters: 410,
    prospectivityScore: 0.832,
    epistemicUncertainty: 0.089,
    conformalInterval: [0.743, 0.908],
    primaryDriver: 'Copernicus DEM Slope & Curvature',
    structuralDomain: 'Edjudina Belt',
    classification: 'BLIND INFERRED'
  }
];

// ASCII Braille Stream sequence
const BRAILLE_FRAMES = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
const ASCII_RAMP = ' .:-=+*#%@';

// ============================================================================
// 2. KINETIC GLYPH-SCRAMBLE HOOK
// ============================================================================

export function useGlyphScramble(targetValue: string | number, durationMs: number = 180): string {
  const [displayValue, setDisplayValue] = useState<string>(String(targetValue));
  const chars = '0123456789ABCDEF!#%&*+=~';

  useEffect(() => {
    const finalStr = String(targetValue);
    const startTime = performance.now();
    let frameId: number;

    const animate = (currentTime: number) => {
      const elapsed = currentTime - startTime;
      const progress = Math.min(elapsed / durationMs, 1);

      if (progress < 1) {
        const scrambleLen = finalStr.length;
        let scrambled = '';
        const revealCount = Math.floor(progress * scrambleLen);

        for (let i = 0; i < scrambleLen; i++) {
          if (finalStr[i] === '.' || finalStr[i] === '-' || finalStr[i] === ' ' || finalStr[i] === '%') {
            scrambled += finalStr[i];
          } else if (i < revealCount) {
            scrambled += finalStr[i];
          } else {
            scrambled += chars[Math.floor(Math.random() * chars.length)];
          }
        }
        setDisplayValue(scrambled);
        frameId = requestAnimationFrame(animate);
      } else {
        setDisplayValue(finalStr);
      }
    };

    frameId = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(frameId);
  }, [targetValue, durationMs]);

  return displayValue;
}

// ============================================================================
// 3. PROCEDURAL ASCII SUBSURFACE DENSITY MATRIX COMPONENT
// ============================================================================

interface AsciiMatrixProps {
  zDepth: number;
  activeTarget: DrillTarget | null;
  onSelectTarget: (target: DrillTarget) => void;
  theme: ThemeMode;
}

export const AsciiSubsurfaceCanvas: React.FC<AsciiMatrixProps> = ({
  zDepth,
  activeTarget,
  onSelectTarget,
  theme
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [mousePos, setMousePos] = useState<{ x: number; y: number } | null>(null);
  const [hoveredCell, setHoveredCell] = useState<{ x: number; y: number; val: number } | null>(null);

  // Resolution of the ASCII grid
  const cols = 48;
  const rows = 28;
  const cellWidth = 14;
  const cellHeight = 16;

  // Render loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Handle high-DPI
    const dpr = window.devicePixelRatio || 1;
    canvas.width = cols * cellWidth * dpr;
    canvas.height = rows * cellHeight * dpr;
    ctx.scale(dpr, dpr);

    const isDark = theme === 'dark';
    const bgVoid = isDark ? '#090D16' : '#F8FAFC';
    const textDim = isDark ? '#334155' : '#CBD5E1';
    const textMid = isDark ? '#64748B' : '#64748B';
    const textBright = isDark ? '#F8FAFC' : '#0F172A';
    const accentCyan = isDark ? '#38BDF8' : '#0284C7';
    const accentAnomaly = isDark ? '#F43F5E' : '#E11D48';

    // Clear background
    ctx.fillStyle = bgVoid;
    ctx.fillRect(0, 0, cols * cellWidth, rows * cellHeight);

    ctx.font = '11px "IBM Plex Mono", monospace';
    ctx.textBaseline = 'middle';
    ctx.textAlign = 'center';

    const normalizedZ = zDepth / 1000;

    // Synthetic anomaly clusters mapped to target coordinates
    const clusters = TARGET_DATASET.map((t, idx) => ({
      gx: Math.floor(10 + (idx * 7) + Math.sin(idx * 1.8) * 4),
      gy: Math.floor(6 + (idx * 4.2) + Math.cos(idx * 2.1) * 3),
      depthRatio: t.depthMeters / 1000,
      target: t
    }));

    let closestHovered: { x: number; y: number; val: number } | null = null;

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const px = c * cellWidth + cellWidth / 2;
        const py = r * cellHeight + cellHeight / 2;

        // Base 3D tomographic potential field
        let potential =
          0.12 * Math.sin(c * 0.22 + normalizedZ * 4.0) +
          0.10 * Math.cos(r * 0.28 - normalizedZ * 3.0) +
          0.05 * Math.sin((c + r) * 0.18);

        // Anomaly additions
        let isPeak = false;
        let matchedTarget: DrillTarget | null = null;

        for (const cluster of clusters) {
          const dx = c - cluster.gx;
          const dy = r - cluster.gy;
          const distSq = dx * dx + dy * dy;
          const dz = Math.abs(normalizedZ - cluster.depthRatio);
          const sigma = 3.2;

          const influence = Math.exp(-distSq / (2 * sigma * sigma)) * Math.max(0, 1 - dz * 2.5);
          potential += influence * cluster.target.prospectivityScore;

          if (distSq < 1.8 && dz < 0.2) {
            isPeak = true;
            matchedTarget = cluster.target;
          }
        }

        // Mouse magnetic lens effect
        let lensDist = 999;
        if (mousePos) {
          const mdx = px - mousePos.x;
          const mdy = py - mousePos.y;
          lensDist = Math.sqrt(mdx * mdx + mdy * mdy);
          if (lensDist < 55) {
            const lensBoost = (1 - lensDist / 55) * 0.35;
            potential += lensBoost;
            if (lensDist < cellWidth * 0.8) {
              closestHovered = { x: c, y: r, val: Math.min(potential, 0.999) };
            }
          }
        }

        // Clamp & choose glyph
        const clamped = Math.max(0, Math.min(potential, 0.999));
        const charIdx = Math.floor(clamped * (ASCII_RAMP.length - 1));
        const glyph = ASCII_RAMP[charIdx];

        // Active target lock-on check
        const isCurrentActive =
          activeTarget && matchedTarget && activeTarget.id === matchedTarget.id;

        // Fill style based on state
        if (isCurrentActive) {
          ctx.fillStyle = accentAnomaly;
          ctx.fillText('█', px, py);
          // Animated lock ring around target
          ctx.strokeStyle = accentAnomaly;
          ctx.lineWidth = 1;
          ctx.strokeRect(px - cellWidth / 2 + 1, py - cellHeight / 2 + 1, cellWidth - 2, cellHeight - 2);
        } else if (isPeak) {
          ctx.fillStyle = accentCyan;
          ctx.fillText(glyph === ' ' ? '·' : glyph, px, py);
        } else if (lensDist < 55) {
          ctx.fillStyle = textBright;
          ctx.fillText(glyph, px, py);
        } else if (clamped > 0.45) {
          ctx.fillStyle = textMid;
          ctx.fillText(glyph, px, py);
        } else {
          ctx.fillStyle = textDim;
          ctx.fillText(glyph, px, py);
        }
      }
    }

    setHoveredCell(closestHovered);
  }, [zDepth, activeTarget, mousePos, theme]);

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    setMousePos({
      x: e.clientX - rect.left,
      y: e.clientY - rect.top
    });
  };

  const handleMouseLeave = () => {
    setMousePos(null);
    setHoveredCell(null);
  };

  const handleClick = () => {
    if (!hoveredCell) return;
    // Check if clicked near any anomaly cluster
    const normalizedZ = zDepth / 1000;
    for (let idx = 0; idx < TARGET_DATASET.length; idx++) {
      const gx = Math.floor(10 + (idx * 7) + Math.sin(idx * 1.8) * 4);
      const gy = Math.floor(6 + (idx * 4.2) + Math.cos(idx * 2.1) * 3);
      const dx = hoveredCell.x - gx;
      const dy = hoveredCell.y - gy;
      if (Math.hypot(dx, dy) < 2.5) {
        onSelectTarget(TARGET_DATASET[idx]);
        break;
      }
    }
  };

  return (
    <div className="relative flex flex-col items-center select-none bg-[var(--bg-surface)] border border-[var(--border-hairline)] rounded-sm p-3">
      {/* Tomography Header */}
      <div className="w-full flex items-center justify-between pb-2 mb-2 border-b border-[var(--border-hairline)]">
        <div className="flex items-center gap-2">
          <Binary className="w-3.5 h-3.5 text-[var(--accent-primary)] animate-pulse" />
          <span className="font-display text-[12px] font-semibold text-[var(--text-primary)] tracking-tight">
            SUB-SURFACE TOMOGRAPHIC ASCII FIELD
          </span>
          <span className="font-mono text-[10px] text-[var(--text-muted)] px-1.5 py-0.5 rounded bg-[var(--bg-elevated)]">
            48×28 RESOLUTION
          </span>
        </div>
        <div className="flex items-center gap-3">
          <span className="font-editorial text-[12px] text-[var(--text-secondary)] italic">
            Z-Slice: <span className="font-mono not-italic text-[var(--accent-primary)] font-semibold">-{zDepth}m RL</span>
          </span>
          {hoveredCell && (
            <span className="font-mono text-[10px] text-[var(--accent-primary)] tracking-wider">
              VAL: {hoveredCell.val.toFixed(3)}
            </span>
          )}
        </div>
      </div>

      {/* Canvas Viewport */}
      <div className="relative overflow-hidden cursor-crosshair rounded-sm border border-[var(--border-hairline)] bg-[var(--bg-canvas)]">
        <canvas
          ref={canvasRef}
          style={{ width: cols * cellWidth, height: rows * cellHeight }}
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
          onClick={handleClick}
          className="block"
        />

        {/* Ambient HUD Crosshairs */}
        <div className="pointer-events-none absolute top-2 left-2 font-mono text-[9px] text-[var(--text-muted)]">
          GRID [50m × 50m PU CUBE]
        </div>
        <div className="pointer-events-none absolute bottom-2 right-2 font-mono text-[9px] text-[var(--text-muted)]">
          EPSG:7851 / GA RADIOMETRICS
        </div>
      </div>

      {/* Sub-Matrix Legend */}
      <div className="w-full flex items-center justify-between pt-2.5 mt-2 border-t border-[var(--border-hairline)] text-[11px] font-mono text-[var(--text-secondary)]">
        <div className="flex items-center gap-2">
          <span className="text-[var(--text-muted)]">DENSITY:</span>
          <span className="tracking-widest text-[var(--text-primary)] font-bold">· : - = + * # % @ █</span>
        </div>
        <div className="flex items-center gap-3">
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-[var(--accent-primary)]"></span> Anomaly Peak
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-rose-500"></span> Locked Target
          </span>
        </div>
      </div>
    </div>
  );
};

// ============================================================================
// 4. STATISTICAL ANALYTICS COMPONENT (@VISX-STYLE HIGH ACUITY CHART)
// ============================================================================

interface AnalyticsChartProps {
  activeTarget: DrillTarget | null;
  onSelectTarget: (target: DrillTarget) => void;
  theme: ThemeMode;
}

export const PrecisionRecallChart: React.FC<AnalyticsChartProps> = ({
  activeTarget,
  onSelectTarget,
  theme
}) => {
  const [crosshairX, setCrosshairX] = useState<number | null>(null);

  const width = 520;
  const height = 240;
  const padding = { top: 20, right: 24, bottom: 36, left: 44 };

  const innerWidth = width - padding.left - padding.right;
  const innerHeight = height - padding.top - padding.bottom;

  // Generate synthetic PR curves for 3 models
  const pointsCount = 40;
  const data = useMemo(() => {
    const list = [];
    for (let i = 0; i <= pointsCount; i++) {
      const recall = i / pointsCount;
      // Fused Ensemble: PR-AUC ~ 0.912
      const precisionFused = Math.min(1, Math.max(0.1, 0.98 - Math.pow(recall, 3.4) * 0.42));
      // nnPU U-Net 3.2: PR-AUC ~ 0.884
      const precisionUnet = Math.min(1, Math.max(0.08, 0.95 - Math.pow(recall, 2.7) * 0.52));
      // XGBoost Baseline: PR-AUC ~ 0.847
      const precisionXgb = Math.min(1, Math.max(0.05, 0.91 - Math.pow(recall, 2.1) * 0.61));
      // Conformal lower bound (1 - alpha = 0.90)
      const confLower = Math.max(0.02, precisionFused - 0.082);

      list.push({ recall, precisionFused, precisionUnet, precisionXgb, confLower });
    }
    return list;
  }, []);

  const scaleX = (val: number) => padding.left + val * innerWidth;
  const scaleY = (val: number) => padding.top + (1 - val) * innerHeight;

  // SVG Paths
  const fusedPath = useMemo(() => {
    return data.map((d, i) => `${i === 0 ? 'M' : 'L'} ${scaleX(d.recall)} ${scaleY(d.precisionFused)}`).join(' ');
  }, [data]);

  const unetPath = useMemo(() => {
    return data.map((d, i) => `${i === 0 ? 'M' : 'L'} ${scaleX(d.recall)} ${scaleY(d.precisionUnet)}`).join(' ');
  }, [data]);

  const xgbPath = useMemo(() => {
    return data.map((d, i) => `${i === 0 ? 'M' : 'L'} ${scaleX(d.recall)} ${scaleY(d.precisionXgb)}`).join(' ');
  }, [data]);

  const confBandPath = useMemo(() => {
    const top = data.map((d, i) => `${i === 0 ? 'M' : 'L'} ${scaleX(d.recall)} ${scaleY(d.precisionFused)}`).join(' ');
    const bottom = data.slice().reverse().map((d) => `L ${scaleX(d.recall)} ${scaleY(d.confLower)}`).join(' ');
    return `${top} ${bottom} Z`;
  }, [data]);

  const handleMouseMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    if (x >= padding.left && x <= width - padding.right) {
      setCrosshairX(x);
    } else {
      setCrosshairX(null);
    }
  };

  const activeRecall = activeTarget ? 1 - activeTarget.epistemicUncertainty * 3.5 : 0.65;
  const targetX = scaleX(Math.max(0.1, Math.min(activeRecall, 0.95)));

  return (
    <div className="flex flex-col bg-[var(--bg-surface)] border border-[var(--border-hairline)] rounded-sm p-3">
      {/* Chart Title Header */}
      <div className="flex items-center justify-between pb-2 mb-2 border-b border-[var(--border-hairline)]">
        <div>
          <span className="font-display text-[12px] font-semibold text-[var(--text-primary)] tracking-tight">
            PRECISION-RECALL VALIDATION SPECTRUM
          </span>
          <span className="block font-editorial text-[12px] text-[var(--text-secondary)] italic">
            Leakage-resistant 5-fold spatial block CV with split-conformal coverage ($1-\alpha = 0.90$)
          </span>
        </div>
        <div className="flex items-center gap-2 font-mono text-[11px]">
          <span className="px-1.5 py-0.5 rounded bg-[var(--bg-elevated)] text-[var(--accent-primary)] font-semibold">
            PR-AUC: 0.912
          </span>
          <span className="px-1.5 py-0.5 rounded bg-[var(--bg-elevated)] text-emerald-400 font-semibold">
            GAIN: +7.6%
          </span>
        </div>
      </div>

      {/* SVG Chart Engine */}
      <div className="relative overflow-hidden">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          className="w-full h-auto cursor-crosshair select-none"
          onMouseMove={handleMouseMove}
          onMouseLeave={() => setCrosshairX(null)}
        >
          {/* Hairline Grid Lines */}
          {[0, 0.25, 0.5, 0.75, 1.0].map((tick) => (
            <g key={`grid-y-${tick}`}>
              <line
                x1={padding.left}
                y1={scaleY(tick)}
                x2={width - padding.right}
                y2={scaleY(tick)}
                stroke="var(--border-hairline)"
                strokeDasharray="2,3"
                strokeWidth={1}
              />
              <text
                x={padding.left - 8}
                y={scaleY(tick) + 3}
                fill="var(--text-muted)"
                fontSize={10}
                fontFamily="IBM Plex Mono"
                textAnchor="end"
              >
                {tick.toFixed(2)}
              </text>
            </g>
          ))}

          {[0, 0.25, 0.5, 0.75, 1.0].map((tick) => (
            <g key={`grid-x-${tick}`}>
              <line
                x1={scaleX(tick)}
                y1={padding.top}
                x2={scaleX(tick)}
                y2={height - padding.bottom}
                stroke="var(--border-hairline)"
                strokeDasharray="2,3"
                strokeWidth={1}
              />
              <text
                x={scaleX(tick)}
                y={height - padding.bottom + 14}
                fill="var(--text-muted)"
                fontSize={10}
                fontFamily="IBM Plex Mono"
                textAnchor="middle"
              >
                {tick.toFixed(2)}
              </text>
            </g>
          ))}

          {/* Random Baseline Prevalence Line (prevalence = 0.059) */}
          <line
            x1={padding.left}
            y1={scaleY(0.059)}
            x2={width - padding.right}
            y2={scaleY(0.059)}
            stroke="var(--text-muted)"
            strokeDasharray="4,4"
            strokeWidth={1}
          />
          <text
            x={width - padding.right - 4}
            y={scaleY(0.059) - 4}
            fill="var(--text-muted)"
            fontSize={9}
            fontFamily="IBM Plex Mono"
            textAnchor="end"
          >
            Prevalence Baseline (0.059)
          </text>

          {/* Conformal Uncertainty Fill Band */}
          <path d={confBandPath} fill="var(--accent-primary)" fillOpacity={0.08} />

          {/* XGBoost Baseline Line */}
          <path d={xgbPath} fill="none" stroke="#F59E0B" strokeWidth={1.4} strokeDasharray="3,2" />

          {/* nnPU U-Net Line */}
          <path d={unetPath} fill="none" stroke="#A855F7" strokeWidth={1.4} />

          {/* Fused Master Model Line */}
          <path d={fusedPath} fill="none" stroke="var(--accent-primary)" strokeWidth={2.4} />

          {/* Active Target Synchronized Crosshair Marker */}
          {activeTarget && (
            <g>
              <line
                x1={targetX}
                y1={padding.top}
                x2={targetX}
                y2={height - padding.bottom}
                stroke="#F43F5E"
                strokeWidth={1.5}
                strokeDasharray="3,3"
              />
              <circle
                cx={targetX}
                cy={scaleY(activeTarget.prospectivityScore)}
                r={4.5}
                fill="#F43F5E"
                stroke="var(--bg-canvas)"
                strokeWidth={2}
              />
            </g>
          )}

          {/* Dynamic Interactive Cursor Crosshair */}
          {crosshairX && (
            <line
              x1={crosshairX}
              y1={padding.top}
              x2={crosshairX}
              y2={height - padding.bottom}
              stroke="var(--text-primary)"
              strokeWidth={1}
              strokeDasharray="2,2"
              opacity={0.6}
            />
          )}

          {/* Axis Labels */}
          <text
            x={padding.left + innerWidth / 2}
            y={height - 6}
            fill="var(--text-secondary)"
            fontSize={10}
            fontFamily="Atkinson Hyperlegible"
            fontWeight={500}
            textAnchor="middle"
          >
            RECALL (POSITIVE DEPOSIT SENSITIVITY)
          </text>
          <text
            transform={`rotate(-90) translate(-${padding.top + innerHeight / 2}, 12)`}
            fill="var(--text-secondary)"
            fontSize={10}
            fontFamily="Atkinson Hyperlegible"
            fontWeight={500}
            textAnchor="middle"
          >
            PRECISION (PURITY)
          </text>
        </svg>
      </div>

      {/* Synchronized Legend & Model KPI Weights */}
      <div className="flex items-center justify-between pt-2.5 mt-2 border-t border-[var(--border-hairline)] text-[11px] font-mono">
        <div className="flex items-center gap-3">
          <span className="flex items-center gap-1.5 text-[var(--accent-primary)] font-semibold">
            <span className="w-2.5 h-0.5 bg-[var(--accent-primary)]"></span> Fused Ensemble (0.912)
          </span>
          <span className="flex items-center gap-1.5 text-purple-400">
            <span className="w-2.5 h-0.5 bg-purple-500"></span> nnPU U-Net (0.884)
          </span>
          <span className="flex items-center gap-1.5 text-amber-400">
            <span className="w-2.5 h-0.5 bg-amber-500"></span> XGBoost (0.847)
          </span>
        </div>
        <div className="text-[var(--text-muted)]">
          Elkan-Noto Prior <span className="text-[var(--text-primary)] font-semibold">ĉ = 0.285</span>
        </div>
      </div>
    </div>
  );
};

// ============================================================================
// 5. GEOSPATIAL VECTOR MAP VIEWPORT (CANVAS / HUD HYBRID)
// ============================================================================

interface MapViewportProps {
  activeTarget: DrillTarget | null;
  onSelectTarget: (target: DrillTarget) => void;
  layers: {
    mines: boolean;
    targets: boolean;
    faults: boolean;
    dilation: boolean;
  };
  theme: ThemeMode;
}

export const GeospatialVectorViewport: React.FC<MapViewportProps> = ({
  activeTarget,
  onSelectTarget,
  layers,
  theme
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [zoom, setZoom] = useState<number>(1.2);
  const [pulsePhase, setPulsePhase] = useState<number>(0);

  // Animation pulse loop for target sonars
  useEffect(() => {
    let frameId: number;
    const animate = (time: number) => {
      setPulsePhase((time % 2000) / 2000);
      frameId = requestAnimationFrame(animate);
    };
    frameId = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(frameId);
  }, []);

  // Canvas drawing
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    ctx.scale(dpr, dpr);

    const isDark = theme === 'dark';
    ctx.fillStyle = isDark ? '#090D16' : '#F1F5F9';
    ctx.fillRect(0, 0, w, h);

    // Subtle 50m Topo-Grid
    ctx.strokeStyle = isDark ? '#162032' : '#E2E8F0';
    ctx.lineWidth = 1;
    const gridSize = 32 * zoom;
    for (let x = 0; x < w; x += gridSize) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.stroke();
    }
    for (let y = 0; y < h; y += gridSize) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }

    // Draw GSWA Structural Shear Faults (Euclidean distance features)
    if (layers.faults) {
      ctx.strokeStyle = isDark ? '#475569' : '#94A3B8';
      ctx.lineWidth = 1.8;
      ctx.setLineDash([6, 4]);

      // Fault Line 1: Boulder-Lefroy Main Trend
      ctx.beginPath();
      ctx.moveTo(w * 0.15, h * 0.95);
      ctx.bezierCurveTo(w * 0.35, h * 0.65, w * 0.48, h * 0.45, w * 0.85, h * 0.1);
      ctx.stroke();

      // Fault Line 2: Zuleika Shear Splay
      ctx.beginPath();
      ctx.moveTo(w * 0.05, h * 0.8);
      ctx.bezierCurveTo(w * 0.25, h * 0.5, w * 0.4, h * 0.35, w * 0.65, h * 0.05);
      ctx.stroke();

      ctx.setLineDash([]);
    }

    // Draw Dilation Stress Heatmap Blurs
    if (layers.dilation) {
      const grad = ctx.createRadialGradient(w * 0.48, h * 0.45, 10, w * 0.48, h * 0.45, 160 * zoom);
      grad.addColorStop(0, isDark ? 'rgba(56, 189, 248, 0.22)' : 'rgba(2, 132, 199, 0.18)');
      grad.addColorStop(0.6, isDark ? 'rgba(56, 189, 248, 0.06)' : 'rgba(2, 132, 199, 0.05)');
      grad.addColorStop(1, 'transparent');
      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, w, h);
    }

    // Draw Ground-Truth MINEDEX Mines (Small green diamond beacons)
    if (layers.mines) {
      ctx.fillStyle = '#10B981';
      const minePoints = [
        { x: w * 0.42, y: h * 0.52 },
        { x: w * 0.46, y: h * 0.48 },
        { x: w * 0.51, y: h * 0.41 },
        { x: w * 0.38, y: h * 0.58 },
        { x: w * 0.62, y: h * 0.28 }
      ];
      minePoints.forEach((pt) => {
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 3, 0, Math.PI * 2);
        ctx.fill();
      });
    }

    // Draw Target Polygons & Animated Sonar Beacons
    if (layers.targets) {
      TARGET_DATASET.forEach((t, i) => {
        // Map target easting/northing to canvas coords
        const tx = w * (0.28 + i * 0.12 + Math.sin(i * 1.5) * 0.06);
        const ty = h * (0.75 - i * 0.13 + Math.cos(i * 1.8) * 0.05);

        const isSelected = activeTarget && activeTarget.id === t.id;

        // Draw animated sonar ripple
        const ringRadius = 8 + pulsePhase * 24;
        ctx.strokeStyle = isSelected ? '#F43F5E' : (isDark ? '#38BDF8' : '#0284C7');
        ctx.lineWidth = isSelected ? 2 : 1.2;
        ctx.globalAlpha = Math.max(0, 1 - pulsePhase);
        ctx.beginPath();
        ctx.arc(tx, ty, ringRadius, 0, Math.PI * 2);
        ctx.stroke();
        ctx.globalAlpha = 1.0;

        // Core target point
        ctx.fillStyle = isSelected ? '#F43F5E' : (isDark ? '#38BDF8' : '#0284C7');
        ctx.beginPath();
        ctx.arc(tx, ty, isSelected ? 5.5 : 4, 0, Math.PI * 2);
        ctx.fill();

        // Target Callout Label
        ctx.font = '10px "IBM Plex Mono", monospace';
        ctx.fillStyle = isDark ? '#F8FAFC' : '#0F172A';
        ctx.textAlign = 'left';
        ctx.fillText(`${t.id} (${(t.prospectivityScore * 100).toFixed(1)}%)`, tx + 9, ty + 3);
      });
    }
  }, [layers, activeTarget, zoom, pulsePhase, theme]);

  return (
    <div className="relative w-full h-[360px] bg-[var(--bg-canvas)] border border-[var(--border-hairline)] rounded-sm overflow-hidden select-none">
      <canvas ref={canvasRef} className="w-full h-full block cursor-grab active:cursor-grabbing" />

      {/* Floating Tactical HUD Header */}
      <div className="absolute top-3 left-3 flex items-center gap-2 px-2.5 py-1.5 rounded bg-[var(--bg-surface)]/90 backdrop-blur-md border border-[var(--border-hairline)] shadow-sm">
        <Radio className="w-3.5 h-3.5 text-[var(--accent-primary)] animate-pulse" />
        <span className="font-display text-[11px] font-semibold text-[var(--text-primary)]">
          YILGARN CRATON // 50m SYNTHESIS
        </span>
        <span className="font-mono text-[9px] text-[var(--text-muted)] border-l border-[var(--border-hairline)] pl-2">
          EPSG:7851 [MGA ZONE 51]
        </span>
      </div>

      {/* Zoom Controls HUD */}
      <div className="absolute bottom-3 right-3 flex flex-col gap-1 bg-[var(--bg-surface)]/90 backdrop-blur-md border border-[var(--border-hairline)] rounded p-1 shadow-sm">
        <button
          onClick={() => setZoom((z) => Math.min(z + 0.2, 2.4))}
          className="w-6 h-6 flex items-center justify-center font-mono text-[13px] text-[var(--text-primary)] hover:bg-[var(--bg-elevated)] rounded"
        >
          +
        </button>
        <button
          onClick={() => setZoom((z) => Math.max(z - 0.2, 0.6))}
          className="w-6 h-6 flex items-center justify-center font-mono text-[13px] text-[var(--text-primary)] hover:bg-[var(--bg-elevated)] rounded"
        >
          -
        </button>
      </div>

      {/* Crosshair Viewport Watermark */}
      <div className="pointer-events-none absolute bottom-3 left-3 font-mono text-[9.5px] text-[var(--text-muted)] flex items-center gap-3">
        <span>LAT: -30.749° / LON: 121.465°</span>
        <span>ELEV: 360m ASL</span>
      </div>
    </div>
  );
};

// ============================================================================
// 6. MAIN WORKSPACE CONTAINER (SPLIT-PANE + COLLAPSIBLE PROGRESSIVE DISCLOSURE)
// ============================================================================

export default function GeospatialWorkspace() {
  const [theme, setTheme] = useState<ThemeMode>('dark');
  const [drawerOpen, setDrawerOpen] = useState<boolean>(true);
  const [zDepth, setZDepth] = useState<number>(380);
  const [activeTarget, setActiveTarget] = useState<DrillTarget | null>(TARGET_DATASET[0]);
  const [brailleIndex, setBrailleIndex] = useState<number>(0);

  // Active layers state
  const [layers, setLayers] = useState({
    mines: true,
    targets: true,
    faults: true,
    dilation: true
  });

  // Cycle ASCII braille stream in header
  useEffect(() => {
    const timer = setInterval(() => {
      setBrailleIndex((idx) => (idx + 1) % BRAILLE_FRAMES.length);
    }, 120);
    return () => clearInterval(timer);
  }, []);

  // Scramble numerical readouts when activeTarget switches
  const scrambledScore = useGlyphScramble(
    activeTarget ? (activeTarget.prospectivityScore * 100).toFixed(1) + '%' : '94.2%',
    220
  );
  const scrambledDepth = useGlyphScramble(activeTarget ? `-${activeTarget.depthMeters}m` : '-380m', 180);
  const scrambledUncertainty = useGlyphScramble(
    activeTarget ? `±${(activeTarget.epistemicUncertainty * 100).toFixed(1)}%` : '±4.8%',
    200
  );

  const toggleTheme = () => {
    setTheme((t) => (t === 'dark' ? 'light' : 'dark'));
  };

  return (
    <div
      data-theme={theme}
      className={`min-h-screen w-full flex flex-col font-sans transition-colors duration-200 ${
        theme === 'dark'
          ? 'bg-[#090D16] text-[#F8FAFC]'
          : 'bg-[#F8FAFC] text-[#0F172A]'
      }`}
      style={{
        // Dual-theme CSS variable bindings matching specification
        ['--bg-canvas' as any]: theme === 'dark' ? '#090D16' : '#F8FAFC',
        ['--bg-surface' as any]: theme === 'dark' ? '#0F1624' : '#FFFFFF',
        ['--bg-elevated' as any]: theme === 'dark' ? '#162032' : '#F1F5F9',
        ['--border-hairline' as any]: theme === 'dark' ? '#1E293B' : '#E2E8F0',
        ['--text-primary' as any]: theme === 'dark' ? '#F8FAFC' : '#0F172A',
        ['--text-secondary' as any]: theme === 'dark' ? '#94A3B8' : '#475569',
        ['--text-muted' as any]: theme === 'dark' ? '#475569' : '#94A3B8',
        ['--accent-primary' as any]: theme === 'dark' ? '#38BDF8' : '#0284C7',
        ['--accent-secondary' as any]: theme === 'dark' ? '#0EA5E9' : '#0369A1'
      }}
    >
      {/* ======================================================================
          TOP HEADER & TELEMETRY STREAM BAR
          ====================================================================== */}
      <header className="h-12 border-b border-[var(--border-hairline)] bg-[var(--bg-surface)] px-4 flex items-center justify-between sticky top-0 z-50">
        {/* Left Brand Unit */}
        <div className="flex items-center gap-3">
          <div className="w-6 h-6 rounded bg-[var(--bg-elevated)] border border-[var(--border-hairline)] flex items-center justify-center text-[var(--accent-primary)]">
            <Crosshair className="w-3.5 h-3.5" />
          </div>
          <div className="flex items-baseline gap-2">
            <span className="font-display font-semibold text-[13px] tracking-tight text-[var(--text-primary)]">
              ANTIGRAVITY // GEOSPATIALS
            </span>
            <span className="font-mono text-[9px] text-[var(--accent-primary)] uppercase px-1.5 py-0.5 rounded bg-[var(--bg-elevated)]">
              MPM PU 7-PASS
            </span>
          </div>
        </div>

        {/* Center Live Braille Telemetry Stream */}
        <div className="hidden md:flex items-center gap-4 font-mono text-[11px] text-[var(--text-secondary)]">
          <div className="flex items-center gap-2">
            <span className="text-[var(--accent-primary)] font-bold text-[13px]">
              {BRAILLE_FRAMES[brailleIndex]}
            </span>
            <span>GA-STREAM: INGESTING MGA51 TENSORS</span>
          </div>
          <span className="text-[var(--border-hairline)]">|</span>
          <div className="flex items-center gap-1.5">
            <span className="text-[var(--text-muted)]">FUSED PR-AUC:</span>
            <span className="text-emerald-400 font-semibold">0.912</span>
          </div>
          <span className="text-[var(--border-hairline)]">|</span>
          <div className="flex items-center gap-1.5">
            <span className="text-[var(--text-muted)]">CONFORMAL COVERAGE:</span>
            <span className="text-[var(--text-primary)]">90.2%</span>
          </div>
        </div>

        {/* Right Action Tools */}
        <div className="flex items-center gap-2">
          {/* Dataset Selector Dropdown */}
          <select className="bg-[var(--bg-elevated)] border border-[var(--border-hairline)] rounded text-[11px] font-sans px-2.5 py-1 text-[var(--text-primary)] outline-none">
            <option>YILGARN CRATON (WA) — 50m Orogenic Gold</option>
            <option>BORNEO HD (50m) — Kelian Epithermal Au-Ag</option>
            <option>KALIMANTAN ISLAND — Regional Reconnaissance</option>
          </select>

          {/* Theme Toggle Button */}
          <button
            onClick={toggleTheme}
            className="w-8 h-8 rounded border border-[var(--border-hairline)] bg-[var(--bg-elevated)] flex items-center justify-center text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors"
            title="Toggle Monochromatic Theme"
          >
            {theme === 'dark' ? <Sun className="w-3.5 h-3.5" /> : <Moon className="w-3.5 h-3.5" />}
          </button>
        </div>
      </header>

      {/* ======================================================================
          MAIN SPLIT-PANE WORKSPACE BODY
          ====================================================================== */}
      <div className="flex-1 flex overflow-hidden">
        {/* COLLAPSIBLE PROGRESSIVE DISCLOSURE DRAWER (280px / 44px) */}
        <aside
          className={`border-r border-[var(--border-hairline)] bg-[var(--bg-surface)] transition-all duration-200 flex flex-col ${
            drawerOpen ? 'w-[280px]' : 'w-[44px]'
          }`}
        >
          {/* Drawer Toggle Header */}
          <div className="h-10 border-b border-[var(--border-hairline)] px-2.5 flex items-center justify-between">
            {drawerOpen && (
              <span className="font-display font-semibold text-[11px] text-[var(--text-secondary)] uppercase tracking-wider">
                CONTROL DRAWER
              </span>
            )}
            <button
              onClick={() => setDrawerOpen(!drawerOpen)}
              className="w-6 h-6 rounded flex items-center justify-center hover:bg-[var(--bg-elevated)] text-[var(--text-secondary)] hover:text-[var(--text-primary)] mx-auto"
            >
              {drawerOpen ? <ChevronLeft className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
            </button>
          </div>

          {/* Expanded Drawer Content */}
          {drawerOpen ? (
            <div className="flex-1 overflow-y-auto p-4 space-y-6">
              {/* Z-Depth Tomography Slider */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <label className="font-sans text-[11px] font-semibold text-[var(--text-primary)] uppercase tracking-tight">
                    Z-DEPTH TOMOGRAPHY
                  </label>
                  <span className="font-mono text-[11px] text-[var(--accent-primary)] font-semibold">
                    -{zDepth} m
                  </span>
                </div>
                <input
                  type="range"
                  min={50}
                  max={1000}
                  step={10}
                  value={zDepth}
                  onChange={(e) => setZDepth(Number(e.target.value))}
                  className="w-full accent-[var(--accent-primary)] cursor-pointer"
                />
                <p className="font-editorial text-[12px] text-[var(--text-secondary)] italic leading-snug">
                  Scrubbing vertical slice through 16-band geophysics cube. Modulates fluid shear trapping in the ASCII tomogram.
                </p>
              </div>

              {/* Sensor Layer Toggles */}
              <div className="space-y-2.5 pt-3 border-t border-[var(--border-hairline)]">
                <span className="font-sans text-[11px] font-semibold text-[var(--text-primary)] uppercase tracking-tight">
                  EVIDENCE LAYERS
                </span>

                <label className="flex items-center justify-between text-[12px] font-sans text-[var(--text-secondary)] hover:text-[var(--text-primary)] cursor-pointer">
                  <span>MINEDEX Deposit Labels</span>
                  <input
                    type="checkbox"
                    checked={layers.mines}
                    onChange={(e) => setLayers({ ...layers, mines: e.target.checked })}
                    className="accent-[var(--accent-primary)]"
                  />
                </label>

                <label className="flex items-center justify-between text-[12px] font-sans text-[var(--text-secondary)] hover:text-[var(--text-primary)] cursor-pointer">
                  <span>Ranked Target Polygons</span>
                  <input
                    type="checkbox"
                    checked={layers.targets}
                    onChange={(e) => setLayers({ ...layers, targets: e.target.checked })}
                    className="accent-[var(--accent-primary)]"
                  />
                </label>

                <label className="flex items-center justify-between text-[12px] font-sans text-[var(--text-secondary)] hover:text-[var(--text-primary)] cursor-pointer">
                  <span>GSWA Shear Faults</span>
                  <input
                    type="checkbox"
                    checked={layers.faults}
                    onChange={(e) => setLayers({ ...layers, faults: e.target.checked })}
                    className="accent-[var(--accent-primary)]"
                  />
                </label>

                <label className="flex items-center justify-between text-[12px] font-sans text-[var(--text-secondary)] hover:text-[var(--text-primary)] cursor-pointer">
                  <span>3D Dilation Stress</span>
                  <input
                    type="checkbox"
                    checked={layers.dilation}
                    onChange={(e) => setLayers({ ...layers, dilation: e.target.checked })}
                    className="accent-[var(--accent-primary)]"
                  />
                </label>
              </div>

              {/* Statistical Calibration Note */}
              <div className="p-3 rounded bg-[var(--bg-elevated)] border border-[var(--border-hairline)] space-y-1">
                <span className="font-mono text-[10px] text-[var(--accent-primary)] uppercase font-semibold">
                  PU LEARNING GUARANTEE
                </span>
                <p className="font-editorial text-[12.5px] text-[var(--text-secondary)] italic leading-relaxed">
                  Negative unlabelled mining via Isolation Forest prevents penalizing unexplored ground. Leakage-proof spatial block CV ensures zero fold bleed.
                </p>
              </div>
            </div>
          ) : (
            /* Collapsed Icon Rail */
            <div className="flex-1 flex flex-col items-center py-4 gap-4 text-[var(--text-secondary)]">
              <button
                onClick={() => setDrawerOpen(true)}
                className="p-1.5 rounded hover:bg-[var(--bg-elevated)]"
                title="Open Controls"
              >
                <Sliders className="w-4 h-4" />
              </button>
              <button
                onClick={() => setDrawerOpen(true)}
                className="p-1.5 rounded hover:bg-[var(--bg-elevated)]"
                title="Layers"
              >
                <Layers className="w-4 h-4" />
              </button>
              <button
                onClick={() => setDrawerOpen(true)}
                className="p-1.5 rounded hover:bg-[var(--bg-elevated)]"
                title="Telemetry"
              >
                <Activity className="w-4 h-4" />
              </button>
            </div>
          )}
        </aside>

        {/* WORKSPACE VIEWPORT & TELEMETRY PANELS */}
        <main className="flex-1 flex flex-col overflow-y-auto p-4 gap-4">
          {/* Top Analytical Grid: Geospatial Vector Viewport + Kinetic KPI Cards */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            {/* Geospatial Map Viewport (Takes 2 Columns on desktop) */}
            <div className="lg:col-span-2 flex flex-col gap-2">
              <GeospatialVectorViewport
                activeTarget={activeTarget}
                onSelectTarget={setActiveTarget}
                layers={layers}
                theme={theme}
              />
            </div>

            {/* Kinetic Active Target Telemetry HUD Card */}
            <div className="flex flex-col justify-between bg-[var(--bg-surface)] border border-[var(--border-hairline)] rounded-sm p-4">
              <div>
                <div className="flex items-center justify-between pb-2 border-b border-[var(--border-hairline)]">
                  <span className="font-display text-[12px] font-semibold text-[var(--text-primary)]">
                    SELECTED DRILL TARGET
                  </span>
                  <span className="font-mono text-[10px] text-rose-400 bg-rose-500/10 px-1.5 py-0.5 rounded font-semibold">
                    {activeTarget?.classification || 'TIER-1'}
                  </span>
                </div>

                <div className="mt-3 space-y-3">
                  <div>
                    <span className="font-sans text-[11px] text-[var(--text-muted)] uppercase">Target Identifier</span>
                    <h3 className="font-display text-[15px] font-semibold text-[var(--text-primary)]">
                      {activeTarget?.name || 'Select Target'}
                    </h3>
                    <span className="font-mono text-[10px] text-[var(--text-secondary)]">
                      {activeTarget?.id} // {activeTarget?.structuralDomain}
                    </span>
                  </div>

                  {/* Scrambled High-Acuity KPIs */}
                  <div className="grid grid-cols-2 gap-2 pt-2">
                    <div className="p-2.5 rounded bg-[var(--bg-elevated)] border border-[var(--border-hairline)]">
                      <span className="font-sans text-[10px] text-[var(--text-muted)] block">PROSPECTIVITY</span>
                      <span className="font-mono text-[18px] font-bold text-[var(--accent-primary)] tracking-tight">
                        {scrambledScore}
                      </span>
                    </div>

                    <div className="p-2.5 rounded bg-[var(--bg-elevated)] border border-[var(--border-hairline)]">
                      <span className="font-sans text-[10px] text-[var(--text-muted)] block">TARGET DEPTH</span>
                      <span className="font-mono text-[18px] font-bold text-[var(--text-primary)] tracking-tight">
                        {scrambledDepth}
                      </span>
                    </div>

                    <div className="p-2.5 rounded bg-[var(--bg-elevated)] border border-[var(--border-hairline)]">
                      <span className="font-sans text-[10px] text-[var(--text-muted)] block">EPISTEMIC UNCERTAINTY</span>
                      <span className="font-mono text-[13px] font-semibold text-emerald-400">
                        {scrambledUncertainty}
                      </span>
                    </div>

                    <div className="p-2.5 rounded bg-[var(--bg-elevated)] border border-[var(--border-hairline)]">
                      <span className="font-sans text-[10px] text-[var(--text-muted)] block">CONFORMAL INTERVAL</span>
                      <span className="font-mono text-[12px] font-semibold text-[var(--text-secondary)]">
                        [{activeTarget?.conformalInterval[0].toFixed(2)}, {activeTarget?.conformalInterval[1].toFixed(2)}]
                      </span>
                    </div>
                  </div>

                  <div className="pt-2">
                    <span className="font-sans text-[10px] text-[var(--text-muted)] block uppercase">Primary Geological Driver</span>
                    <p className="font-editorial text-[13px] text-[var(--text-secondary)] italic">
                      "{activeTarget?.primaryDriver}"
                    </p>
                  </div>
                </div>
              </div>

              {/* Dispatch Action */}
              <button className="w-full mt-4 py-2 px-3 rounded bg-[var(--accent-primary)] hover:opacity-90 text-[var(--bg-canvas)] font-sans text-[12px] font-semibold flex items-center justify-center gap-2 transition-opacity">
                <Crosshair className="w-3.5 h-3.5" />
                Dispatch Rig to Collar Coordinates
              </button>
            </div>
          </div>

          {/* Mid Section: Interactive Procedural ASCII Subsurface Matrix + @visx PR-AUC Chart */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <AsciiSubsurfaceCanvas
              zDepth={zDepth}
              activeTarget={activeTarget}
              onSelectTarget={setActiveTarget}
              theme={theme}
            />

            <PrecisionRecallChart
              activeTarget={activeTarget}
              onSelectTarget={setActiveTarget}
              theme={theme}
            />
          </div>

          {/* Bottom Analytical Table: Synchronized Cross-Pane Targets */}
          <div className="bg-[var(--bg-surface)] border border-[var(--border-hairline)] rounded-sm p-4">
            <div className="flex items-center justify-between pb-3 mb-2 border-b border-[var(--border-hairline)]">
              <div>
                <span className="font-display text-[13px] font-semibold text-[var(--text-primary)]">
                  DISPATCH-VERIFIED DRILL TARGET CALLSET
                </span>
                <span className="block font-editorial text-[12px] text-[var(--text-secondary)] italic">
                  Ranked vector candidates derived from 50m PU feature cube with spatial CV holdouts
                </span>
              </div>
              <span className="font-mono text-[10px] text-[var(--accent-primary)] px-2 py-0.5 rounded bg-[var(--bg-elevated)]">
                5 TARGETS SYNCHRONIZED
              </span>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left font-sans text-[12px]">
                <thead>
                  <tr className="border-b border-[var(--border-hairline)] text-[var(--text-muted)] font-mono text-[10.5px]">
                    <th className="py-2 px-3">TARGET ID</th>
                    <th className="py-2 px-3">PROSPECT / LOCATION</th>
                    <th className="py-2 px-3">EASTING (m)</th>
                    <th className="py-2 px-3">NORTHING (m)</th>
                    <th className="py-2 px-3">DEPTH</th>
                    <th className="py-2 px-3">SCORE</th>
                    <th className="py-2 px-3">UNCERTAINTY</th>
                    <th className="py-2 px-3">STATUS</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[var(--border-hairline)]">
                  {TARGET_DATASET.map((target) => {
                    const isSelected = activeTarget?.id === target.id;
                    return (
                      <tr
                        key={target.id}
                        onClick={() => setActiveTarget(target)}
                        className={`cursor-pointer transition-colors ${
                          isSelected
                            ? 'bg-[var(--bg-elevated)] text-[var(--accent-primary)] font-medium'
                            : 'hover:bg-[var(--bg-elevated)] text-[var(--text-primary)]'
                        }`}
                      >
                        <td className="py-2.5 px-3 font-mono text-[11px] font-semibold">
                          {target.id}
                        </td>
                        <td className="py-2.5 px-3">
                          {target.name}
                        </td>
                        <td className="py-2.5 px-3 font-mono text-[11px] text-[var(--text-secondary)]">
                          {target.easting}
                        </td>
                        <td className="py-2.5 px-3 font-mono text-[11px] text-[var(--text-secondary)]">
                          {target.northing}
                        </td>
                        <td className="py-2.5 px-3 font-mono text-[11px] text-[var(--text-secondary)]">
                          -{target.depthMeters} m
                        </td>
                        <td className="py-2.5 px-3 font-mono text-[11px] font-bold text-[var(--accent-primary)]">
                          {(target.prospectivityScore * 100).toFixed(1)}%
                        </td>
                        <td className="py-2.5 px-3 font-mono text-[11px] text-emerald-400">
                          ±{(target.epistemicUncertainty * 100).toFixed(1)}%
                        </td>
                        <td className="py-2.5 px-3">
                          <span
                            className={`inline-block px-1.5 py-0.5 rounded text-[9.5px] font-mono ${
                              target.classification === 'TIER-1 CORE'
                                ? 'bg-rose-500/10 text-rose-400'
                                : 'bg-[var(--bg-elevated)] text-[var(--text-secondary)]'
                            }`}
                          >
                            {target.classification}
                          </span>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}
