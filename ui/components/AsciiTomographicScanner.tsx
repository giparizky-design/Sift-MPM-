import React, { useRef, useEffect, useState, useMemo, useCallback } from 'react';
import { DrillTarget, TARGET_DATASET, ASCII_GLYPH_RAMP, ThemeMode } from '../types';
import { useScrambledValue } from '../hooks/useScrambledValue';

interface AsciiTomographicScannerProps {
  zDepth: number;
  onZDepthChange?: (depth: number) => void;
  activeTarget: DrillTarget | null;
  onSelectTarget: (target: DrillTarget) => void;
  hoveredTargetId?: string | null;
  onHoverTarget?: (targetId: string | null) => void;
  theme?: ThemeMode;
}

interface AnomalyFieldNode {
  id: string;
  gridX: number;
  gridY: number;
  targetDepth: number;
  intensity: number;
  name: string;
}

export const AsciiTomographicScanner: React.FC<AsciiTomographicScannerProps> = ({
  zDepth,
  onZDepthChange,
  activeTarget,
  onSelectTarget,
  hoveredTargetId,
  onHoverTarget,
  theme = 'dark'
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [mousePos, setMousePos] = useState<{ x: number; y: number } | null>(null);
  const [lensMetric, setLensMetric] = useState<string>('PROXIMITY: STANDBY');

  // Mapped anomalies based on targets in Yilgarn dataset
  const anomalyNodes = useMemo<AnomalyFieldNode[]>(() => [
    { id: 'TGT-YIL-01', gridX: 14, gridY: 6, targetDepth: 380, intensity: 0.942, name: 'Boulder-Lefroy Node A' },
    { id: 'TGT-YIL-02', gridX: 28, gridY: 10, targetDepth: 520, intensity: 0.887, name: 'Zuleika Shear Jog' },
    { id: 'TGT-YIL-03', gridX: 40, gridY: 4, targetDepth: 290, intensity: 0.915, name: 'Laverton Crest' },
    { id: 'TGT-YIL-04', gridX: 9, gridY: 13, targetDepth: 640, intensity: 0.794, name: 'Ida Splay' },
    { id: 'TGT-YIL-05', gridX: 34, gridY: 14, targetDepth: 410, intensity: 0.832, name: 'Kurnalpi Diatreme' }
  ], []);

  const cols = 52;
  const rows = 18;
  const glyphs = useMemo(() => Array.from(ASCII_GLYPH_RAMP), []);

  const scrambledZ = useScrambledValue(`-${zDepth}m`, 180);

  // Render loop for Tomographic Monospace ASCII Canvas
  const render = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    const cellW = width / cols;
    const cellH = height / rows;

    // Background clearing based on theme tokens
    const isLight = theme === 'light';
    ctx.fillStyle = isLight ? '#F8FAFC' : '#090D16';
    ctx.fillRect(0, 0, width, height);

    // Font configuration: Precision Telemetry Monospace
    ctx.font = '500 11.5px "IBM Plex Mono", ui-monospace, monospace';
    ctx.textBaseline = 'middle';
    ctx.textAlign = 'center';

    const normZ = zDepth / 1000.0;
    let closestNode: AnomalyFieldNode | null = null;
    let minNodeDist = 999;

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const px = c * cellW + cellW / 2;
        const py = r * cellH + cellH / 2;

        // Base regional tectonic potential field (harmonic wave synthesis)
        let density = 
          0.12 * Math.sin(c * 0.28 + normZ * 3.8) +
          0.10 * Math.cos(r * 0.38 - normZ * 2.6) +
          0.06 * Math.sin((c + r) * 0.18);

        let isTargetPeak = false;
        let isHoveredNode = false;
        let isSelectedNode = false;

        // Superimpose subsurface anomaly cores
        for (const node of anomalyNodes) {
          const dx = c - node.gridX;
          const dy = r - node.gridY;
          const distSq = dx * dx + dy * dy;
          const dz = Math.abs(normZ - node.targetDepth / 1000.0);

          // 3D Gaussian falloff
          const influence = Math.exp(-distSq / 14.0) * Math.max(0, 1 - dz * 2.4);
          density += influence * node.intensity;

          if (distSq < 1.8 && dz < 0.22) {
            isTargetPeak = true;
            if (activeTarget && activeTarget.id === node.id) isSelectedNode = true;
            if (hoveredTargetId === node.id) isHoveredNode = true;
          }
        }

        // Magnetic lens / ripple proximity calculation
        let lensProximity = 0;
        if (mousePos) {
          const mdx = px - mousePos.x;
          const mdy = py - mousePos.y;
          const mDist = Math.hypot(mdx, mdy);
          const lensRadius = 68;

          if (mDist < lensRadius) {
            lensProximity = 1 - mDist / lensRadius;
            // Lens ripple resolves higher subsurface density
            density += lensProximity * 0.36;
          }
        }

        density = Math.max(0, Math.min(density, 0.999));
        const glyphIdx = Math.floor(density * (glyphs.length - 1));
        const glyph = glyphs[glyphIdx];

        // Kinetic Coloring & Styling
        if (isSelectedNode) {
          ctx.fillStyle = isLight ? '#0284C7' : '#38BDF8';
          ctx.fillText('◈', px, py);
        } else if (isHoveredNode) {
          ctx.fillStyle = '#F59E0B'; // Amber sync highlight
          ctx.fillText('◈', px, py);
        } else if (isTargetPeak) {
          ctx.fillStyle = isLight ? '#0369A1' : '#38BDF8';
          ctx.fillText('◈', px, py);
        } else if (lensProximity > 0.5) {
          ctx.fillStyle = isLight ? '#0F172A' : '#F8FAFC';
          ctx.fillText(glyph, px, py);
        } else if (density > 0.52) {
          ctx.fillStyle = isLight ? '#334155' : '#94A3B8';
          ctx.fillText(glyph, px, py);
        } else {
          ctx.fillStyle = isLight ? '#94A3B8' : '#334155';
          ctx.fillText(glyph, px, py);
        }
      }
    }

    // Check closest anomaly node to mouse
    if (mousePos) {
      for (const node of anomalyNodes) {
        const npx = node.gridX * cellW + cellW / 2;
        const npy = node.gridY * cellH + cellH / 2;
        const d = Math.hypot(npx - mousePos.x, npy - mousePos.y);
        if (d < minNodeDist) {
          minNodeDist = d;
          closestNode = node;
        }
      }

      if (closestNode && minNodeDist < 42) {
        setLensMetric(`LOCKED: ${closestNode.id} [${closestNode.name.toUpperCase()}]`);
        if (onHoverTarget && hoveredTargetId !== closestNode.id) {
          onHoverTarget(closestNode.id);
        }
      } else {
        const valEstimate = (0.42 + (1 - minNodeDist / 300) * 0.5).toFixed(3);
        setLensMetric(`RESOLVED // POTENTIAL: ${valEstimate} // DEPTH: -${zDepth}m`);
        if (onHoverTarget && hoveredTargetId) {
          onHoverTarget(null);
        }
      }
    }
  }, [theme, zDepth, anomalyNodes, mousePos, glyphs, activeTarget, hoveredTargetId, onHoverTarget]);

  useEffect(() => {
    render();
  }, [render]);

  // Handle Mouse movement on canvas
  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) * (canvas.width / rect.width);
    const y = (e.clientY - rect.top) * (canvas.height / rect.height);
    setMousePos({ x, y });
  };

  const handleMouseLeave = () => {
    setMousePos(null);
    setLensMetric('PROXIMITY: STANDBY');
    if (onHoverTarget) onHoverTarget(null);
  };

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) * (canvas.width / rect.width);
    const y = (e.clientY - rect.top) * (canvas.height / rect.height);

    const cellW = canvas.width / cols;
    const cellH = canvas.height / rows;

    for (const node of anomalyNodes) {
      const npx = node.gridX * cellW + cellW / 2;
      const npy = node.gridY * cellH + cellH / 2;
      if (Math.hypot(npx - x, npy - y) < 42) {
        const targetObj = TARGET_DATASET.find(t => t.id === node.id);
        if (targetObj) {
          onSelectTarget(targetObj);
        }
        break;
      }
    }
  };

  return (
    <div 
      ref={containerRef}
      className="flex flex-col bg-[var(--bg-surface)] border border-[var(--border-hairline)] rounded p-5 transition-all duration-200"
    >
      {/* Header bar: Analytical Display & Editorial typography */}
      <div className="flex flex-wrap items-center justify-between pb-3 mb-3 border-b border-[var(--border-hairline)] gap-3">
        <div className="flex items-center gap-2.5">
          <span className="font-mono text-[11px] font-bold text-[var(--accent-primary)] bg-[var(--bg-elevated)] px-1.5 py-0.5 rounded border border-[var(--border-hairline)]">
            01
          </span>
          <h2 className="font-display text-[13px] font-semibold tracking-[-0.02em] text-[var(--text-primary)]">
            SUB-SURFACE TOMOGRAPHIC ASCII SCANNER
          </h2>
          <span className="font-mono text-[10px] text-[var(--text-muted)] border-l border-[var(--border-hairline)] pl-2">
            52×18 GRID // GA TMI-1VD & RADIOMETRICS
          </span>
        </div>

        {/* Z-Depth Scrubber Control */}
        <div className="flex items-center gap-3">
          <span className="font-editorial text-[13px] italic text-[var(--text-secondary)]">
            Z-Depth Slice:
          </span>
          <span className="font-mono text-[12px] font-bold text-[var(--accent-primary)] tabular-nums slashed-zero">
            {scrambledZ}
          </span>
          {onZDepthChange && (
            <input
              type="range"
              min={50}
              max={1000}
              step={10}
              value={zDepth}
              onChange={(e) => onZDepthChange(parseInt(e.target.value, 10))}
              className="w-28 accent-[var(--accent-primary)] cursor-pointer"
              title="Scrub Subsurface Z-Depth"
            />
          )}
        </div>
      </div>

      {/* Main ASCII Canvas Viewport */}
      <div className="relative w-full overflow-hidden bg-[var(--bg-canvas)] rounded border border-[var(--border-hairline)] cursor-crosshair">
        <canvas
          ref={canvasRef}
          width={728}
          height={260}
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
          onClick={handleCanvasClick}
          className="w-full h-auto block select-none"
        />

        {/* Precision Telemetry Monospace HUD Tags */}
        <div className="pointer-events-none absolute top-2.5 left-3 font-mono text-[10px] tracking-tight text-[var(--text-muted)]">
          TOMOGRAPHY HARMONICS // 3D GAUSSIAN FALLOFF
        </div>
        <div className="pointer-events-none absolute bottom-2.5 right-3 font-mono text-[10.5px] tracking-tight font-semibold text-[var(--accent-primary)] bg-[var(--bg-surface)]/80 px-2 py-0.5 rounded border border-[var(--border-hairline)]">
          {lensMetric}
        </div>
      </div>

      {/* Footer Legend & Mathematical Annotation */}
      <div className="flex flex-wrap items-center justify-between pt-3 mt-3 border-t border-[var(--border-hairline)] text-[11px] font-mono text-[var(--text-secondary)] gap-2">
        <div className="flex items-center gap-2">
          <span className="text-[var(--text-muted)]">DENSITY RAMP:</span>
          <span className="text-[var(--text-primary)] font-bold tracking-widest">
            · : - = + * # % @ ◈
          </span>
        </div>
        <div className="flex items-center gap-4">
          <span className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-[var(--accent-primary)]"></span>
            Peak Anomaly Target
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-amber-400"></span>
            Cross-Pane Focus
          </span>
          <span className="font-editorial italic text-[12px] text-[var(--text-muted)]">
            p(y=1|x) ≥ 1 - α
          </span>
        </div>
      </div>
    </div>
  );
};
