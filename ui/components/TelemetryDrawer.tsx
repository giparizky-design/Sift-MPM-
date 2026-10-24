'use client';

import React, { useState } from 'react';
import { 
  ChevronLeft, 
  ChevronRight, 
  Sliders, 
  Layers, 
  Activity, 
  Filter, 
  RotateCcw 
} from 'lucide-react';
import { ThemeMode } from '../types';

interface TelemetryDrawerProps {
  isOpen: boolean;
  onToggle: () => void;
  zDepth: number;
  onZDepthChange: (depth: number) => void;
  lThreshold: number;
  onLThresholdChange: (val: number) => void;
  minAreaHa: number;
  onMinAreaChange: (val: number) => void;
  theme?: ThemeMode;
}

export const TelemetryDrawer: React.FC<TelemetryDrawerProps> = ({
  isOpen,
  onToggle,
  zDepth,
  onZDepthChange,
  lThreshold,
  onLThresholdChange,
  minAreaHa,
  onMinAreaChange,
  theme = 'dark'
}) => {
  const [layers, setLayers] = useState({
    emitSwir: true,
    tmi1vd: true,
    aemConductance: false,
    copernicusDem: true,
    actualMines: true,
    targetPolygons: true
  });

  const toggleLayer = (k: keyof typeof layers) => {
    setLayers(prev => ({ ...prev, [k]: !prev[k] }));
  };

  const handleReset = () => {
    onLThresholdChange(0.50);
    onZDepthChange(380);
    onMinAreaChange(5.0);
  };

  return (
    <aside
      className={`relative h-full flex flex-col bg-[var(--bg-surface)] border-r border-[var(--border-hairline)] transition-all duration-200 ease-in-out shrink-0 z-30 select-none ${
        isOpen ? 'w-[270px]' : 'w-[48px]'
      }`}
    >
      {/* Drawer Header & Collapse Toggle */}
      <div className="h-[46px] flex items-center justify-between px-3 border-b border-[var(--border-hairline)] shrink-0">
        {isOpen ? (
          <div className="flex items-center gap-2">
            <Filter className="w-3.5 h-3.5 text-[var(--accent-primary)]" />
            <span className="font-display font-semibold text-[11px] uppercase tracking-wider text-[var(--text-secondary)]">
              PIPELINE PARAMETERS
            </span>
          </div>
        ) : (
          <div className="mx-auto">
            <Sliders className="w-4 h-4 text-[var(--accent-primary)]" />
          </div>
        )}

        <button
          onClick={onToggle}
          className="p-1 rounded-md bg-[var(--bg-elevated)] border border-[var(--border-hairline)] text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:border-[var(--accent-primary)] transition-colors cursor-pointer"
          title={isOpen ? 'Collapse rail' : 'Expand drawer'}
          aria-label={isOpen ? 'Collapse parameter drawer' : 'Expand parameter drawer'}
        >
          {isOpen ? <ChevronLeft className="w-3.5 h-3.5" /> : <ChevronRight className="w-3.5 h-3.5" />}
        </button>
      </div>

      {/* Expanded vs Collapsed Rail */}
      {isOpen ? (
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {/* Conformal Risk Threshold */}
          <div className="space-y-1.5">
            <div className="flex items-center justify-between text-[11px] font-mono">
              <span className="text-[var(--text-primary)] font-semibold uppercase">
                CONFORMAL L(x) THR
              </span>
              <span className="text-[var(--accent-primary)] font-bold">
                {lThreshold.toFixed(2)}
              </span>
            </div>
            <input
              type="range"
              min={0.1}
              max={0.95}
              step={0.05}
              value={lThreshold}
              onChange={(e) => onLThresholdChange(parseFloat(e.target.value))}
              className="w-full accent-[var(--accent-primary)] cursor-pointer"
            />
            <p className="font-editorial italic text-[11px] text-[var(--text-muted)] leading-relaxed">
              Finite-sample coverage bound P(Y=1|X) ≥ L(x) under Elkan-Noto PU-learning.
            </p>
          </div>

          {/* Subsurface Z-Depth Slice */}
          <div className="space-y-1.5 pt-3 border-t border-[var(--border-hairline)]">
            <div className="flex items-center justify-between text-[11px] font-mono">
              <span className="text-[var(--text-primary)] font-semibold uppercase">
                Z-DEPTH SLICE
              </span>
              <span className="text-[var(--accent-primary)] font-bold">
                -{zDepth} m
              </span>
            </div>
            <input
              type="range"
              min={50}
              max={1000}
              step={10}
              value={zDepth}
              onChange={(e) => onZDepthChange(parseInt(e.target.value, 10))}
              className="w-full accent-[var(--accent-primary)] cursor-pointer"
            />
          </div>

          {/* Minimum Target Footprint Area */}
          <div className="space-y-1.5 pt-3 border-t border-[var(--border-hairline)]">
            <div className="flex items-center justify-between text-[11px] font-mono">
              <span className="text-[var(--text-primary)] font-semibold uppercase">
                MIN TARGET AREA
              </span>
              <span className="text-[var(--accent-primary)] font-bold">
                {minAreaHa.toFixed(1)} ha
              </span>
            </div>
            <input
              type="range"
              min={1.0}
              max={50.0}
              step={1.0}
              value={minAreaHa}
              onChange={(e) => onMinAreaChange(parseFloat(e.target.value))}
              className="w-full accent-[var(--accent-primary)] cursor-pointer"
            />
          </div>

          {/* Sensor Ingestion Layer Toggles */}
          <div className="space-y-2 pt-3 border-t border-[var(--border-hairline)]">
            <span className="font-display font-semibold text-[10.5px] uppercase tracking-wider text-[var(--text-muted)] block">
              SENSOR INGESTION LAYERS
            </span>

            <div className="space-y-1.5">
              {[
                { key: 'emitSwir' as const, label: 'EMIT Hyperspectral SWIR' },
                { key: 'tmi1vd' as const, label: 'GA Magnetic TMI-1VD' },
                { key: 'aemConductance' as const, label: 'AEM Conductance (0-300m)' },
                { key: 'copernicusDem' as const, label: 'Copernicus 30m DEM Slope' },
                { key: 'actualMines' as const, label: 'GSWA MINEDEX Ground Truth' },
                { key: 'targetPolygons' as const, label: 'Target Vector Polygons' }
              ].map(({ key, label }) => (
                <label
                  key={key}
                  className="flex items-center justify-between p-2 rounded-md bg-[var(--bg-elevated)] border border-[var(--border-hairline)] text-[11px] text-[var(--text-secondary)] hover:text-[var(--text-primary)] cursor-pointer transition-colors"
                >
                  <span>{label}</span>
                  <input
                    type="checkbox"
                    checked={layers[key]}
                    onChange={() => toggleLayer(key)}
                    className="accent-[var(--accent-primary)] cursor-pointer"
                  />
                </label>
              ))}
            </div>
          </div>

          {/* Reset Action */}
          <button
            onClick={handleReset}
            className="w-full py-2 px-3 rounded-md bg-[var(--bg-elevated)] border border-[var(--border-hairline)] text-[10.5px] font-mono font-medium text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:border-[var(--accent-primary)] flex items-center justify-center gap-2 transition-colors cursor-pointer"
          >
            <RotateCcw className="w-3 h-3 text-[var(--accent-primary)]" />
            RESET DEFAULT PRIOR
          </button>
        </div>
      ) : (
        /* Collapsed 48px Icon Rail */
        <div className="flex-1 flex flex-col items-center gap-4 py-4">
          <button
            onClick={onToggle}
            className="p-1.5 rounded-md bg-[var(--bg-elevated)] border border-[var(--border-hairline)] text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:border-[var(--accent-primary)] transition-colors cursor-pointer"
            title="Expand Drawer"
            aria-label="Expand Drawer"
          >
            <ChevronRight className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={onToggle}
            className="p-1.5 rounded-md bg-[var(--bg-elevated)] border border-[var(--border-hairline)] text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:border-[var(--accent-primary)] transition-colors cursor-pointer"
            title="Parameters"
            aria-label="Parameters"
          >
            <Sliders className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={onToggle}
            className="p-1.5 rounded-md bg-[var(--bg-elevated)] border border-[var(--border-hairline)] text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:border-[var(--accent-primary)] transition-colors cursor-pointer"
            title="Layers"
            aria-label="Layers"
          >
            <Layers className="w-3.5 h-3.5" />
          </button>
          <span className="font-mono text-[9px] text-[var(--text-muted)] tracking-wider [writing-mode:vertical-rl] rotate-180 mt-4">
            -{zDepth} m
          </span>
        </div>
      )}
    </aside>
  );
};
