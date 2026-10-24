'use client';

import React, { useState } from 'react';
import { DrillTarget, TARGET_DATASET, ThemeMode } from '../types';

interface StatisticalAnalyticsChartProps {
  activeTarget: DrillTarget | null;
  onSelectTarget: (target: DrillTarget) => void;
  hoveredTargetId?: string | null;
  onHoverTarget?: (targetId: string | null) => void;
  theme?: ThemeMode;
}

export const StatisticalAnalyticsChart: React.FC<StatisticalAnalyticsChartProps> = ({
  activeTarget,
  onSelectTarget,
  hoveredTargetId,
  onHoverTarget,
  theme = 'dark'
}) => {
  const [activeTab, setActiveTab] = useState<'conformal' | 'prauc' | 'calibration'>('conformal');
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);

  const isLight = theme === 'light';

  // SVG dimensions & margins
  const width = 620;
  const height = 210;
  const padding = { top: 22, right: 28, bottom: 36, left: 48 };
  const innerWidth = width - padding.left - padding.right;
  const innerHeight = height - padding.top - padding.bottom;

  // PR-AUC synthetic points for XGBoost, U-Net, and Fused
  const recallSteps = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
  const xgbPrec = [1.0, 0.96, 0.94, 0.91, 0.88, 0.84, 0.78, 0.71, 0.62, 0.48, 0.22];
  const unetPrec = [1.0, 0.92, 0.89, 0.86, 0.82, 0.79, 0.73, 0.66, 0.58, 0.44, 0.18];
  const fusedPrec = [1.0, 0.98, 0.96, 0.93, 0.90, 0.87, 0.83, 0.76, 0.69, 0.54, 0.26];

  const toSvgCoords = (rec: number, prec: number) => {
    const x = padding.left + rec * innerWidth;
    const y = padding.top + (1 - prec) * innerHeight;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  };

  const fusedPath = fusedPrec.map((p, i) => toSvgCoords(recallSteps[i], p)).join(' L ');
  const xgbPath = xgbPrec.map((p, i) => toSvgCoords(recallSteps[i], p)).join(' L ');
  const unetPath = unetPrec.map((p, i) => toSvgCoords(recallSteps[i], p)).join(' L ');

  return (
    <div className="flex flex-col bg-[var(--bg-surface)] border border-[var(--border-hairline)] rounded-lg p-4 transition-all duration-200">
      {/* Header with multi-tier typography */}
      <div className="flex flex-wrap items-center justify-between pb-2.5 mb-2.5 border-b border-[var(--border-hairline)] gap-2">
        <div className="flex items-center gap-2">
          <span className="font-mono text-[10px] font-bold text-[var(--accent-primary)] bg-[var(--bg-elevated)] px-1.5 py-0.5 rounded border border-[var(--border-hairline)]">
            02
          </span>
          <h2 className="font-display text-[12.5px] font-semibold tracking-[-0.02em] text-[var(--text-primary)]">
            CONFORMAL RISK &amp; PU-LEARNING CALIBRATION
          </h2>
        </div>

        {/* Analytical Tab Pills */}
        <div className="flex items-center gap-1 bg-[var(--bg-elevated)] p-1 rounded-md border border-[var(--border-hairline)]">
          <button
            onClick={() => setActiveTab('conformal')}
            className={`px-2.5 py-1 text-[10.5px] font-sans font-semibold rounded transition-all cursor-pointer ${
              activeTab === 'conformal'
                ? 'bg-[var(--accent-primary)] text-slate-950 shadow-sm'
                : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
            }`}
          >
            Conformal 90% CI
          </button>
          <button
            onClick={() => setActiveTab('prauc')}
            className={`px-2.5 py-1 text-[10.5px] font-sans font-semibold rounded transition-all cursor-pointer ${
              activeTab === 'prauc'
                ? 'bg-[var(--accent-primary)] text-slate-950 shadow-sm'
                : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
            }`}
          >
            PR-AUC Ensembles
          </button>
        </div>
      </div>

      {/* SVG Analytical Chart Body */}
      <div className="relative w-full overflow-hidden bg-[var(--bg-canvas)] rounded-md border border-[var(--border-hairline)] p-2">
        {activeTab === 'conformal' ? (
          <div className="w-full">
            <svg
              viewBox={`0 0 ${width} ${height}`}
              className="w-full h-auto select-none"
            >
              {/* Hairline Grid Lines */}
              {[0, 0.25, 0.5, 0.75, 1.0].map((frac) => {
                const y = padding.top + frac * innerHeight;
                return (
                  <g key={`grid-${frac}`}>
                    <line
                      x1={padding.left}
                      y1={y}
                      x2={width - padding.right}
                      y2={y}
                      stroke="var(--border-hairline)"
                      strokeWidth="1"
                      strokeDasharray="3 3"
                    />
                    <text
                      x={padding.left - 8}
                      y={y + 3.5}
                      textAnchor="end"
                      fill="var(--text-muted)"
                      fontFamily="var(--font-mono)"
                      fontSize="9.5"
                    >
                      {(1 - frac).toFixed(2)}
                    </text>
                  </g>
                );
              })}

              {/* Target Conformal Confidence Boxes */}
              {TARGET_DATASET.map((target, idx) => {
                const colW = innerWidth / TARGET_DATASET.length;
                const cx = padding.left + (idx + 0.5) * colW;
                const cyScore = padding.top + (1 - target.prospectivityScore) * innerHeight;
                const cyLow = padding.top + (1 - target.conformalInterval[0]) * innerHeight;
                const cyHigh = padding.top + (1 - target.conformalInterval[1]) * innerHeight;

                const isSelected = activeTarget?.id === target.id;
                const isHovered = hoveredTargetId === target.id || hoveredIndex === idx;

                return (
                  <g
                    key={target.id}
                    className="cursor-pointer"
                    onClick={() => onSelectTarget(target)}
                    onMouseEnter={() => {
                      setHoveredIndex(idx);
                      if (onHoverTarget) onHoverTarget(target.id);
                    }}
                    onMouseLeave={() => {
                      setHoveredIndex(null);
                      if (onHoverTarget) onHoverTarget(null);
                    }}
                  >
                    {/* Confidence Range Box */}
                    <rect
                      x={cx - 18}
                      y={cyHigh}
                      width={36}
                      height={Math.max(4, cyLow - cyHigh)}
                      rx="3"
                      fill={isSelected ? (isLight ? 'rgba(2, 132, 199, 0.2)' : 'rgba(56, 189, 248, 0.2)') : 'var(--bg-elevated)'}
                      stroke={isSelected ? 'var(--accent-primary)' : isHovered ? '#F59E0B' : 'var(--border-hairline)'}
                      strokeWidth={isSelected ? 1.5 : 1}
                    />

                    {/* Point Estimate Circle */}
                    <circle
                      cx={cx}
                      cy={cyScore}
                      r={isSelected ? 5.5 : 4}
                      fill={isSelected ? 'var(--accent-primary)' : isHovered ? '#F59E0B' : 'var(--text-primary)'}
                      stroke="var(--bg-surface)"
                      strokeWidth="1.5"
                    />

                    {/* Target Label */}
                    <text
                      x={cx}
                      y={height - padding.bottom + 15}
                      textAnchor="middle"
                      fill={isSelected ? 'var(--accent-primary)' : 'var(--text-secondary)'}
                      fontFamily="var(--font-mono)"
                      fontSize="10"
                      fontWeight="600"
                    >
                      {target.id.replace('TGT-YIL-', 'T-')}
                    </text>
                  </g>
                );
              })}
            </svg>
          </div>
        ) : (
          <div className="w-full">
            <svg
              viewBox={`0 0 ${width} ${height}`}
              className="w-full h-auto select-none"
            >
              {[0, 0.25, 0.5, 0.75, 1.0].map((frac) => {
                const y = padding.top + frac * innerHeight;
                return (
                  <g key={`grid-pr-${frac}`}>
                    <line
                      x1={padding.left}
                      y1={y}
                      x2={width - padding.right}
                      y2={y}
                      stroke="var(--border-hairline)"
                      strokeWidth="1"
                      strokeDasharray="3 3"
                    />
                    <text
                      x={padding.left - 8}
                      y={y + 3.5}
                      textAnchor="end"
                      fill="var(--text-muted)"
                      fontFamily="var(--font-mono)"
                      fontSize="9.5"
                    >
                      {(1 - frac).toFixed(2)}
                    </text>
                  </g>
                );
              })}

              {/* Curves */}
              <path
                d={`M ${xgbPath}`}
                fill="none"
                stroke="#F97316"
                strokeWidth="1.5"
                strokeDasharray="4 2"
              />
              <path
                d={`M ${unetPath}`}
                fill="none"
                stroke="var(--accent-purple)"
                strokeWidth="1.5"
                strokeDasharray="3 3"
              />
              <path
                d={`M ${fusedPath}`}
                fill="none"
                stroke="var(--accent-primary)"
                strokeWidth="2.5"
              />

              {/* Axis Label */}
              <text
                x={width / 2}
                y={height - 6}
                textAnchor="middle"
                fill="var(--text-muted)"
                fontFamily="var(--font-mono)"
                fontSize="9.5"
              >
                Recall (Coverage of Known Deposits)
              </text>
            </svg>
          </div>
        )}
      </div>

      {/* Footer Notes */}
      <div className="flex justify-between items-center mt-2.5 pt-2 border-t border-[var(--border-hairline)] text-[11px] font-mono">
        <span className="font-editorial italic text-[12px] text-[var(--text-secondary)]">
          Prior: <span className="font-mono not-italic font-bold text-[var(--text-primary)]">ĉ = 0.285</span>, Conformal Coverage <span className="font-mono not-italic font-bold text-[var(--accent-primary)]">P(Y ∈ C(X)) ≥ 1 - α</span>
        </span>
        <span className="text-[var(--accent-primary)] font-bold">Fused PR-AUC: 0.872</span>
      </div>
    </div>
  );
};
