'use client';

import React, { useState, useMemo } from 'react';
import { DrillTarget, TARGET_DATASET, ThemeMode } from '../types';
import { Download, Search, ArrowUpDown } from 'lucide-react';

interface TelemetryTableProps {
  activeTarget: DrillTarget | null;
  onSelectTarget: (target: DrillTarget) => void;
  hoveredTargetId?: string | null;
  onHoverTarget?: (targetId: string | null) => void;
  theme?: ThemeMode;
}

export const TelemetryTable: React.FC<TelemetryTableProps> = ({
  activeTarget,
  onSelectTarget,
  hoveredTargetId,
  onHoverTarget,
  theme = 'dark'
}) => {
  const [search, setSearch] = useState('');
  const [sortKey, setSortKey] = useState<'id' | 'name' | 'depthMeters' | 'prospectivityScore' | 'epistemicUncertainty'>('prospectivityScore');
  const [sortAsc, setSortAsc] = useState(false);

  const isLight = theme === 'light';

  const handleSort = (key: typeof sortKey) => {
    if (sortKey === key) {
      setSortAsc(!sortAsc);
    } else {
      setSortKey(key);
      setSortAsc(key === 'id' || key === 'name');
    }
  };

  const filtered = useMemo(() => {
    const q = search.toLowerCase().trim();
    const list = TARGET_DATASET.filter(t => 
      !q ||
      t.id.toLowerCase().includes(q) ||
      t.name.toLowerCase().includes(q) ||
      t.structuralDomain.toLowerCase().includes(q) ||
      t.primaryDriver.toLowerCase().includes(q) ||
      t.classification.toLowerCase().includes(q)
    );

    return list.sort((a, b) => {
      let va = a[sortKey];
      let vb = b[sortKey];
      if (va < vb) return sortAsc ? -1 : 1;
      if (va > vb) return sortAsc ? 1 : -1;
      return 0;
    });
  }, [search, sortKey, sortAsc]);

  const exportCSV = () => {
    const headers = ['ID', 'Name', 'Easting', 'Northing', 'Depth_m', 'Score', 'Uncertainty', 'Conformal_Low', 'Conformal_High', 'Classification', 'Domain'];
    const rows = TARGET_DATASET.map(t => [
      t.id,
      `"${t.name}"`,
      t.easting,
      t.northing,
      t.depthMeters,
      t.prospectivityScore,
      t.epistemicUncertainty,
      t.conformalInterval[0],
      t.conformalInterval[1],
      `"${t.classification}"`,
      `"${t.structuralDomain}"`
    ]);
    const csvContent = 'data:text/csv;charset=utf-8,' + [headers.join(','), ...rows.map(e => e.join(','))].join('\n');
    const encoded = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encoded);
    link.setAttribute('download', 'drill_targets_portfolio.csv');
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="flex flex-col bg-[var(--bg-surface)] border border-[var(--border-hairline)] rounded-lg p-4 transition-all duration-200">
      {/* Table Section Header */}
      <div className="flex flex-wrap items-center justify-between pb-2.5 mb-2.5 border-b border-[var(--border-hairline)] gap-2">
        <div className="flex items-center gap-2">
          <span className="font-mono text-[10px] font-bold text-[var(--accent-primary)] bg-[var(--bg-elevated)] px-1.5 py-0.5 rounded border border-[var(--border-hairline)]">
            04
          </span>
          <h2 className="font-display text-[12.5px] font-semibold tracking-[-0.02em] text-[var(--text-primary)]">
            RANKED DRILL TARGET PORTFOLIO
          </h2>
        </div>

        {/* Search & Export Actions */}
        <div className="flex items-center gap-2">
          <div className="relative flex items-center">
            <Search className="w-3 h-3 text-[var(--text-muted)] absolute left-2 pointer-events-none" />
            <input
              type="text"
              placeholder="Search targets..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="bg-[var(--bg-elevated)] border border-[var(--border-hairline)] rounded text-[10.5px] font-mono pl-6 pr-2 py-1 text-[var(--text-primary)] placeholder-[var(--text-muted)] focus:outline-none focus:border-[var(--accent-primary)] w-[130px]"
            />
          </div>

          <button
            onClick={exportCSV}
            className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-[var(--bg-elevated)] border border-[var(--border-hairline)] text-[10.5px] font-mono text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:border-[var(--accent-primary)] transition-colors cursor-pointer"
            title="Export CSV"
          >
            <Download className="w-3 h-3 text-[var(--accent-primary)]" />
            <span>CSV</span>
          </button>
        </div>
      </div>

      {/* Responsive High-Density Table */}
      <div className="w-full overflow-x-auto max-h-[220px]">
        <table className="w-full text-left font-mono text-[11px] border-collapse">
          <thead>
            <tr className="border-b border-[var(--border-hairline)] text-[var(--text-muted)] text-[9.5px] tracking-wider uppercase sticky top-0 bg-[var(--bg-surface)] z-10">
              <th className="py-2 px-2.5 cursor-pointer" onClick={() => handleSort('id')}>
                TARGET ID ↕
              </th>
              <th className="py-2 px-2.5 font-sans cursor-pointer" onClick={() => handleSort('name')}>
                NAME &amp; GEOLOGICAL DOMAIN ↕
              </th>
              <th className="py-2 px-2.5">COORDINATES (MGA51)</th>
              <th className="py-2 px-2.5 cursor-pointer" onClick={() => handleSort('depthMeters')}>
                DEPTH ↕
              </th>
              <th className="py-2 px-2.5 cursor-pointer" onClick={() => handleSort('prospectivityScore')}>
                SCORE ↕
              </th>
              <th className="py-2 px-2.5 cursor-pointer" onClick={() => handleSort('epistemicUncertainty')}>
                UNCERTAINTY ↕
              </th>
              <th className="py-2 px-2.5">90% CI</th>
              <th className="py-2 px-2.5">CLASS</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-[var(--border-hairline)]">
            {filtered.map((target) => {
              const isSelected = activeTarget?.id === target.id;
              const isHovered = hoveredTargetId === target.id;

              return (
                <tr
                  key={target.id}
                  onClick={() => onSelectTarget(target)}
                  onMouseEnter={() => onHoverTarget && onHoverTarget(target.id)}
                  onMouseLeave={() => onHoverTarget && onHoverTarget(null)}
                  className={`cursor-pointer transition-colors duration-120 ${
                    isSelected
                      ? 'bg-[var(--bg-elevated)] text-[var(--accent-primary)] font-semibold'
                      : isHovered
                      ? isLight ? 'bg-slate-100/80' : 'bg-[#141D2E]/80'
                      : 'hover:bg-[var(--bg-elevated)]/50'
                  }`}
                >
                  <td className="py-2 px-2.5 font-bold text-[var(--accent-primary)]">
                    {target.id}
                  </td>
                  <td className="py-2 px-2.5 font-sans">
                    <div className="text-[var(--text-primary)] font-medium text-[11px]">
                      {target.name}
                    </div>
                    <div className="font-editorial italic text-[10.5px] text-[var(--text-secondary)]">
                      {target.structuralDomain}
                    </div>
                  </td>
                  <td className="py-2 px-2.5 tabular-nums text-[var(--text-secondary)]">
                    {target.easting.toLocaleString()}E / {target.northing.toLocaleString()}N
                  </td>
                  <td className="py-2 px-2.5 tabular-nums text-[var(--text-secondary)]">
                    -{target.depthMeters} m
                  </td>
                  <td className="py-2 px-2.5 tabular-nums font-bold text-[var(--accent-primary)]">
                    {target.prospectivityScore.toFixed(3)}
                  </td>
                  <td className="py-2 px-2.5 tabular-nums text-emerald-400">
                    ±{target.epistemicUncertainty.toFixed(3)}
                  </td>
                  <td className="py-2 px-2.5 tabular-nums text-[var(--text-secondary)] text-[10px]">
                    [{target.conformalInterval[0].toFixed(3)}, {target.conformalInterval[1].toFixed(3)}]
                  </td>
                  <td className="py-2 px-2.5">
                    <span
                      className={`inline-block px-1.5 py-0.5 rounded text-[9px] font-mono tracking-tight font-semibold ${
                        target.classification === 'TIER-1 CORE'
                          ? 'bg-sky-500/15 text-sky-400 border border-sky-500/30'
                          : target.classification === 'STRUCTURAL EXT'
                          ? 'bg-amber-500/15 text-amber-400 border border-amber-500/30'
                          : target.classification === 'SHEAR ANOMALY'
                          ? 'bg-purple-500/15 text-purple-400 border border-purple-500/30'
                          : 'bg-emerald-500/15 text-emerald-400 border border-emerald-500/30'
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
  );
};
