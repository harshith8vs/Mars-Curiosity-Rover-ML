import React from 'react';
import type { ModelInfo } from '../types';

interface BenchmarkTableProps {
  models: ModelInfo[];
  isLoading: boolean;
}

export const BenchmarkTable: React.FC<BenchmarkTableProps> = ({
  models,
  isLoading,
}) => {
  const dlModels = models.filter((m) => m.category === 'deep_learning');
  const clModels = models.filter((m) => m.category === 'classical_ml');

  return (
    <div style={{
      backgroundColor: 'var(--color-surface)',
      border: '1px solid var(--border-structural)',
      borderRadius: '2px',
      overflow: 'hidden',
      fontFamily: 'var(--font-mono)',
    }}>
      {/* Header */}
      <div style={{
        padding: '8px 12px',
        backgroundColor: 'var(--color-charcoal)',
        borderBottom: '1px solid var(--border-hairline)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        fontSize: '11px',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ color: 'var(--color-mars)', fontWeight: 800 }}>■</span>
          <span style={{ fontWeight: 800, color: '#FFFFFF', letterSpacing: '0.04em' }}>
            MODEL BENCHMARK MATRIX
          </span>
          <span style={{ color: 'var(--text-muted)', fontSize: '9px' }}>
            STANDARDIZED EVALUATION ON NASA PDS MSL TEST PARTITION
          </span>
        </div>

        <span className="badge badge-amber" style={{ fontSize: '9px', padding: '1px 5px' }}>
          BENCHMARK READY
        </span>
      </div>

      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '11px' }}>
          <thead>
            <tr style={{ backgroundColor: 'var(--color-charcoal)', borderBottom: '1px solid var(--border-structural)' }}>
              <th style={{ padding: '6px 10px', color: 'var(--text-muted)' }}>MODEL</th>
              <th style={{ padding: '6px 10px', color: 'var(--text-muted)' }}>TYPE</th>
              <th style={{ padding: '6px 10px', color: 'var(--text-muted)' }}>ARCHITECTURE</th>
              <th style={{ padding: '6px 10px', color: 'var(--text-muted)' }}>ACCURACY</th>
              <th style={{ padding: '6px 10px', color: 'var(--text-muted)' }}>MACRO-F1</th>
              <th style={{ padding: '6px 10px', color: 'var(--text-muted)' }}>WEIGHTED-F1</th>
              <th style={{ padding: '6px 10px', color: 'var(--text-muted)' }}>STATUS</th>
            </tr>
          </thead>
          <tbody>
            {isLoading ? (
              <tr>
                <td colSpan={7} style={{ padding: '30px', textAlign: 'center', color: 'var(--text-muted)' }}>
                  LOADING BENCHMARK TELEMETRY...
                </td>
              </tr>
            ) : (
              <>
                {/* Deep Learning Group Header */}
                <tr style={{ backgroundColor: 'var(--color-obsidian)', borderBottom: '1px solid var(--border-hairline)' }}>
                  <td colSpan={7} style={{ padding: '4px 10px', fontSize: '9px', color: 'var(--color-mars)', fontWeight: 700 }}>
                    // DEEP LEARNING NEURAL ARCHITECTURES (TRANSFER & ATTENTION)
                  </td>
                </tr>

                {dlModels.map((m) => {
                  const cleanName = m.name.replace(/\[Champion\]\s*/i, '').replace(/Champion\s*/i, '').trim();
                  return (
                    <tr key={m.id} style={{ borderBottom: '1px solid var(--border-hairline)' }}>
                      <td style={{ padding: '6px 10px', fontWeight: 700, color: '#FFFFFF' }}>
                        <span style={{ color: 'var(--color-mars)', marginRight: '4px' }}>■</span>
                        {cleanName}
                      </td>
                      <td style={{ padding: '6px 10px', color: 'var(--color-mars)', fontSize: '10px' }}>
                        Deep Learning
                      </td>
                      <td style={{ padding: '6px 10px', color: 'var(--text-secondary)', fontSize: '10px' }}>
                        {m.architecture}
                      </td>
                      <td style={{ padding: '6px 10px', fontWeight: 700, color: '#FFFFFF' }}>
                        {m.benchmark ? `${m.benchmark.accuracy_pct.toFixed(2)}%` : '—'}
                      </td>
                      <td style={{ padding: '6px 10px', fontWeight: 700, color: 'var(--color-mars)' }}>
                        {m.benchmark ? m.benchmark.macro_f1.toFixed(4) : '—'}
                      </td>
                      <td style={{ padding: '6px 10px', color: 'var(--color-amber)' }}>
                        {m.benchmark ? m.benchmark.weighted_f1.toFixed(4) : '—'}
                      </td>
                      <td style={{ padding: '6px 10px' }}>
                        <span className="badge badge-emerald" style={{ fontSize: '8px', padding: '1px 5px' }}>
                          CALIBRATED
                        </span>
                      </td>
                    </tr>
                  );
                })}

                {/* Classical ML Group Header */}
                <tr style={{ backgroundColor: 'var(--color-obsidian)', borderBottom: '1px solid var(--border-hairline)' }}>
                  <td colSpan={7} style={{ padding: '4px 10px', fontSize: '9px', color: 'var(--color-amber)', fontWeight: 700 }}>
                    // CLASSICAL MACHINE LEARNING (SUPERVISED STATISTICAL BASELINES)
                  </td>
                </tr>

                {clModels.map((m) => {
                  return (
                    <tr key={m.id} style={{ borderBottom: '1px solid var(--border-hairline)' }}>
                      <td style={{ padding: '6px 10px', fontWeight: 700, color: '#FFFFFF' }}>
                        <span style={{ color: 'var(--color-amber)', marginRight: '4px' }}>■</span>
                        {m.name}
                      </td>
                      <td style={{ padding: '6px 10px', color: 'var(--color-amber)', fontSize: '10px' }}>
                        Classical ML
                      </td>
                      <td style={{ padding: '6px 10px', color: 'var(--text-secondary)', fontSize: '10px' }}>
                        {m.architecture}
                      </td>
                      <td style={{ padding: '6px 10px', fontWeight: 700, color: '#FFFFFF' }}>
                        {m.benchmark ? `${m.benchmark.accuracy_pct.toFixed(2)}%` : '—'}
                      </td>
                      <td style={{ padding: '6px 10px', fontWeight: 700, color: 'var(--color-mars)' }}>
                        {m.benchmark ? m.benchmark.macro_f1.toFixed(4) : '—'}
                      </td>
                      <td style={{ padding: '6px 10px', color: 'var(--color-amber)' }}>
                        {m.benchmark ? m.benchmark.weighted_f1.toFixed(4) : '—'}
                      </td>
                      <td style={{ padding: '6px 10px' }}>
                        <span className="badge badge-emerald" style={{ fontSize: '8px', padding: '1px 5px' }}>
                          CALIBRATED
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </>
            )}
          </tbody>
        </table>
      </div>

      <div style={{
        padding: '6px 12px',
        backgroundColor: 'var(--color-charcoal)',
        borderTop: '1px solid var(--border-hairline)',
        display: 'flex',
        justifyContent: 'space-between',
        fontSize: '9px',
        color: 'var(--text-muted)',
      }}>
        <span>PIPELINE SPECIFICATION: MSL-SCI-ML-04.2</span>
        <span>WEIGHT ARTIFACTS: SECURE LOCAL PDS REPO</span>
      </div>
    </div>
  );
};
