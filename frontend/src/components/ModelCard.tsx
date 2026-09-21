import React from 'react';
import { AlertTriangle } from 'lucide-react';
import type { ModelInfo } from '../types';

interface ModelCardProps {
  model: ModelInfo;
  isSelected: boolean;
  onSelect: (model: ModelInfo) => void;
  canRunInference: boolean;
  cannotRunReason?: string;
}

export const ModelCard: React.FC<ModelCardProps> = ({
  model,
  isSelected,
  onSelect,
  canRunInference,
  cannotRunReason,
}) => {
  // Ensure "Champion" is never displayed
  const cleanDisplayName = model.name.replace(/\[Champion\]\s*/i, '').replace(/Champion\s*/i, '').trim();
  const isDeepLearning = model.category === 'deep_learning';

  return (
    <div
      onClick={() => {
        if (canRunInference) {
          onSelect(model);
        }
      }}
      style={{
        padding: '8px 10px',
        backgroundColor: isSelected ? 'var(--color-charcoal)' : 'var(--color-surface)',
        borderRadius: '2px',
        border: isSelected
          ? '1px solid var(--color-mars)'
          : canRunInference
          ? '1px solid var(--border-structural)'
          : '1px dashed var(--border-hairline)',
        cursor: canRunInference ? 'pointer' : 'not-allowed',
        opacity: canRunInference ? 1 : 0.5,
        transition: 'border-color 0.12s ease',
        display: 'flex',
        flexDirection: 'column',
        gap: '5px',
        fontFamily: 'var(--font-mono)',
      }}
    >
      {/* Top Header: Model Name & Status */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <span style={{ color: isSelected ? 'var(--color-mars)' : 'var(--text-muted)', fontSize: '11px', fontWeight: 800 }}>
            {isSelected ? '■' : '□'}
          </span>
          <span style={{
            fontWeight: 800,
            fontSize: '12px',
            color: isSelected ? 'var(--color-mars)' : 'var(--text-primary)',
            letterSpacing: '0.02em',
          }}>
            {cleanDisplayName}
          </span>
        </div>

        {isSelected ? (
          <span style={{
            backgroundColor: 'var(--color-mars)',
            color: '#090B0E',
            fontSize: '8px',
            fontWeight: 800,
            padding: '1px 5px',
            borderRadius: '1px',
            letterSpacing: '0.04em',
          }}>
            ACTIVE / SELECTED
          </span>
        ) : (
          <span style={{
            fontSize: '8px',
            color: isDeepLearning ? 'var(--color-amber)' : 'var(--text-muted)',
            border: '1px solid var(--border-hairline)',
            padding: '1px 4px',
            borderRadius: '1px',
          }}>
            {isDeepLearning ? 'DEEP LEARNING' : 'CLASSICAL ML'}
          </span>
        )}
      </div>

      {/* Description / Configuration */}
      <div style={{
        fontSize: '10px',
        color: 'var(--text-secondary)',
        whiteSpace: 'nowrap',
        overflow: 'hidden',
        textOverflow: 'ellipsis',
        fontFamily: 'var(--font-sans)',
      }}>
        {model.configuration || model.description}
      </div>

      {/* Bottom Telemetry: Accuracy & Source */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        fontSize: '9px',
        paddingTop: '4px',
        borderTop: '1px solid var(--border-hairline)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
          <span style={{ color: 'var(--text-muted)' }}>ACC:</span>
          <span style={{ color: '#FFFFFF', fontWeight: 700 }}>
            {model.benchmark ? `${model.benchmark.accuracy_pct.toFixed(2)}%` : '—'}
          </span>
          {model.benchmark && (
            <span style={{ color: 'var(--color-mars)' }}>
              • F1: {model.benchmark.macro_f1.toFixed(3)}
            </span>
          )}
        </div>

        <div>
          {model.has_checkpoint ? (
            <span style={{ color: 'var(--color-green-bright)', fontWeight: 600 }}>LIVE MODEL</span>
          ) : (
            <span style={{ color: 'var(--text-muted)' }}>STORED TEST</span>
          )}
        </div>
      </div>

      {/* Reason why inference disabled if any */}
      {!canRunInference && cannotRunReason && (
        <div style={{
          padding: '2px 4px',
          backgroundColor: 'rgba(255, 113, 108, 0.1)',
          color: 'var(--color-error)',
          fontSize: '9px',
          display: 'flex',
          alignItems: 'center',
          gap: '4px',
        }}>
          <AlertTriangle size={10} />
          <span>{cannotRunReason}</span>
        </div>
      )}
    </div>
  );
};
