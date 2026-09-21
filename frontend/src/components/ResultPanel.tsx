import React from 'react';
import { AlertTriangle, Activity } from 'lucide-react';
import type { ClassifyResponse } from '../types';

interface ResultPanelProps {
  result: ClassifyResponse | null;
  isLoading: boolean;
  error?: string | null;
}

export const ResultPanel: React.FC<ResultPanelProps> = ({
  result,
  isLoading,
  error,
}) => {
  if (isLoading) {
    return (
      <div style={{
        height: '100%',
        backgroundColor: 'var(--color-surface)',
        border: '1px solid var(--border-structural)',
        borderRadius: '2px',
        padding: '24px',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        gap: '14px',
      }}>
        <div style={{
          width: '28px',
          height: '28px',
          border: '2px solid var(--border-structural)',
          borderTopColor: 'var(--color-mars)',
          borderRadius: '50%',
          animation: 'spin 0.8s linear infinite',
        }} />
        <div className="font-mono" style={{ fontSize: '11px', color: 'var(--text-secondary)', letterSpacing: '0.08em' }}>
          EXECUTING INFERENCE PIPELINE...
        </div>
        <style>{`
          @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
          }
        `}</style>
      </div>
    );
  }

  if (error) {
    return (
      <div style={{
        height: '100%',
        backgroundColor: 'var(--color-surface)',
        border: '1px solid var(--border-structural)',
        borderRadius: '2px',
        padding: '20px',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        gap: '10px',
        textAlign: 'center',
      }}>
        <AlertTriangle size={28} color="var(--color-error)" />
        <div className="font-mono" style={{ fontWeight: 700, fontSize: '12px', color: 'var(--color-error)' }}>
          CLASSIFICATION PIPELINE ERROR
        </div>
        <div style={{ fontSize: '11px', color: 'var(--text-secondary)', maxWidth: '280px' }}>
          {error}
        </div>
      </div>
    );
  }

  if (!result) {
    return (
      <div style={{
        height: '100%',
        backgroundColor: 'var(--color-surface)',
        border: '1px solid var(--border-structural)',
        borderRadius: '2px',
        padding: '20px',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        gap: '10px',
        textAlign: 'center',
        color: 'var(--text-muted)',
      }}>
        <Activity size={28} color="var(--border-structural)" />
        <div className="font-mono" style={{ fontSize: '11px', letterSpacing: '0.06em' }}>
          AWAITING CLASSIFICATION TRIGGER
        </div>
        <div style={{ fontSize: '11px', maxWidth: '240px', color: 'var(--text-secondary)' }}>
          Select sample & model, then execute classification to view telemetry.
        </div>
      </div>
    );
  }

  const cleanDisplayName = result.model_name.replace(/\[Champion\]\s*/i, '').replace(/Champion\s*/i, '').trim();
  const isLive = result.inference_source === 'LIVE MODEL';
  const hasConfidence = result.confidence !== null && !isNaN(result.confidence);
  const hasTopPredictions = Array.isArray(result.top_predictions) && result.top_predictions.length > 0;

  return (
    <div style={{
      height: '100%',
      backgroundColor: 'var(--color-surface)',
      border: '1px solid var(--border-structural)',
      borderRadius: '2px',
      display: 'flex',
      flexDirection: 'column',
      fontFamily: 'var(--font-mono)',
    }}>
      {/* Top Header */}
      <div style={{
        padding: '6px 12px',
        backgroundColor: 'var(--color-charcoal)',
        borderBottom: '1px solid var(--border-hairline)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        fontSize: '11px',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ color: 'var(--color-green-bright)', fontWeight: 800 }}>■</span>
          <span style={{ fontWeight: 800, letterSpacing: '0.04em', color: '#FFFFFF' }}>
            03. CLASSIFICATION RESULT
          </span>
        </div>

        {/* Source Badge */}
        {isLive ? (
          <span className="badge badge-emerald" style={{ fontSize: '9px', padding: '1px 5px' }}>
            LIVE MODEL
          </span>
        ) : (
          <span className="badge badge-amber" style={{ fontSize: '9px', padding: '1px 5px' }}>
            STORED TEST PREDICTION
          </span>
        )}
      </div>

      {/* Main Result Body */}
      <div style={{ padding: '14px', flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '12px' }}>
        
        {/* Row 1: Predicted Class & Confidence */}
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'flex-start',
          gap: '12px',
          paddingBottom: '10px',
          borderBottom: '1px solid var(--border-hairline)',
        }}>
          <div>
            <div style={{ fontSize: '9px', color: 'var(--text-muted)', letterSpacing: '0.06em', textTransform: 'uppercase' }}>
              PREDICTED CLASS [ARGMAX]
            </div>
            <div style={{
              fontSize: '20px',
              fontWeight: 800,
              fontFamily: 'var(--font-heading)',
              color: '#FFFFFF',
              marginTop: '2px',
              textTransform: 'capitalize',
              lineHeight: 1.1,
            }}>
              {result.predicted_class_name}
            </div>
            <div style={{ fontSize: '9px', color: 'var(--text-muted)', marginTop: '2px' }}>
              NASA TAXA ID: {result.predicted_class_id}
            </div>
          </div>

          <div style={{ textAlign: 'right' }}>
            <div style={{ fontSize: '9px', color: 'var(--text-muted)', letterSpacing: '0.06em' }}>
              CONFIDENCE
            </div>
            <div style={{
              fontSize: '20px',
              fontWeight: 800,
              color: hasConfidence ? 'var(--color-mars)' : 'var(--text-muted)',
              marginTop: '2px',
              lineHeight: 1.1,
            }}>
              {hasConfidence ? `${(result.confidence! * 100).toFixed(2)}%` : 'N/A'}
            </div>
            <div style={{ fontSize: '8px', color: 'var(--text-muted)', marginTop: '2px' }}>
              {hasConfidence ? 'SOFTMAX ARGMAX' : 'DISCRETE ONLY'}
            </div>
          </div>
        </div>

        {/* Row 2: Ground Truth Verification Box */}
        <div style={{
          padding: '8px 10px',
          backgroundColor: 'var(--color-obsidian)',
          border: '1px solid var(--border-hairline)',
          borderRadius: '2px',
          display: 'flex',
          flexDirection: 'column',
          gap: '5px',
        }}>
          <div style={{ fontSize: '8px', color: 'var(--text-muted)', letterSpacing: '0.06em', textTransform: 'uppercase' }}>
            GROUND TRUTH VERIFICATION
          </div>
          <div style={{ fontSize: '11px', color: 'var(--text-primary)', textTransform: 'capitalize' }}>
            {result.ground_truth_name}
          </div>

          {/* Correct / Incorrect Banner */}
          <div style={{
            padding: '4px 8px',
            borderRadius: '1px',
            backgroundColor: result.correct ? 'rgba(35, 134, 54, 0.15)' : 'rgba(255, 113, 108, 0.15)',
            border: `1px solid ${result.correct ? 'rgba(35, 134, 54, 0.45)' : 'rgba(255, 113, 108, 0.45)'}`,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            marginTop: '2px',
          }}>
            <span style={{
              fontSize: '10px',
              fontWeight: 800,
              color: result.correct ? 'var(--color-green-bright)' : 'var(--color-error)',
              display: 'flex',
              alignItems: 'center',
              gap: '4px',
            }}>
              {result.correct ? '✓ CORRECT [MATCH]' : '✕ MISMATCH'}
            </span>
            <span style={{ fontSize: '9px', color: 'var(--text-muted)' }}>
              {result.correct ? 'VERIFIED' : `GROUND TRUTH: ${result.ground_truth_name.toUpperCase()}`}
            </span>
          </div>
        </div>

        {/* Row 3: Posterior Distributions / Top-5 Probabilities */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginTop: '2px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '9px', color: 'var(--text-muted)' }}>
            <span>TOP-5 POSTERIOR DISTRIBUTIONS</span>
            <span>{hasTopPredictions ? 'Σ = 1.000' : 'UNAVAILABLE'}</span>
          </div>

          {hasTopPredictions ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
              {result.top_predictions.slice(0, 5).map((prob, i) => {
                const pct = (prob.probability * 100).toFixed(1);
                const isTop = i === 0;
                return (
                  <div key={prob.class_name || i} style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px' }}>
                      <span style={{ color: isTop ? '#FFFFFF' : 'var(--text-secondary)', fontWeight: isTop ? 700 : 400 }}>
                        {`0${i + 1}. ${prob.class_name}`}
                      </span>
                      <span style={{ color: isTop ? 'var(--color-mars)' : 'var(--text-muted)', fontWeight: isTop ? 800 : 400 }}>
                        {pct}%
                      </span>
                    </div>

                    <div style={{
                      height: isTop ? '4px' : '3px',
                      backgroundColor: 'var(--color-obsidian)',
                      border: '1px solid var(--border-hairline)',
                      overflow: 'hidden',
                    }}>
                      <div style={{
                        height: '100%',
                        width: `${pct}%`,
                        backgroundColor: isTop ? 'var(--color-mars)' : 'var(--border-bright)',
                      }} />
                    </div>
                  </div>
                );
              })}
            </div>
          ) : (
            <div style={{
              padding: '10px',
              backgroundColor: 'var(--color-obsidian)',
              border: '1px dashed var(--border-hairline)',
              textAlign: 'center',
            }}>
              <div style={{ fontSize: '10px', color: 'var(--text-muted)', fontWeight: 700 }}>
                Not Available
              </div>
              <div style={{ fontSize: '9px', color: 'var(--text-dim)', marginTop: '2px', fontFamily: 'var(--font-sans)' }}>
                Classical model stored test prediction contains discrete label only.
              </div>
            </div>
          )}
        </div>

        {/* Row 4: Telemetry Footer */}
        <div style={{
          marginTop: 'auto',
          paddingTop: '8px',
          borderTop: '1px solid var(--border-hairline)',
          display: 'grid',
          gridTemplateColumns: '1fr 1fr',
          gap: '8px',
          fontSize: '9px',
        }}>
          <div>
            <span style={{ color: 'var(--text-muted)' }}>INFERENCE SOURCE:</span>
            <div style={{ color: isLive ? 'var(--color-green-bright)' : 'var(--color-amber)', fontWeight: 700, marginTop: '1px' }}>
              {result.inference_source} ({cleanDisplayName})
            </div>
          </div>

          <div style={{ textAlign: 'right' }}>
            <span style={{ color: 'var(--text-muted)' }}>INFERENCE TIME:</span>
            <div style={{ color: isLive && result.inference_time_ms !== null ? 'var(--color-green-bright)' : 'var(--text-muted)', fontWeight: 700, marginTop: '1px' }}>
              {isLive && result.inference_time_ms !== null
                ? `${result.inference_time_ms.toFixed(1)} MS`
                : 'N/A — STORED PREDICTION'}
            </div>
          </div>
        </div>

      </div>
    </div>
  );
};
