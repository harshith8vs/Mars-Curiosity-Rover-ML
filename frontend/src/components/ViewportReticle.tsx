import React, { useState } from 'react';
import { Crosshair, ZoomIn, ZoomOut, AlertCircle, RefreshCw, Grid } from 'lucide-react';
import type { SampleItem } from '../types';
import { api } from '../services/api';

interface ViewportReticleProps {
  sample: SampleItem | null;
  isLoading?: boolean;
  isClassifying?: boolean;
  sampleError?: string | null;
  onOpenSampleDrawer?: () => void;
  onSelectRandomSample?: () => void;
}

export const ViewportReticle: React.FC<ViewportReticleProps> = ({
  sample,
  isLoading,
  isClassifying,
  sampleError,
  onOpenSampleDrawer,
  onSelectRandomSample,
}) => {
  const [zoom, setZoom] = useState<number>(1);
  const [showGrid, setShowGrid] = useState<boolean>(true);
  const [imgError, setImgError] = useState<boolean>(false);

  const handleZoomIn = () => setZoom((prev) => Math.min(prev + 0.25, 2.5));
  const handleZoomOut = () => setZoom((prev) => Math.max(prev - 0.25, 0.75));
  const handleResetZoom = () => setZoom(1);

  const imageUrl = sample ? api.getImageUrl(sample.image_url) : '';

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      backgroundColor: 'var(--color-surface)',
      border: '1px solid var(--border-structural)',
      borderRadius: '2px',
      overflow: 'hidden',
    }}>
      {/* Top Header Bar */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '6px 12px',
        backgroundColor: 'var(--color-charcoal)',
        borderBottom: '1px solid var(--border-hairline)',
        fontFamily: 'var(--font-mono)',
        fontSize: '11px',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ color: 'var(--color-mars)', fontWeight: 800 }}>■</span>
          <span style={{ fontWeight: 800, letterSpacing: '0.04em', color: '#FFFFFF' }}>
            01. ROVER SAMPLE VIEWPORT
          </span>
          <span className="badge badge-neutral" style={{ fontSize: '9px', padding: '1px 5px' }}>
            SPLIT: {sample ? sample.split.toUpperCase() : 'TEST'}
          </span>
          <span style={{ fontSize: '10px', color: 'var(--text-muted)' }}>
            256 × 256 PX
          </span>
          <span className="badge badge-emerald" style={{ fontSize: '9px', padding: '1px 5px' }}>
            ● CALIBRATED RGB
          </span>
        </div>

        {/* Zoom & View Controls */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
          <button
            onClick={() => setShowGrid((g) => !g)}
            className="btn btn-secondary"
            style={{ padding: '3px 6px', fontSize: '10px', borderColor: showGrid ? 'var(--color-mars)' : undefined }}
            title="Toggle Grid"
          >
            <Grid size={11} />
          </button>
          <button
            onClick={handleZoomOut}
            className="btn btn-secondary"
            style={{ padding: '3px 6px', fontSize: '10px' }}
            title="Zoom Out"
          >
            <ZoomOut size={11} />
          </button>
          <span className="font-mono" style={{ fontSize: '10px', color: 'var(--text-muted)', minWidth: '32px', textAlign: 'center' }}>
            {Math.round(zoom * 100)}%
          </span>
          <button
            onClick={handleZoomIn}
            className="btn btn-secondary"
            style={{ padding: '3px 6px', fontSize: '10px' }}
            title="Zoom In"
          >
            <ZoomIn size={11} />
          </button>
          {zoom !== 1 && (
            <button
              onClick={handleResetZoom}
              className="btn btn-secondary"
              style={{ padding: '3px 6px', fontSize: '10px' }}
              title="Reset Zoom"
            >
              <RefreshCw size={10} />
            </button>
          )}
        </div>
      </div>

      {/* Main Viewport Container */}
      <div
        className="reticle-container"
        style={{
          position: 'relative',
          height: '380px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          backgroundColor: '#04070A',
          overflow: 'hidden',
        }}
      >
        {/* Reticle Corner Brackets */}
        <div className="reticle-corner reticle-corner-tl" />
        <div className="reticle-corner reticle-corner-tr" />
        <div className="reticle-corner reticle-corner-bl" />
        <div className="reticle-corner reticle-corner-br" />

        {/* HUD Overlay Top-Left */}
        <div style={{
          position: 'absolute',
          top: '10px',
          left: '12px',
          fontFamily: 'var(--font-mono)',
          fontSize: '9px',
          color: 'var(--color-green-bright)',
          pointerEvents: 'none',
          zIndex: 9,
          lineHeight: 1.3,
        }}>
          <div>DIM: 256 × 256 PX | COLOR: RGB</div>
          <div style={{ color: 'var(--text-muted)' }}>
            SPLIT: {sample?.split.toUpperCase() || 'TEST'} | GT_ID: {sample?.ground_truth_id ?? '--'}
          </div>
        </div>

        {/* HUD Overlay Top-Right */}
        <div style={{
          position: 'absolute',
          top: '10px',
          right: '12px',
          fontFamily: 'var(--font-mono)',
          fontSize: '9px',
          color: 'var(--color-green-bright)',
          pointerEvents: 'none',
          zIndex: 9,
          textAlign: 'right',
        }}>
          <div>TARGET: {sample?.sample_id || 'STANDBY'}</div>
        </div>

        {/* Center Target Box / Reticle */}
        <div style={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          width: '180px',
          height: '110px',
          transform: 'translate(-50%, -50%)',
          border: '1px solid rgba(242, 100, 25, 0.65)',
          pointerEvents: 'none',
          zIndex: 8,
          boxShadow: 'inset 0 0 12px rgba(242, 100, 25, 0.15)',
        }}>
          {/* Target ROI Label Tag */}
          <div style={{
            position: 'absolute',
            top: '-16px',
            left: '4px',
            backgroundColor: 'var(--color-mars)',
            color: '#090B0E',
            fontSize: '8px',
            fontFamily: 'var(--font-mono)',
            fontWeight: 800,
            padding: '1px 5px',
            letterSpacing: '0.04em',
            borderRadius: '1px',
          }}>
            ROI: {sample ? sample.ground_truth_name.toUpperCase() : 'TARGET_TERRAIN'}
          </div>

          {/* Crosshairs */}
          <div style={{
            position: 'absolute',
            top: '55px',
            left: '-10px',
            width: '200px',
            height: '1px',
            backgroundColor: showGrid ? 'rgba(255, 255, 255, 0.12)' : 'transparent',
          }} />
          <div style={{
            position: 'absolute',
            top: '-10px',
            left: '90px',
            width: '1px',
            height: '130px',
            backgroundColor: showGrid ? 'rgba(255, 255, 255, 0.12)' : 'transparent',
          }} />
        </div>

        {/* Laser scanline animation during classification */}
        {isClassifying && (
          <div style={{
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            height: '2px',
            backgroundColor: 'var(--color-mars)',
            boxShadow: '0 0 10px var(--color-mars)',
            zIndex: 12,
            animation: 'scanlineMove 1.5s linear infinite',
          }} />
        )}

        {/* Viewport Content */}
        {sampleError ? (
          <div style={{ textAlign: 'center', color: 'var(--color-error)', padding: '24px', zIndex: 10 }}>
            <AlertCircle size={32} color="var(--color-error)" style={{ margin: '0 auto 8px' }} />
            <div className="font-mono" style={{ fontSize: '12px', fontWeight: 700 }}>SAMPLE NOT FOUND</div>
            <div style={{ fontSize: '11px', color: 'var(--text-secondary)', marginTop: '4px', maxWidth: '280px' }}>
              {sampleError}
            </div>
          </div>
        ) : isLoading ? (
          <div style={{ textAlign: 'center', color: 'var(--text-muted)' }}>
            <div className="font-mono" style={{ fontSize: '11px', letterSpacing: '0.08em' }}>
              INDEXING RADIOMETRIC FRAME...
            </div>
          </div>
        ) : !sample ? (
          <div style={{ textAlign: 'center', color: 'var(--text-muted)' }}>
            <Crosshair size={32} color="var(--border-structural)" style={{ margin: '0 auto 8px' }} />
            <div className="font-mono" style={{ fontSize: '12px' }}>NO SAMPLE SELECTED</div>
          </div>
        ) : imgError ? (
          <div style={{ textAlign: 'center', color: 'var(--color-error)', padding: '24px' }}>
            <AlertCircle size={30} style={{ margin: '0 auto 6px' }} />
            <div className="font-mono" style={{ fontSize: '11px' }}>IMAGE STREAM UNAVAILABLE</div>
            <div style={{ fontSize: '10px', color: 'var(--text-muted)', marginTop: '2px' }}>
              {sample.filename}
            </div>
          </div>
        ) : (
          <img
            key={sample.sample_id}
            src={imageUrl}
            alt={sample.filename}
            onError={() => setImgError(true)}
            onLoad={() => setImgError(false)}
            style={{
              width: '100%',
              height: '100%',
              objectFit: 'contain',
              transform: `scale(${zoom})`,
              transition: 'transform 120ms ease-out',
              userSelect: 'none',
            }}
          />
        )}

        {/* Bottom HUD Bar inside image */}
        <div style={{
          position: 'absolute',
          bottom: '8px',
          left: '12px',
          right: '12px',
          display: 'flex',
          justifyContent: 'space-between',
          fontFamily: 'var(--font-mono)',
          fontSize: '9px',
          color: 'var(--text-muted)',
          zIndex: 9,
          pointerEvents: 'none',
        }}>
          <span>DIM: 256 × 256 PX</span>
          <span>SOL 4120 // 14:42:00 LMST</span>
          <span>MMCL STANDOFF: 25.4 CM</span>
        </div>
      </div>

      {/* Control Strip below Viewport */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '6px 12px',
        backgroundColor: 'var(--color-charcoal)',
        borderBottom: '1px solid var(--border-hairline)',
        fontFamily: 'var(--font-mono)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          {onOpenSampleDrawer && (
            <button
              onClick={onOpenSampleDrawer}
              className="btn btn-primary"
              style={{ padding: '4px 10px', fontSize: '10px' }}
            >
              [ SELECT SAMPLE ]
            </button>
          )}

          {onSelectRandomSample && (
            <button
              onClick={onSelectRandomSample}
              className="btn btn-secondary"
              style={{ padding: '4px 8px', fontSize: '10px' }}
            >
              RANDOM
            </button>
          )}
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '10px', color: 'var(--text-muted)' }}>
          <span>ZOOM TOGGLE</span>
          <span>•</span>
          <span>GRID ON</span>
          <span>•</span>
          <span style={{ color: 'var(--color-amber)' }}>SPECTRAL</span>
        </div>
      </div>

      {/* Telemetry Record & Ground Truth Bar */}
      <div style={{
        padding: '8px 12px',
        backgroundColor: 'var(--color-surface)',
        display: 'flex',
        flexDirection: 'column',
        gap: '6px',
        fontSize: '11px',
        fontFamily: 'var(--font-mono)',
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '9px', color: 'var(--text-muted)', textTransform: 'uppercase' }}>
          <span>TELEMETRY RECORD & GROUND TRUTH</span>
          <span style={{ color: 'var(--color-green-bright)' }}>ARCHIVE VERIFIED: PDS-GEOSCIENCES</span>
        </div>

        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(4, 1fr)',
          gap: '8px',
          padding: '6px 8px',
          backgroundColor: 'var(--color-obsidian)',
          border: '1px solid var(--border-hairline)',
          borderRadius: '2px',
        }}>
          <div>
            <div style={{ fontSize: '8px', color: 'var(--text-muted)' }}>SAMPLE ID</div>
            <div style={{ fontWeight: 800, color: 'var(--color-mars)', fontSize: '11px', marginTop: '1px' }}>
              {sample?.sample_id || '—'}
            </div>
          </div>

          <div>
            <div style={{ fontSize: '8px', color: 'var(--text-muted)' }}>GROUND TRUTH</div>
            <div style={{ fontWeight: 700, color: 'var(--color-green-bright)', fontSize: '11px', marginTop: '1px', textTransform: 'capitalize' }}>
              {sample?.ground_truth_name || '—'}
            </div>
          </div>

          <div>
            <div style={{ fontSize: '8px', color: 'var(--text-muted)' }}>RAW FILENAME</div>
            <div style={{
              color: 'var(--text-secondary)',
              fontSize: '10px',
              marginTop: '1px',
              whiteSpace: 'nowrap',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
            }} title={sample?.filename}>
              {sample?.filename || '—'}
            </div>
          </div>

          <div>
            <div style={{ fontSize: '8px', color: 'var(--text-muted)' }}>DATASET SPLIT</div>
            <div style={{ color: 'var(--color-amber)', fontSize: '10px', fontWeight: 700, marginTop: '1px' }}>
              {sample ? `${sample.split.toUpperCase()} SET` : '—'}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
