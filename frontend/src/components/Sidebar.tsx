import React from 'react';
import { Activity, Terminal, Disc } from 'lucide-react';

export const Sidebar: React.FC = () => {
  return (
    <aside style={{
      width: '180px',
      backgroundColor: 'var(--color-charcoal)',
      borderRight: '1px solid var(--border-hairline)',
      display: 'flex',
      flexDirection: 'column',
      justifyContent: 'space-between',
      padding: '14px 12px',
      fontFamily: 'var(--font-mono)',
      flexShrink: 0,
    }}>
      {/* Top Section */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
        <div>
          <div style={{ fontSize: '9px', color: 'var(--text-muted)', letterSpacing: '0.08em', textTransform: 'uppercase' }}>
            CORE SUBSYSTEMS
          </div>
          <div style={{ fontSize: '10px', color: 'var(--color-amber)', fontWeight: 700, letterSpacing: '0.04em' }}>
            INSTRUMENT ARRAYS
          </div>
        </div>

        {/* Subsystem Items */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            padding: '6px 8px',
            backgroundColor: 'var(--color-surface)',
            border: '1px solid var(--border-hairline)',
            borderRadius: '2px',
            fontSize: '11px',
            color: 'var(--text-primary)',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <Disc size={13} color="var(--color-mars)" />
              <span style={{ fontSize: '10px' }}>Spectral Lab</span>
            </div>
            <span className="badge badge-emerald" style={{ fontSize: '8px', padding: '1px 4px' }}>SYNC</span>
          </div>

          <div style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            padding: '6px 8px',
            backgroundColor: 'transparent',
            border: '1px solid var(--border-hairline)',
            borderRadius: '2px',
            fontSize: '11px',
            color: 'var(--text-secondary)',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <Activity size={13} color="var(--color-amber)" />
              <span style={{ fontSize: '10px' }}>Telemetry BBD</span>
            </div>
            <span className="badge badge-amber" style={{ fontSize: '8px', padding: '1px 4px' }}>NOM</span>
          </div>

          <div style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            padding: '6px 8px',
            backgroundColor: 'transparent',
            border: '1px solid var(--border-hairline)',
            borderRadius: '2px',
            fontSize: '11px',
            color: 'var(--text-secondary)',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <Terminal size={13} color="var(--text-muted)" />
              <span style={{ fontSize: '10px' }}>Inference Logs</span>
            </div>
            <span className="badge badge-neutral" style={{ fontSize: '8px', padding: '1px 4px' }}>V1.0</span>
          </div>
        </div>
      </div>

      {/* Bottom Coordinates & Location Box */}
      <div style={{
        padding: '10px',
        backgroundColor: 'var(--color-obsidian)',
        border: '1px solid var(--border-hairline)',
        borderRadius: '2px',
        display: 'flex',
        flexDirection: 'column',
        gap: '4px',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: '9px', color: 'var(--text-muted)' }}>
          <span>GALE CRATER LOC</span>
          <span style={{ color: 'var(--color-mars)' }}>TARGET #69</span>
        </div>
        <div style={{ fontSize: '10px', color: 'var(--text-primary)', fontWeight: 700 }}>
          -4.5895°S, 137.4417°E
        </div>
        <div style={{ fontSize: '9px', color: 'var(--text-secondary)' }}>
          ELEV: -4462M • ROLL 0.4°
        </div>
      </div>
    </aside>
  );
};
