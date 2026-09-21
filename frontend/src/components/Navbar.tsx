import React, { useEffect, useState } from 'react';
import { NavLink } from 'react-router-dom';
import { User } from 'lucide-react';
import { api } from '../services/api';
import type { HealthResponse } from '../types';

export const Navbar: React.FC = () => {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [isOnline, setIsOnline] = useState<boolean>(false);

  useEffect(() => {
    let isMounted = true;
    const check = async () => {
      try {
        const data = await api.getHealth();
        if (isMounted) {
          setHealth(data);
          setIsOnline(data.status === 'ok');
        }
      } catch {
        if (isMounted) {
          setIsOnline(false);
        }
      }
    };
    check();
    const interval = setInterval(check, 10000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  return (
    <header style={{
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      padding: '0 18px',
      height: '50px',
      backgroundColor: 'var(--color-charcoal)',
      borderBottom: '1px solid var(--border-hairline)',
      position: 'sticky',
      top: 0,
      zIndex: 100,
    }}>
      {/* Left: NASA / MSL Branding */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
        {/* MSL Circular Orange Badge */}
        <div style={{
          width: '28px',
          height: '28px',
          borderRadius: '50%',
          backgroundColor: 'var(--color-mars)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: '#090B0E',
          fontWeight: 900,
          fontFamily: 'var(--font-heading)',
          fontSize: '11px',
          letterSpacing: '-0.03em',
        }}>
          MSL
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', lineHeight: 1.15 }}>
          <span style={{
            fontSize: '9px',
            fontFamily: 'var(--font-mono)',
            color: 'var(--text-muted)',
            letterSpacing: '0.08em',
            textTransform: 'uppercase',
          }}>
            NASA / MARS SCIENCE LABORATORY
          </span>
          <span style={{
            fontFamily: 'var(--font-mono)',
            fontWeight: 700,
            fontSize: '11px',
            color: 'var(--text-primary)',
            letterSpacing: '0.04em',
          }}>
            CURIOSITY ROVER • ML CLASSIFICATION SYSTEM
          </span>
        </div>
      </div>

      {/* Middle: Mission Telemetry Readouts */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: '12px',
        fontSize: '10px',
        fontFamily: 'var(--font-mono)',
        color: 'var(--text-muted)',
      }}>
        {isOnline ? (
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: 'var(--color-green-bright)' }}>
            <span style={{ width: '6px', height: '6px', borderRadius: '50%', backgroundColor: 'var(--color-green-bright)' }} />
            <span style={{ fontWeight: 700 }}>SYSTEM ONLINE ({health?.models_available || 9} MODELS)</span>
          </div>
        ) : (
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: 'var(--color-error)' }}>
            <span style={{ width: '6px', height: '6px', borderRadius: '50%', backgroundColor: 'var(--color-error)' }} />
            <span style={{ fontWeight: 700 }}>API OFFLINE</span>
          </div>
        )}

        <span style={{ color: 'var(--border-structural)' }}>|</span>
        <span style={{ color: 'var(--text-secondary)' }}>6,691 LABELED IMAGES</span>
        <span style={{ color: 'var(--border-structural)' }}>|</span>
        <span style={{ color: 'var(--text-secondary)' }}>24 ACTIVE TAXA</span>
        <span style={{ color: 'var(--border-structural)' }}>|</span>
        <span style={{ color: 'var(--color-amber)' }}>
          {health?.test_samples ? `${health.test_samples.toLocaleString()} TEST SAMPLES` : '1,305 TEST SAMPLES'}
        </span>
      </div>

      {/* Right: Technical Navigation Tabs */}
      <nav style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
        <NavLink
          to="/dashboard"
          style={({ isActive }) => ({
            padding: '6px 14px',
            borderRadius: '2px',
            fontSize: '11px',
            fontFamily: 'var(--font-mono)',
            fontWeight: 700,
            textDecoration: 'none',
            textTransform: 'uppercase',
            letterSpacing: '0.05em',
            backgroundColor: isActive ? 'var(--color-mars)' : 'transparent',
            color: isActive ? '#090B0E' : 'var(--text-secondary)',
            border: isActive ? '1px solid var(--color-mars)' : '1px solid transparent',
            transition: 'all 0.12s ease',
          })}
        >
          MISSION DASHBOARD
        </NavLink>

        <NavLink
          to="/"
          end
          style={({ isActive }) => ({
            padding: '6px 14px',
            borderRadius: '2px',
            fontSize: '11px',
            fontFamily: 'var(--font-mono)',
            fontWeight: 700,
            textDecoration: 'none',
            textTransform: 'uppercase',
            letterSpacing: '0.05em',
            backgroundColor: isActive ? 'var(--color-mars)' : 'transparent',
            color: isActive ? '#090B0E' : 'var(--text-secondary)',
            border: isActive ? '1px solid var(--color-mars)' : '1px solid transparent',
            transition: 'all 0.12s ease',
          })}
        >
          CLASSIFICATION LAB
        </NavLink>

        <NavLink
          to="/dataset-models"
          style={({ isActive }) => ({
            padding: '6px 14px',
            borderRadius: '2px',
            fontSize: '11px',
            fontFamily: 'var(--font-mono)',
            fontWeight: 700,
            textDecoration: 'none',
            textTransform: 'uppercase',
            letterSpacing: '0.05em',
            backgroundColor: isActive ? 'var(--color-mars)' : 'transparent',
            color: isActive ? '#090B0E' : 'var(--text-secondary)',
            border: isActive ? '1px solid var(--color-mars)' : '1px solid transparent',
            transition: 'all 0.12s ease',
          })}
        >
          DATASET & MODELS
        </NavLink>

        {/* User Icon Circle */}
        <div style={{
          width: '26px',
          height: '26px',
          borderRadius: '50%',
          backgroundColor: 'var(--color-surface)',
          border: '1px solid var(--border-structural)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          marginLeft: '8px',
          color: 'var(--text-secondary)',
        }}>
          <User size={14} />
        </div>
      </nav>
    </header>
  );
};
