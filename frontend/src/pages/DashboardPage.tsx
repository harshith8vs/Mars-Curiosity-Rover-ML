import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowRight } from 'lucide-react';
import { BenchmarkTable } from '../components/BenchmarkTable';
import { api } from '../services/api';
import type { ModelInfo, SampleItem, DatasetStats } from '../types';

export const DashboardPage: React.FC = () => {
  const navigate = useNavigate();
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [stats, setStats] = useState<DatasetStats | null>(null);
  const [previewSamples, setPreviewSamples] = useState<SampleItem[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);

  useEffect(() => {
    let isMounted = true;
    const loadDashboardData = async () => {
      try {
        const [modelsData, statsData, samplesData] = await Promise.all([
          api.getModels(),
          api.getDatasetStats().catch(() => null),
          api.getSamples({ split: 'test', page: 1, pageSize: 4 }).catch(() => null),
        ]);
        if (isMounted) {
          setModels(modelsData);
          setStats(statsData);
          if (samplesData && samplesData.samples) {
            setPreviewSamples(samplesData.samples);
          }
        }
      } catch (err: unknown) {
        console.error('Failed to load dashboard telemetry:', err);
      } finally {
        if (isMounted) setIsLoading(false);
      }
    };

    loadDashboardData();
    return () => {
      isMounted = false;
    };
  }, []);

  const sortedClasses = stats?.classes
    ? [...stats.classes].sort((a, b) => b.total - a.total)
    : [];
  const maxCount = sortedClasses.length > 0 ? Math.max(...sortedClasses.map((c) => c.total), 1) : 1;
  const totalLabeled = stats?.total_labeled_images || 6691;

  return (
    <div style={{
      padding: '14px 18px',
      display: 'flex',
      flexDirection: 'column',
      gap: '14px',
      maxWidth: '1700px',
      margin: '0 auto',
      fontFamily: 'var(--font-mono)',
    }}>
      {/* 1. Hero Mission Header */}
      <div style={{
        backgroundColor: 'var(--color-surface)',
        border: '1px solid var(--border-structural)',
        borderRadius: '2px',
        padding: '16px 20px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: '16px',
      }}>
        <div style={{ maxWidth: '850px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '9px', color: 'var(--text-muted)', letterSpacing: '0.08em', textTransform: 'uppercase' }}>
            <span>NASA / MARS SCIENCE LAB</span>
            <span>•</span>
            <span>CURIOSITY ROVER</span>
            <span>•</span>
            <span>ML CLASSIFICATION SYSTEM</span>
          </div>

          <div style={{ fontSize: '10px', color: 'var(--color-mars)', fontWeight: 700, marginTop: '4px', letterSpacing: '0.05em' }}>
            ▲ MACHINE LEARNING ANALYSIS OF NASA CURIOSITY ROVER IMAGERY
          </div>

          <h1 style={{
            fontSize: '24px',
            fontWeight: 800,
            fontFamily: 'var(--font-heading)',
            color: '#FFFFFF',
            marginTop: '4px',
            letterSpacing: '0.02em',
          }}>
            MARTIAN SURFACE <span style={{ color: 'var(--color-mars)' }}>IMAGE</span> CLASSIFICATION
          </h1>

          <p style={{
            fontSize: '11px',
            color: 'var(--text-secondary)',
            fontFamily: 'var(--font-sans)',
            marginTop: '6px',
            lineHeight: 1.5,
          }}>
            Automated geological feature characterization and hardware context segmentation pipeline operating over Planetary Data System (PDS) archive imagery from the Mast Camera (Mastcam) and Mars Hand Lens Imager (MAHLI).
          </p>
        </div>

        {/* Right CTA & Mission Clock */}
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: '8px' }}>
          <button
            onClick={() => navigate('/')}
            className="btn btn-primary"
            style={{ padding: '10px 18px', fontSize: '11px', letterSpacing: '0.06em' }}
          >
            <span>[ ⟁ OPEN CLASSIFICATION LAB ]</span>
          </button>
          <div style={{ fontSize: '10px', color: 'var(--text-muted)' }}>
            ARCHIVE: <span style={{ color: 'var(--color-amber)', fontWeight: 700 }}>NASA PDS MARS ARCHIVE • 24 SCIENTIFIC TAXA</span>
          </div>
        </div>
      </div>

      {/* 2. Four Top Metrics Cards */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(4, 1fr)',
        gap: '8px',
      }}>
        {/* Metric 1 */}
        <div style={{
          padding: '10px 12px',
          backgroundColor: 'var(--color-surface)',
          border: '1px solid var(--border-structural)',
          borderRadius: '2px',
          display: 'flex',
          flexDirection: 'column',
          gap: '2px',
        }}>
          <div style={{ fontSize: '9px', color: 'var(--text-muted)' }}>
            04-01 // REPOSITORY
          </div>
          <div style={{ fontSize: '24px', fontWeight: 800, color: '#FFFFFF', lineHeight: 1.1 }}>
            6,691
          </div>
          <div style={{ fontSize: '10px', color: 'var(--color-mars)', fontWeight: 700, marginTop: '2px' }}>
            LABELED IMAGES
          </div>
          <div style={{ fontSize: '9px', color: 'var(--text-secondary)', fontFamily: 'var(--font-sans)' }}>
            Calibrated planetary dataset from MSL Mastcam & MAHLI archives.
          </div>
        </div>

        {/* Metric 2 */}
        <div style={{
          padding: '10px 12px',
          backgroundColor: 'var(--color-surface)',
          border: '1px solid var(--border-structural)',
          borderRadius: '2px',
          display: 'flex',
          flexDirection: 'column',
          gap: '2px',
        }}>
          <div style={{ fontSize: '9px', color: 'var(--text-muted)' }}>
            04-02 // TAXONOMY
          </div>
          <div style={{ fontSize: '24px', fontWeight: 800, color: 'var(--color-mars)', lineHeight: 1.1 }}>
            24
          </div>
          <div style={{ fontSize: '10px', color: 'var(--color-amber)', fontWeight: 700, marginTop: '2px' }}>
            ACTIVE CLASSES
          </div>
          <div style={{ fontSize: '9px', color: 'var(--text-secondary)', fontFamily: 'var(--font-sans)' }}>
            Martian geologic formations, regolith types, and rover hardware context.
          </div>
        </div>

        {/* Metric 3 */}
        <div style={{
          padding: '10px 12px',
          backgroundColor: 'var(--color-surface)',
          border: '1px solid var(--border-structural)',
          borderRadius: '2px',
          display: 'flex',
          flexDirection: 'column',
          gap: '2px',
        }}>
          <div style={{ fontSize: '9px', color: 'var(--text-muted)' }}>
            04-03 // PIPELINES
          </div>
          <div style={{ fontSize: '24px', fontWeight: 800, color: '#FFFFFF', lineHeight: 1.1 }}>
            9
          </div>
          <div style={{ fontSize: '10px', color: 'var(--color-green-bright)', fontWeight: 700, marginTop: '2px' }}>
            ML MODELS
          </div>
          <div style={{ fontSize: '9px', color: 'var(--text-secondary)', fontFamily: 'var(--font-sans)' }}>
            Classical statistical estimators alongside deep vision attention models.
          </div>
        </div>

        {/* Metric 4 */}
        <div style={{
          padding: '10px 12px',
          backgroundColor: 'var(--color-surface)',
          border: '1px solid var(--border-structural)',
          borderRadius: '2px',
          display: 'flex',
          flexDirection: 'column',
          gap: '2px',
        }}>
          <div style={{ fontSize: '9px', color: 'var(--text-muted)' }}>
            04-04 // PLATFORM
          </div>
          <div style={{ fontSize: '24px', fontWeight: 800, color: '#FFFFFF', lineHeight: 1.1 }}>
            1
          </div>
          <div style={{ fontSize: '10px', color: '#FFFFFF', fontWeight: 700, marginTop: '2px' }}>
            CURIOSITY ROVER
          </div>
          <div style={{ fontSize: '9px', color: 'var(--text-secondary)', fontFamily: 'var(--font-sans)' }}>
            Operating in Gale Crater since landing August 6, 2012 (Sol 0).
          </div>
        </div>
      </div>

      {/* 3. Middle Section: Model Benchmark Matrix (Left) + Class Distribution & Telemetry (Right) */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'minmax(600px, 1.4fr) minmax(320px, 0.9fr)',
        gap: '12px',
        alignItems: 'start',
      }}>
        {/* Left: Benchmark Table */}
        <BenchmarkTable
          models={models}
          isLoading={isLoading}
        />

        {/* Right: Full 24 Active Class Distribution */}
        <div style={{
          backgroundColor: 'var(--color-surface)',
          border: '1px solid var(--border-structural)',
          borderRadius: '2px',
          overflow: 'hidden',
          display: 'flex',
          flexDirection: 'column',
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
            flexWrap: 'wrap',
            gap: '6px',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span style={{ color: 'var(--color-amber)', fontWeight: 800 }}>■</span>
              <span style={{ fontWeight: 800, color: '#FFFFFF', letterSpacing: '0.04em' }}>
                CLASS DISTRIBUTION (24 ACTIVE TAXA)
              </span>
            </div>
            <span style={{ color: 'var(--color-mars)', fontSize: '10px', fontWeight: 700 }}>
              {totalLabeled.toLocaleString()} TOTAL LABELED
            </span>
          </div>

          {/* Subheader description */}
          <div style={{
            padding: '6px 12px',
            backgroundColor: 'rgba(9, 11, 14, 0.5)',
            borderBottom: '1px solid var(--border-hairline)',
            fontSize: '9px',
            color: 'var(--text-muted)',
            display: 'flex',
            justifyContent: 'space-between',
          }}>
            <span>MARTIAN GEOLOGICAL FORMATIONS & HARDWARE CONTEXT</span>
            <span>COUNT / SHARE</span>
          </div>

          {/* Table list */}
          <div style={{
            padding: '4px 8px',
            display: 'flex',
            flexDirection: 'column',
            gap: '2px',
            maxHeight: '435px',
            overflowY: 'auto',
          }}>
            {sortedClasses.length > 0 ? (
              sortedClasses.map((cls, idx) => {
                const pct = ((cls.total / totalLabeled) * 100).toFixed(1);
                const barWidth = `${Math.max((cls.total / maxCount) * 100, 2)}%`;
                return (
                  <div
                    key={cls.class_id}
                    style={{
                      display: 'flex',
                      flexDirection: 'column',
                      padding: '5px 8px',
                      backgroundColor: idx % 2 === 0 ? 'rgba(255, 255, 255, 0.015)' : 'transparent',
                      borderBottom: '1px solid rgba(48, 54, 61, 0.25)',
                    }}
                  >
                    <div style={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      fontSize: '10px',
                    }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <span style={{ color: 'var(--text-muted)', fontSize: '9px', width: '16px', textAlign: 'right' }}>
                          {idx + 1}.
                        </span>
                        <span style={{ color: '#FFFFFF', fontWeight: 600, textTransform: 'capitalize' }}>
                          {cls.class_name}
                        </span>
                        <span style={{ color: 'var(--text-muted)', fontSize: '9px' }}>
                          [ID {cls.class_id}]
                        </span>
                      </div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                        <span style={{ color: 'var(--color-mars)', fontWeight: 700 }}>
                          {cls.total.toLocaleString()}
                        </span>
                        <span style={{ color: 'var(--color-amber)', fontSize: '9px', minWidth: '40px', textAlign: 'right' }}>
                          {pct}%
                        </span>
                      </div>
                    </div>

                    {/* Subtle horizontal proportion bar */}
                    <div style={{
                      height: '3px',
                      backgroundColor: 'rgba(255, 255, 255, 0.05)',
                      borderRadius: '1px',
                      overflow: 'hidden',
                      marginTop: '3px',
                    }}>
                      <div style={{
                        height: '100%',
                        width: barWidth,
                        backgroundColor: idx === 0 ? 'var(--color-mars)' : 'var(--border-structural)',
                        borderRadius: '1px',
                      }} />
                    </div>
                  </div>
                );
              })
            ) : (
              <div style={{ padding: '20px', textAlign: 'center', color: 'var(--text-muted)' }}>
                LOADING TAXONOMIC DISTRIBUTION...
              </div>
            )}
          </div>
        </div>
      </div>

      {/* 4. Bottom Section: PDS Image Artifact Catalog Previews */}
      <div style={{
        backgroundColor: 'var(--color-surface)',
        border: '1px solid var(--border-structural)',
        borderRadius: '2px',
        overflow: 'hidden',
      }}>
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
            <span style={{ color: 'var(--color-mars)', fontWeight: 800 }}>■</span>
            <span style={{ fontWeight: 800, color: '#FFFFFF' }}>
              PDS IMAGE ARTIFACT CATALOG (SOL 4120 - 4128 SAMPLES)
            </span>
          </div>

          <button
            onClick={() => navigate('/dataset-models')}
            style={{
              backgroundColor: 'transparent',
              border: 'none',
              color: 'var(--color-mars)',
              fontSize: '10px',
              fontWeight: 700,
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '4px',
              fontFamily: 'var(--font-mono)',
            }}
          >
            <span>EXPLORE COMPLETE SET</span>
            <ArrowRight size={12} />
          </button>
        </div>

        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(4, 1fr)',
          gap: '8px',
          padding: '10px 12px',
        }}>
          {previewSamples.map((s) => (
            <div
              key={s.sample_id}
              onClick={() => navigate(`/classification?sample_id=${s.sample_id}`)}
              style={{
                backgroundColor: 'var(--color-charcoal)',
                border: '1px solid var(--border-hairline)',
                borderRadius: '2px',
                overflow: 'hidden',
                cursor: 'pointer',
              }}
            >
              <div style={{ height: '90px', backgroundColor: '#000000', overflow: 'hidden', position: 'relative' }}>
                <img
                  src={api.getImageUrl(s.image_url)}
                  alt={s.sample_id}
                  style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                />
                <div style={{ position: 'absolute', top: '4px', left: '4px' }}>
                  <span style={{
                    backgroundColor: 'rgba(9, 11, 14, 0.85)',
                    color: '#FFFFFF',
                    fontSize: '8px',
                    padding: '1px 4px',
                    border: '1px solid var(--border-hairline)',
                  }}>
                    {s.sample_id}
                  </span>
                </div>
              </div>

              <div style={{ padding: '6px', display: 'flex', flexDirection: 'column', gap: '2px' }}>
                <div style={{ fontSize: '10px', fontWeight: 700, color: 'var(--color-mars)', textTransform: 'uppercase' }}>
                  {s.ground_truth_name}
                </div>
                <div style={{ fontSize: '8px', color: 'var(--text-muted)' }}>
                  TARGET: MSL-CALIB • TEST SET
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

    </div>
  );
};
