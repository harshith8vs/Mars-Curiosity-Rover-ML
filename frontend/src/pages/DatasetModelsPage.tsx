import React, { useState, useEffect } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import {
  Database,
  Cpu,
  Search,
  ChevronLeft,
  ChevronRight,
} from 'lucide-react';
import { api } from '../services/api';
import type {
  DatasetStats,
  ModelInfo,
  ModelPerformanceResponse,
  SampleItem,
  SampleListResponse,
} from '../types';

export const DatasetModelsPage: React.FC = () => {
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();

  // Active Top Tab: 'dataset' or 'models'
  const activeTab = (searchParams.get('tab') as 'dataset' | 'models') || 'dataset';
  const setActiveTab = (tab: 'dataset' | 'models') => {
    setSearchParams({ tab });
  };

  // State
  const [stats, setStats] = useState<DatasetStats | null>(null);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [performance, setPerformance] = useState<ModelPerformanceResponse | null>(null);
  const [isLoadingMain, setIsLoadingMain] = useState<boolean>(true);

  // Sample Explorer State
  const [samplesResponse, setSamplesResponse] = useState<SampleListResponse | null>(null);
  const [sampleSplit, setSampleSplit] = useState<'test' | 'train' | 'val'>('test');
  const [samplePage, setSamplePage] = useState<number>(1);
  const [samplePageSize, setSamplePageSize] = useState<number>(36);
  const [sampleSearch, setSampleSearch] = useState<string>('');
  const [sampleClassId, setSampleClassId] = useState<string>('all');
  const [isLoadingSamples, setIsLoadingSamples] = useState<boolean>(false);
  const [activeSelectedSample, setActiveSelectedSample] = useState<SampleItem | null>(null);

  // Initial load
  useEffect(() => {
    let isMounted = true;
    const loadOverview = async () => {
      try {
        const [statsData, modelsData, perfData] = await Promise.all([
          api.getDatasetStats(),
          api.getModels(),
          api.getPerformance().catch(() => null),
        ]);
        if (isMounted) {
          setStats(statsData);
          setModels(modelsData);
          setPerformance(perfData);
        }
      } catch (err: unknown) {
        console.error('Failed to load dataset and models metadata:', err);
      } finally {
        if (isMounted) setIsLoadingMain(false);
      }
    };
    loadOverview();
    return () => {
      isMounted = false;
    };
  }, []);

  // Fetch samples
  useEffect(() => {
    let isMounted = true;
    const fetchSamples = async () => {
      setIsLoadingSamples(true);
      try {
        const params: {
          split: 'test' | 'train' | 'val';
          page: number;
          pageSize: number;
          search?: string;
          classId?: number;
        } = {
          split: sampleSplit,
          page: samplePage,
          pageSize: samplePageSize,
        };

        if (sampleSearch.trim()) {
          params.search = sampleSearch.trim();
        }
        if (sampleClassId !== 'all') {
          params.classId = parseInt(sampleClassId, 10);
        }

        const data = await api.getSamples(params);
        if (isMounted) {
          setSamplesResponse(data);
          // Default active target to first sample if none selected
          if (!activeSelectedSample && data.samples.length > 0) {
            setActiveSelectedSample(data.samples[0]);
          }
        }
      } catch (err: unknown) {
        console.error('Failed to fetch samples:', err);
      } finally {
        if (isMounted) setIsLoadingSamples(false);
      }
    };

    fetchSamples();
    return () => {
      isMounted = false;
    };
  }, [sampleSplit, samplePage, samplePageSize, sampleSearch, sampleClassId]);

  // Transfer sample to Classification Lab
  const handleUseSampleInLab = (sample: SampleItem) => {
    navigate(`/classification?sample_id=${encodeURIComponent(sample.sample_id)}`, {
      state: { sample, sample_id: sample.sample_id },
    });
  };

  const classicalModels = models.filter((m) => m.category === 'classical_ml');
  const deepLearningModels = models.filter((m) => m.category === 'deep_learning');

  if (isLoadingMain) {
    return (
      <div style={{ padding: '60px', textAlign: 'center', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
        <div style={{
          width: '24px',
          height: '24px',
          border: '2px solid var(--border-structural)',
          borderTopColor: 'var(--color-mars)',
          borderRadius: '50%',
          animation: 'spin 0.8s linear infinite',
          margin: '0 auto 12px auto',
        }} />
        <div style={{ fontSize: '11px', letterSpacing: '0.06em' }}>
          INITIALIZING DATASET & MODEL REGISTRY...
        </div>
      </div>
    );
  }

  return (
    <div style={{
      padding: '14px 18px',
      display: 'flex',
      flexDirection: 'column',
      gap: '12px',
      maxWidth: '1700px',
      margin: '0 auto',
      fontFamily: 'var(--font-mono)',
    }}>
      {/* 1. Top Repository Status Header */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '6px 12px',
        backgroundColor: 'var(--color-charcoal)',
        border: '1px solid var(--border-hairline)',
        borderRadius: '2px',
        fontSize: '11px',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ color: 'var(--color-mars)' }}>●</span>
          <span style={{ color: 'var(--text-muted)' }}>REPOSITORY 900E // PDS-GEO-4124 | ARCHIVE SOL: MSL-RAW-CAL-V3</span>
          <span style={{ color: 'var(--border-structural)' }}>|</span>
          <span style={{ color: 'var(--color-green-bright)' }}>INDEX STATUS: VERIFIED SYNCHRONIZED</span>
        </div>

        {/* Top Tabs: [ DATASET EXPLORER ] and [ MODEL ARCHITECTURES (9) ] */}
        <div style={{ display: 'flex', gap: '4px' }}>
          <button
            onClick={() => setActiveTab('dataset')}
            style={{
              padding: '4px 12px',
              fontSize: '10px',
              fontFamily: 'var(--font-mono)',
              fontWeight: 700,
              textTransform: 'uppercase',
              borderRadius: '2px',
              border: activeTab === 'dataset' ? '1px solid var(--color-mars)' : '1px solid var(--border-hairline)',
              backgroundColor: activeTab === 'dataset' ? 'var(--color-mars)' : 'var(--color-surface)',
              color: activeTab === 'dataset' ? '#090B0E' : 'var(--text-secondary)',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
            }}
          >
            <Database size={12} />
            <span>DATASET EXPLORER</span>
          </button>

          <button
            onClick={() => setActiveTab('models')}
            style={{
              padding: '4px 12px',
              fontSize: '10px',
              fontFamily: 'var(--font-mono)',
              fontWeight: 700,
              textTransform: 'uppercase',
              borderRadius: '2px',
              border: activeTab === 'models' ? '1px solid var(--color-mars)' : '1px solid var(--border-hairline)',
              backgroundColor: activeTab === 'models' ? 'var(--color-mars)' : 'var(--color-surface)',
              color: activeTab === 'models' ? '#090B0E' : 'var(--text-secondary)',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
            }}
          >
            <Cpu size={12} />
            <span>MODEL ARCHITECTURES (9)</span>
          </button>
        </div>
      </div>

      {/* ========================================================================= */}
      {/* TAB 1: DATASET EXPLORER                                                   */}
      {/* ========================================================================= */}
      {activeTab === 'dataset' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
          
          {/* 4 Top Telemetry KPI Cards */}
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(4, 1fr)',
            gap: '8px',
          }}>
            {/* Card 1: Total Inventory */}
            <div style={{
              padding: '10px 12px',
              backgroundColor: 'var(--color-surface)',
              border: '1px solid var(--border-structural)',
              borderRadius: '2px',
              display: 'flex',
              flexDirection: 'column',
              gap: '2px',
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '9px', color: 'var(--text-muted)' }}>
                <span>TOTAL INVENTORY</span>
                <span style={{ color: 'var(--color-green-bright)' }}>100% VERIFIED</span>
              </div>
              <div style={{ fontSize: '24px', fontWeight: 800, color: '#FFFFFF', lineHeight: 1.1 }}>
                6,691
              </div>
              <div style={{ fontSize: '9px', color: 'var(--text-secondary)', fontFamily: 'var(--font-sans)', marginTop: '2px' }}>
                Calibrated radiometric surface science frames (6,737 total)
              </div>
            </div>

            {/* Card 2: Split Distribution */}
            <div style={{
              padding: '10px 12px',
              backgroundColor: 'var(--color-surface)',
              border: '1px solid var(--border-structural)',
              borderRadius: '2px',
              display: 'flex',
              flexDirection: 'column',
              gap: '2px',
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '9px', color: 'var(--text-muted)' }}>
                <span>SPLIT DISTRIBUTION</span>
                <span>TR / VA / TS</span>
              </div>
              <div style={{ fontSize: '15px', fontWeight: 700, color: 'var(--color-amber)', marginTop: '4px' }}>
                TRN: 3,746 | VAL: 1,640 | TST: 1,305
              </div>
              <div style={{ fontSize: '9px', color: 'var(--text-secondary)', fontFamily: 'var(--font-sans)', marginTop: '2px' }}>
                Stratified multi-class partitioning (56% / 24.5% / 19.5%)
              </div>
            </div>

            {/* Card 3: Surface Taxonomy */}
            <div style={{
              padding: '10px 12px',
              backgroundColor: 'var(--color-surface)',
              border: '1px solid var(--border-structural)',
              borderRadius: '2px',
              display: 'flex',
              flexDirection: 'column',
              gap: '2px',
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '9px', color: 'var(--text-muted)' }}>
                <span>SURFACE TAXONOMY</span>
                <span style={{ color: 'var(--color-mars)' }}>ACTIVE TAXA</span>
              </div>
              <div style={{ fontSize: '24px', fontWeight: 800, color: 'var(--color-mars)', lineHeight: 1.1 }}>
                24
              </div>
              <div style={{ fontSize: '9px', color: 'var(--text-secondary)', fontFamily: 'var(--font-sans)', marginTop: '2px' }}>
                Sedimentary, basaltic, regolith & rover hardware
              </div>
            </div>

            {/* Card 4: Feature Representation */}
            <div style={{
              padding: '10px 12px',
              backgroundColor: 'var(--color-surface)',
              border: '1px solid var(--border-structural)',
              borderRadius: '2px',
              display: 'flex',
              flexDirection: 'column',
              gap: '2px',
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '9px', color: 'var(--text-muted)' }}>
                <span>FEATURE VECTOR & FORMAT</span>
                <span>DESCRIPTORS</span>
              </div>
              <div style={{ fontSize: '15px', fontWeight: 700, color: '#FFFFFF', marginTop: '4px' }}>
                8,186 DIM • 256×256
              </div>
              <div style={{ fontSize: '9px', color: 'var(--text-secondary)', fontFamily: 'var(--font-sans)', marginTop: '2px' }}>
                6,519 RGB · 172 Grayscale (HOG, GLCM, Color)
              </div>
            </div>
          </div>

          {/* Feature Extraction Breakdown Panel */}
          <div style={{
            backgroundColor: 'var(--color-surface)',
            border: '1px solid var(--border-structural)',
            borderRadius: '2px',
            padding: '10px 12px',
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', color: 'var(--text-muted)', marginBottom: '8px' }}>
              <span style={{ color: 'var(--color-mars)', fontWeight: 700 }}>■ FEATURE EXTRACTION BREAKDOWN (8,186 FEATURES)</span>
              <span>SCIKIT-LEARN CLASSICAL PIPELINE INPUTS</span>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: '6px', textAlign: 'center' }}>
              <div style={{ padding: '6px', backgroundColor: 'var(--color-obsidian)', border: '1px solid var(--border-hairline)' }}>
                <div style={{ fontSize: '8px', color: 'var(--text-muted)' }}>COLOR STATS</div>
                <div style={{ fontSize: '14px', fontWeight: 800, color: 'var(--color-mars)', marginTop: '2px' }}>19</div>
              </div>
              <div style={{ padding: '6px', backgroundColor: 'var(--color-obsidian)', border: '1px solid var(--border-hairline)' }}>
                <div style={{ fontSize: '8px', color: 'var(--text-muted)' }}>RGB HISTOGRAMS</div>
                <div style={{ fontSize: '14px', fontWeight: 800, color: 'var(--color-amber)', marginTop: '2px' }}>48</div>
              </div>
              <div style={{ padding: '6px', backgroundColor: 'var(--color-obsidian)', border: '1px solid var(--border-hairline)' }}>
                <div style={{ fontSize: '8px', color: 'var(--text-muted)' }}>GLCM TEXTURE</div>
                <div style={{ fontSize: '14px', fontWeight: 800, color: 'var(--color-green-bright)', marginTop: '2px' }}>12</div>
              </div>
              <div style={{ padding: '6px', backgroundColor: 'var(--color-obsidian)', border: '1px solid var(--border-hairline)' }}>
                <div style={{ fontSize: '8px', color: 'var(--text-muted)' }}>HOG DESCRIPTORS</div>
                <div style={{ fontSize: '14px', fontWeight: 800, color: '#FFFFFF', marginTop: '2px' }}>8,100</div>
              </div>
              <div style={{ padding: '6px', backgroundColor: 'var(--color-obsidian)', border: '1px solid var(--border-hairline)' }}>
                <div style={{ fontSize: '8px', color: 'var(--text-muted)' }}>EDGE ENERGY</div>
                <div style={{ fontSize: '14px', fontWeight: 800, color: 'var(--text-secondary)', marginTop: '2px' }}>2</div>
              </div>
              <div style={{ padding: '6px', backgroundColor: 'var(--color-obsidian)', border: '1px solid var(--border-hairline)' }}>
                <div style={{ fontSize: '8px', color: 'var(--text-muted)' }}>GRADIENTS</div>
                <div style={{ fontSize: '14px', fontWeight: 800, color: 'var(--color-error)', marginTop: '2px' }}>5</div>
              </div>
            </div>
          </div>

          {/* Filter Toolbar */}
          <div style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            padding: '8px 12px',
            backgroundColor: 'var(--color-charcoal)',
            border: '1px solid var(--border-hairline)',
            borderRadius: '2px',
            gap: '8px',
            flexWrap: 'wrap',
          }}>
            {/* Search Input */}
            <div style={{ position: 'relative', display: 'flex', alignItems: 'center' }}>
              <Search size={12} color="var(--text-muted)" style={{ position: 'absolute', left: '8px' }} />
              <input
                type="text"
                placeholder="Filter by Sample ID, Filename..."
                value={sampleSearch}
                onChange={(e) => {
                  setSampleSearch(e.target.value);
                  setSamplePage(1);
                }}
                style={{
                  backgroundColor: 'var(--color-obsidian)',
                  border: '1px solid var(--border-structural)',
                  borderRadius: '2px',
                  padding: '5px 8px 5px 26px',
                  fontSize: '11px',
                  fontFamily: 'var(--font-mono)',
                  color: 'var(--text-primary)',
                  outline: 'none',
                  width: '260px',
                }}
              />
            </div>

            {/* Class Dropdown */}
            {stats && (
              <select
                value={sampleClassId}
                onChange={(e) => {
                  setSampleClassId(e.target.value);
                  setSamplePage(1);
                }}
                style={{
                  backgroundColor: 'var(--color-obsidian)',
                  border: '1px solid var(--border-structural)',
                  borderRadius: '2px',
                  padding: '5px 8px',
                  fontSize: '11px',
                  fontFamily: 'var(--font-mono)',
                  color: 'var(--text-primary)',
                  outline: 'none',
                  cursor: 'pointer',
                  maxWidth: '220px',
                }}
              >
                <option value="all">ALL 24 TAXA</option>
                {stats.classes.map((cls) => (
                  <option key={cls.class_id} value={cls.class_id}>
                    {cls.class_id}: {cls.class_name.toUpperCase()}
                  </option>
                ))}
              </select>
            )}

            {/* Split Filter Buttons */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
              {(['test', 'train', 'val'] as const).map((s) => (
                <button
                  key={s}
                  onClick={() => {
                    setSampleSplit(s);
                    setSamplePage(1);
                  }}
                  style={{
                    padding: '4px 10px',
                    fontSize: '10px',
                    fontFamily: 'var(--font-mono)',
                    fontWeight: 700,
                    textTransform: 'uppercase',
                    borderRadius: '2px',
                    border: sampleSplit === s ? '1px solid var(--color-mars)' : '1px solid var(--border-hairline)',
                    backgroundColor: sampleSplit === s ? 'var(--color-mars)' : 'var(--color-surface)',
                    color: sampleSplit === s ? '#090B0E' : 'var(--text-secondary)',
                    cursor: 'pointer',
                  }}
                >
                  {s.toUpperCase()}
                </button>
              ))}
            </div>

            {/* Page Size Selector */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '10px', color: 'var(--text-muted)' }}>
              <span>PER PAGE:</span>
              <select
                value={samplePageSize}
                onChange={(e) => {
                  setSamplePageSize(Number(e.target.value));
                  setSamplePage(1);
                }}
                style={{
                  backgroundColor: 'var(--color-obsidian)',
                  border: '1px solid var(--border-structural)',
                  borderRadius: '2px',
                  padding: '3px 6px',
                  fontSize: '10px',
                  fontFamily: 'var(--font-mono)',
                  color: 'var(--text-primary)',
                  cursor: 'pointer',
                }}
              >
                <option value={12}>12</option>
                <option value={24}>24</option>
                <option value={36}>36 (DEFAULT)</option>
                <option value={48}>48</option>
              </select>
            </div>
          </div>

          {/* Sample Cards Grid */}
          {isLoadingSamples ? (
            <div style={{ padding: '60px 0', textAlign: 'center', color: 'var(--text-muted)' }}>
              INDEXING ARCHIVE RECONNAISSANCE FRAMES...
            </div>
          ) : (
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fill, minmax(210px, 1fr))',
              gap: '8px',
            }}>
              {samplesResponse?.samples.map((sample) => {
                const isSelected = activeSelectedSample?.sample_id === sample.sample_id;
                return (
                  <div
                    key={sample.sample_id}
                    onClick={() => {
                      setActiveSelectedSample(sample);
                      handleUseSampleInLab(sample);
                    }}
                    style={{
                      backgroundColor: 'var(--color-surface)',
                      border: isSelected ? '1px solid var(--color-mars)' : '1px solid var(--border-structural)',
                      borderRadius: '2px',
                      overflow: 'hidden',
                      cursor: 'pointer',
                      display: 'flex',
                      flexDirection: 'column',
                    }}
                  >
                    {/* Image Box */}
                    <div style={{ position: 'relative', height: '115px', backgroundColor: '#000000', overflow: 'hidden' }}>
                      <img
                        src={api.getImageUrl(sample.image_url)}
                        alt={sample.sample_id}
                        loading="lazy"
                        style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                      />
                      <div style={{ position: 'absolute', top: '4px', right: '4px' }}>
                        <span className="badge badge-amber" style={{ fontSize: '8px', padding: '1px 4px' }}>
                          {sample.split.toUpperCase()}
                        </span>
                      </div>
                      <div style={{ position: 'absolute', bottom: '4px', left: '4px' }}>
                        <span style={{
                          backgroundColor: 'rgba(9, 11, 14, 0.85)',
                          color: '#FFFFFF',
                          fontSize: '8px',
                          padding: '1px 4px',
                          border: '1px solid var(--border-hairline)',
                        }}>
                          {sample.sample_id}
                        </span>
                      </div>
                    </div>

                    {/* Metadata Card Footer */}
                    <div style={{ padding: '8px', display: 'flex', flexDirection: 'column', gap: '4px', flex: 1, justifyContent: 'space-between' }}>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                          <span style={{ fontSize: '10px', fontWeight: 800, color: '#FFFFFF' }}>
                            {sample.sample_id}
                          </span>
                          <span style={{ color: 'var(--color-green-bright)', fontSize: '8px' }}>
                            CALIBRATED
                          </span>
                        </div>
                        <div style={{
                          fontSize: '11px',
                          fontWeight: 700,
                          color: 'var(--color-mars)',
                          textTransform: 'uppercase',
                          whiteSpace: 'nowrap',
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                        }}>
                          {sample.ground_truth_name}
                        </div>
                        <div style={{ fontSize: '8px', color: 'var(--text-muted)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                          {sample.filename}
                        </div>
                      </div>

                      {/* Technical Action Button on Every Card */}
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleUseSampleInLab(sample);
                        }}
                        className="btn btn-secondary"
                        style={{
                          width: '100%',
                          padding: '5px 8px',
                          fontSize: '9px',
                          letterSpacing: '0.04em',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          gap: '6px',
                          marginTop: '4px',
                          color: 'var(--color-mars)',
                          borderColor: 'rgba(242, 100, 25, 0.4)',
                          backgroundColor: 'rgba(242, 100, 25, 0.06)',
                        }}
                        onMouseEnter={(e) => {
                          e.currentTarget.style.backgroundColor = 'var(--color-mars)';
                          e.currentTarget.style.color = '#090B0E';
                        }}
                        onMouseLeave={(e) => {
                          e.currentTarget.style.backgroundColor = 'rgba(242, 100, 25, 0.06)';
                          e.currentTarget.style.color = 'var(--color-mars)';
                        }}
                      >
                        <span>[ USE THIS SAMPLE IN LAB → ]</span>
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>
          )}

          {/* Pagination Controls */}
          {samplesResponse && samplesResponse.total_pages > 1 && (
            <div style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              padding: '8px 12px',
              backgroundColor: 'var(--color-charcoal)',
              border: '1px solid var(--border-hairline)',
              borderRadius: '2px',
              fontSize: '11px',
              color: 'var(--text-muted)',
            }}>
              <div>
                PAGE <span style={{ color: '#FFFFFF', fontWeight: 700 }}>{samplesResponse.page}</span> / {samplesResponse.total_pages} ({samplesResponse.total.toLocaleString()} SAMPLES)
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <button
                  disabled={samplePage <= 1}
                  onClick={() => setSamplePage((p) => Math.max(1, p - 1))}
                  className="btn btn-secondary"
                  style={{ padding: '3px 8px', fontSize: '10px' }}
                >
                  <ChevronLeft size={12} /> PREV
                </button>
                <button
                  disabled={samplePage >= samplesResponse.total_pages}
                  onClick={() => setSamplePage((p) => Math.min(samplesResponse.total_pages, p + 1))}
                  className="btn btn-secondary"
                  style={{ padding: '3px 8px', fontSize: '10px' }}
                >
                  NEXT <ChevronRight size={12} />
                </button>
              </div>
            </div>
          )}

        </div>
      )}

      {/* ========================================================================= */}
      {/* TAB 2: MODEL ARCHITECTURES & CANONICAL BENCHMARKS                         */}
      {/* ========================================================================= */}
      {activeTab === 'models' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
          
          {/* Canonical Benchmark Source Banner */}
          <div style={{
            padding: '8px 12px',
            backgroundColor: 'var(--color-charcoal)',
            border: '1px solid var(--border-hairline)',
            borderRadius: '2px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            fontSize: '11px',
          }}>
            <div>
              <span style={{ color: 'var(--text-muted)' }}>BENCHMARK SOURCE: </span>
              <span style={{ color: 'var(--color-mars)', fontWeight: 700 }}>
                results/10_Overall_Comparison/final_test_comparison.json
              </span>
            </div>
            <div style={{ display: 'flex', gap: '6px' }}>
              <span className="badge badge-emerald">OFFICIAL TEST-SET PERFORMANCE</span>
              <span className="badge badge-amber">1,305 SAMPLES</span>
            </div>
          </div>

          {/* Group 1: Classical Machine Learning */}
          <div style={{
            backgroundColor: 'var(--color-surface)',
            border: '1px solid var(--border-structural)',
            borderRadius: '2px',
            padding: '12px',
          }}>
            <div style={{ fontSize: '11px', fontWeight: 800, color: 'var(--color-amber)', letterSpacing: '0.04em', marginBottom: '8px' }}>
              // GROUP 1: CLASSICAL MACHINE LEARNING (5 SCIKIT-LEARN BASELINES)
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '8px' }}>
              {classicalModels.map((m) => (
                <div
                  key={m.id}
                  style={{
                    backgroundColor: 'var(--color-charcoal)',
                    border: '1px solid var(--border-hairline)',
                    padding: '8px 10px',
                    borderRadius: '2px',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '4px',
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ fontWeight: 800, fontSize: '12px', color: '#FFFFFF' }}>{m.name}</span>
                    <span className="badge badge-amber" style={{ fontSize: '8px' }}>CLASSICAL</span>
                  </div>
                  <div style={{ fontSize: '9px', color: 'var(--color-amber)' }}>
                    CONFIG: {m.configuration}
                  </div>
                  <div style={{ fontSize: '9px', color: 'var(--text-muted)' }}>
                    INPUT: {m.input_representation}
                  </div>
                  {m.benchmark && (
                    <div style={{
                      marginTop: '4px',
                      paddingTop: '4px',
                      borderTop: '1px solid var(--border-hairline)',
                      display: 'flex',
                      justifyContent: 'space-between',
                      fontSize: '9px',
                    }}>
                      <span style={{ color: '#FFFFFF', fontWeight: 700 }}>ACC: {m.benchmark.accuracy_pct.toFixed(2)}%</span>
                      <span style={{ color: 'var(--color-mars)' }}>F1: {m.benchmark.macro_f1.toFixed(4)}</span>
                      <span style={{ color: 'var(--text-muted)' }}>REC: {m.benchmark.macro_recall.toFixed(4)}</span>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>

          {/* Group 2: Deep Learning Architectures */}
          <div style={{
            backgroundColor: 'var(--color-surface)',
            border: '1px solid var(--border-structural)',
            borderRadius: '2px',
            padding: '12px',
          }}>
            <div style={{ fontSize: '11px', fontWeight: 800, color: 'var(--color-mars)', letterSpacing: '0.04em', marginBottom: '8px' }}>
              // GROUP 2: DEEP LEARNING (4 PYTORCH ARCHITECTURES)
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '8px' }}>
              {deepLearningModels.map((m) => {
                const cleanName = m.name.replace(/\[Champion\]\s*/i, '').replace(/Champion\s*/i, '').trim();
                return (
                  <div
                    key={m.id}
                    style={{
                      backgroundColor: 'var(--color-charcoal)',
                      border: '1px solid var(--border-hairline)',
                      padding: '8px 10px',
                      borderRadius: '2px',
                      display: 'flex',
                      flexDirection: 'column',
                      gap: '4px',
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <span style={{ fontWeight: 800, fontSize: '12px', color: '#FFFFFF' }}>{cleanName}</span>
                      <span className="badge badge-orange" style={{ fontSize: '8px' }}>DEEP LEARNING</span>
                    </div>
                    <div style={{ fontSize: '9px', color: 'var(--color-mars)' }}>
                      CONFIG: {m.configuration}
                    </div>
                    <div style={{ fontSize: '9px', color: 'var(--text-muted)' }}>
                      INPUT: {m.input_representation}
                    </div>
                    {m.benchmark && (
                      <div style={{
                        marginTop: '4px',
                        paddingTop: '4px',
                        borderTop: '1px solid var(--border-hairline)',
                        display: 'flex',
                        justifyContent: 'space-between',
                        fontSize: '9px',
                      }}>
                        <span style={{ color: '#FFFFFF', fontWeight: 700 }}>ACC: {m.benchmark.accuracy_pct.toFixed(2)}%</span>
                        <span style={{ color: 'var(--color-mars)' }}>F1: {m.benchmark.macro_f1.toFixed(4)}</span>
                        <span style={{ color: 'var(--text-muted)' }}>REC: {m.benchmark.macro_recall.toFixed(4)}</span>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>

          {/* Canonical Official Test Leaderboard Table */}
          {performance && (
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
                <span style={{ fontWeight: 800, color: '#FFFFFF' }}>
                  OFFICIAL 9-MODEL TEST-SET BENCHMARK MATRIX
                </span>
                <span style={{ color: 'var(--color-green-bright)' }}>EVALUATED ON 1,305 SAMPLES</span>
              </div>

              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '11px' }}>
                  <thead>
                    <tr style={{ borderBottom: '1px solid var(--border-structural)', backgroundColor: 'var(--color-charcoal)' }}>
                      <th style={{ padding: '6px 10px', color: 'var(--text-muted)' }}>MODEL</th>
                      <th style={{ padding: '6px 10px', color: 'var(--text-muted)' }}>CATEGORY</th>
                      <th style={{ padding: '6px 10px', color: 'var(--text-muted)' }}>TEST SAMPLES</th>
                      <th style={{ padding: '6px 10px', color: 'var(--text-muted)' }}>ACCURACY</th>
                      <th style={{ padding: '6px 10px', color: 'var(--text-muted)' }}>MACRO-F1</th>
                      <th style={{ padding: '6px 10px', color: 'var(--text-muted)' }}>WEIGHTED-F1</th>
                      <th style={{ padding: '6px 10px', color: 'var(--text-muted)' }}>PRECISION</th>
                      <th style={{ padding: '6px 10px', color: 'var(--text-muted)' }}>RECALL</th>
                    </tr>
                  </thead>
                  <tbody>
                    {performance.models.map((bm) => {
                      const cleanName = bm.model.replace(/\(Champion\)/i, '').trim();
                      const isDL = bm.category.toLowerCase().includes('deep');
                      return (
                        <tr key={bm.key} style={{ borderBottom: '1px solid var(--border-hairline)' }}>
                          <td style={{ padding: '6px 10px', fontWeight: 700, color: '#FFFFFF' }}>{cleanName}</td>
                          <td style={{ padding: '6px 10px' }}>
                            <span style={{ fontSize: '9px', color: isDL ? 'var(--color-mars)' : 'var(--color-amber)' }}>
                              {bm.category}
                            </span>
                          </td>
                          <td style={{ padding: '6px 10px', color: 'var(--text-secondary)' }}>{bm.n_test_samples}</td>
                          <td style={{ padding: '6px 10px', fontWeight: 700, color: '#FFFFFF' }}>{bm.accuracy_pct.toFixed(2)}%</td>
                          <td style={{ padding: '6px 10px', fontWeight: 700, color: 'var(--color-mars)' }}>{bm.macro_f1.toFixed(4)}</td>
                          <td style={{ padding: '6px 10px', color: 'var(--color-amber)' }}>{bm.weighted_f1.toFixed(4)}</td>
                          <td style={{ padding: '6px 10px', color: 'var(--text-secondary)' }}>{bm.macro_precision.toFixed(4)}</td>
                          <td style={{ padding: '6px 10px', color: 'var(--text-secondary)' }}>{bm.macro_recall.toFixed(4)}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}

        </div>
      )}

    </div>
  );
};
