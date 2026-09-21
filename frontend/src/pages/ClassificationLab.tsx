import React, { useState, useEffect } from 'react';
import { useLocation, useSearchParams } from 'react-router-dom';
import { Play, Sparkles, AlertTriangle, ChevronDown, ChevronUp } from 'lucide-react';
import { ViewportReticle } from '../components/ViewportReticle';
import { ModelCard } from '../components/ModelCard';
import { ResultPanel } from '../components/ResultPanel';
import { SampleDrawer } from '../components/SampleDrawer';
import { api } from '../services/api';
import type { ClassifyResponse, ModelInfo, SampleItem } from '../types';

export const ClassificationLab: React.FC = () => {
  const location = useLocation();
  const [searchParams, setSearchParams] = useSearchParams();

  // Extract URL query parameters
  const sampleIdParam = searchParams.get('sample_id') || searchParams.get('sampleId');
  const modelParam = searchParams.get('model') || searchParams.get('model_id');

  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedModel, setSelectedModel] = useState<ModelInfo | null>(null);
  const [selectedSample, setSelectedSample] = useState<SampleItem | null>(null);
  const [isLoadingSample, setIsLoadingSample] = useState<boolean>(false);
  const [sampleError, setSampleError] = useState<string | null>(null);
  const [isDrawerOpen, setIsDrawerOpen] = useState<boolean>(false);

  const [result, setResult] = useState<ClassifyResponse | null>(null);
  const [isClassifying, setIsClassifying] = useState<boolean>(false);
  const [classificationError, setClassificationError] = useState<string | null>(null);
  const [isLoadingModels, setIsLoadingModels] = useState<boolean>(true);

  // 1. Fetch & Initialize Models
  useEffect(() => {
    let isMounted = true;
    const fetchModels = async () => {
      try {
        const data = await api.getModels();
        if (isMounted) {
          setModels(data);

          if (modelParam) {
            const found = data.find((m) => m.id.toLowerCase() === modelParam.toLowerCase());
            if (found) {
              setSelectedModel(found);
              return;
            }
          }

          // Default to EfficientNet-B3
          const effNet = data.find((m) => m.id === 'efficientnet_b3');
          setSelectedModel(effNet || data[0] || null);
        }
      } catch (err: unknown) {
        console.error('Failed to load models:', err);
      } finally {
        if (isMounted) setIsLoadingModels(false);
      }
    };

    fetchModels();
    return () => {
      isMounted = false;
    };
  }, [modelParam]);

  // 2. Fetch & Initialize Sample based on URL param or navigation state
  useEffect(() => {
    let isMounted = true;
    const navState = location.state as { sample?: SampleItem; sample_id?: string; sampleId?: string } | null;
    const targetSampleId = sampleIdParam || navState?.sample_id || navState?.sampleId;

    if (targetSampleId) {
      if (navState?.sample && navState.sample.sample_id === targetSampleId) {
        setSelectedSample(navState.sample);
        setSampleError(null);
        setResult(null);
        if (!sampleIdParam) {
          setSearchParams((prev) => {
            const updated = new URLSearchParams(prev);
            updated.set('sample_id', targetSampleId);
            return updated;
          }, { replace: true });
        }
        return;
      }

      setIsLoadingSample(true);
      setSampleError(null);

      api.getSample(targetSampleId)
        .then((sampleData) => {
          if (isMounted) {
            setSelectedSample(sampleData);
            setSampleError(null);
            setResult(null);
          }
        })
        .catch((err) => {
          if (isMounted) {
            console.error(`Failed to fetch sample ${targetSampleId}:`, err);
            setSelectedSample(null);
            setSampleError(`Sample "${targetSampleId}" was not found in the dataset archive.`);
            setResult(null);
          }
        })
        .finally(() => {
          if (isMounted) setIsLoadingSample(false);
        });
    } else {
      if (!selectedSample) {
        setIsLoadingSample(true);
        api.getSamples({ split: 'test', page: 1, pageSize: 1 })
          .then((res) => {
            if (isMounted && res.samples && res.samples.length > 0) {
              const defaultSample = res.samples[0];
              setSelectedSample(defaultSample);
              setSampleError(null);
              setSearchParams((prev) => {
                const updated = new URLSearchParams(prev);
                updated.set('sample_id', defaultSample.sample_id);
                return updated;
              }, { replace: true });
            }
          })
          .catch((err) => {
            if (isMounted) {
              setSampleError('Unable to load initial test sample: ' + (err instanceof Error ? err.message : 'API unavailable'));
            }
          })
          .finally(() => {
            if (isMounted) setIsLoadingSample(false);
          });
      }
    }

    return () => {
      isMounted = false;
    };
  }, [sampleIdParam, location.state]);

  const handleSelectSample = (sample: SampleItem) => {
    setSelectedSample(sample);
    setSampleError(null);
    setResult(null);
    setSearchParams((prev) => {
      const updated = new URLSearchParams(prev);
      updated.set('sample_id', sample.sample_id);
      return updated;
    }, { replace: true });
  };

  const handleSelectRandomSample = () => {
    const randomIdx = Math.floor(Math.random() * 1305);
    const sid = `TEST-${String(randomIdx).padStart(4, '0')}`;
    api.getSample(sid).then((s) => {
      handleSelectSample(s);
    }).catch(() => {});
  };

  const handleSelectModel = (model: ModelInfo) => {
    setSelectedModel(model);
    setResult(null);
    setSearchParams((prev) => {
      const updated = new URLSearchParams(prev);
      updated.set('model', model.id);
      return updated;
    }, { replace: true });
  };

  const isTestSplit = selectedSample?.split === 'test';
  const hasCheckpoint = selectedModel?.has_checkpoint ?? false;
  const canRunInference = isTestSplit || hasCheckpoint;

  const cannotRunReason = !canRunInference
    ? 'TRAIN/VAL samples require a local model checkpoint for live inference.'
    : undefined;

  const handleClassify = async () => {
    if (!selectedSample || !selectedModel || !canRunInference || sampleError) return;

    setIsClassifying(true);
    setClassificationError(null);
    try {
      const res = await api.classify(selectedSample.sample_id, selectedModel.id);
      setResult(res);
    } catch (err: unknown) {
      setClassificationError(err instanceof Error ? err.message : 'Classification failed');
    } finally {
      setIsClassifying(false);
    }
  };

  const dlModels = models.filter((m) => m.category === 'deep_learning');
  const clModels = models.filter((m) => m.category === 'classical_ml');

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
      {/* 1. Top Analysis Protocol Status Strip */}
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
          <span style={{ color: 'var(--text-muted)' }}>ANALYSIS PROTOCOL:</span>
          <span style={{ color: 'var(--color-mars)', fontWeight: 700 }}>
            [ ⟁ SURFACE REGOLITH & BEDROCK IDENTIFIER ]
          </span>
          <span className="badge badge-neutral" style={{ fontSize: '9px', padding: '1px 5px' }}>
            SUBSYSTEM: SCI-ML-VX
          </span>
        </div>

        {/* Workflow Breadcrumbs */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '10px' }}>
          <span style={{ color: selectedSample ? 'var(--color-green-bright)' : 'var(--text-muted)' }}>
            01 SELECT SAMPLE [✓]
          </span>
          <span style={{ color: 'var(--border-structural)' }}>→</span>
          <span style={{ color: selectedModel ? 'var(--color-green-bright)' : 'var(--text-muted)' }}>
            02 SELECT MODEL [✓]
          </span>
          <span style={{ color: 'var(--border-structural)' }}>→</span>
          <span style={{ color: canRunInference ? 'var(--color-mars)' : 'var(--text-muted)' }}>
            03 RUN CLASSIFICATION [●]
          </span>
          <span style={{ color: 'var(--border-structural)' }}>→</span>
          <span style={{ color: result ? 'var(--color-green-bright)' : 'var(--text-muted)' }}>
            04 VIEW RESULT [⚑]
          </span>
        </div>
      </div>

      {/* 2. Top Workstation Row: Viewport (Left) | Execution CTA & Result Panel (Right) */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'minmax(0, 1.2fr) minmax(0, 1fr)',
        gap: '14px',
        alignItems: 'stretch',
      }}>
        {/* Left: Viewport Reticle */}
        <div style={{ display: 'flex', flexDirection: 'column' }}>
          <ViewportReticle
            sample={selectedSample}
            isLoading={isLoadingSample}
            isClassifying={isClassifying}
            sampleError={sampleError}
            onOpenSampleDrawer={() => setIsDrawerOpen((o) => !o)}
            onSelectRandomSample={handleSelectRandomSample}
          />
        </div>

        {/* Right: Execution CTA Box + Result Panel */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
          
          {/* Execution Trigger Box */}
          <div style={{
            backgroundColor: 'var(--color-surface)',
            border: '1px solid var(--border-structural)',
            borderRadius: '2px',
            padding: '12px 14px',
            display: 'flex',
            flexDirection: 'column',
            gap: '10px',
          }}>
            {/* Input Sample & Target Readout */}
            <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              fontSize: '10px',
              fontFamily: 'var(--font-mono)',
              borderBottom: '1px solid var(--border-hairline)',
              paddingBottom: '8px',
              flexWrap: 'wrap',
              gap: '6px',
            }}>
              <div>
                <span style={{ color: 'var(--text-muted)' }}>INPUT SAMPLE: </span>
                <span style={{ color: 'var(--color-mars)', fontWeight: 700 }}>
                  {selectedSample?.sample_id || 'NONE'}
                </span>
                <span style={{ color: 'var(--text-muted)', marginLeft: '10px' }}>GROUND TRUTH: </span>
                <span style={{ color: 'var(--color-green-bright)', fontWeight: 700, textTransform: 'uppercase' }}>
                  {selectedSample?.ground_truth_name || '--'}
                </span>
              </div>
              <div>
                <span style={{ color: 'var(--text-muted)' }}>ACTIVE MODEL: </span>
                <span style={{ color: '#FFFFFF', fontWeight: 700 }}>
                  {selectedModel?.name || 'NONE'}
                </span>
              </div>
            </div>

            {/* Warning if cannot run */}
            {!canRunInference && cannotRunReason && (
              <div style={{
                padding: '5px 8px',
                backgroundColor: 'rgba(210, 153, 34, 0.12)',
                color: 'var(--color-amber)',
                fontSize: '10px',
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                border: '1px solid rgba(210, 153, 34, 0.3)',
              }}>
                <AlertTriangle size={12} />
                <span>{cannotRunReason}</span>
              </div>
            )}

            {/* Big Technical Classification Button */}
            <button
              onClick={handleClassify}
              disabled={!selectedSample || !selectedModel || !canRunInference || isClassifying || !!sampleError}
              className="btn btn-primary"
              style={{
                width: '100%',
                padding: '12px',
                fontSize: '13px',
                letterSpacing: '0.08em',
                borderRadius: '2px',
              }}
            >
              {isClassifying ? (
                <>
                  <Sparkles size={16} className="animate-spin" />
                  <span>[ EXECUTING PIPELINE... ]</span>
                </>
              ) : (
                <>
                  <Play size={14} fill="currentColor" />
                  <span>[ RUN CLASSIFICATION ]</span>
                </>
              )}
            </button>

            {/* Status Strip */}
            <div style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              fontSize: '9px',
              color: 'var(--text-muted)',
              paddingTop: '2px',
            }}>
              <span style={{ color: canRunInference ? 'var(--color-green-bright)' : 'var(--color-amber)' }}>
                {canRunInference ? '● PIPELINE ARMED & READY' : '▲ LOCAL CHECKPOINT REQUIRED FOR INFERENCE'}
              </span>
              <span>{selectedModel?.category === 'deep_learning' ? 'PYTORCH DL PIPELINE' : 'SCIKIT-LEARN 8,186-D PIPELINE'}</span>
            </div>
          </div>

          {/* Result Panel */}
          <div style={{ flex: 1, minHeight: '340px' }}>
            <ResultPanel
              result={result}
              isLoading={isClassifying}
              error={classificationError}
            />
          </div>

        </div>

      </div>

      {/* 3. Full-Width Model Selection Section */}
      <div style={{
        backgroundColor: 'var(--color-surface)',
        border: '1px solid var(--border-structural)',
        borderRadius: '2px',
        overflow: 'hidden',
        width: '100%',
      }}>
        {/* Header */}
        <div style={{
          padding: '8px 14px',
          backgroundColor: 'var(--color-charcoal)',
          borderBottom: '1px solid var(--border-hairline)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          fontSize: '11px',
          flexWrap: 'wrap',
          gap: '8px',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{ color: 'var(--color-mars)', fontWeight: 800 }}>■</span>
            <span style={{ fontWeight: 800, letterSpacing: '0.04em', color: '#FFFFFF' }}>
              02. SELECT CLASSIFICATION MODEL
            </span>
            <span style={{ color: 'var(--text-muted)', fontSize: '10px' }}>
              TOTAL 9 MODELS (4 DEEP LEARNING • 5 CLASSICAL ML)
            </span>
          </div>

          <span className="badge badge-amber" style={{ fontSize: '9px', padding: '1px 5px' }}>
            EVALUATED ON 1,305 CANONICAL TEST IMAGES
          </span>
        </div>

        {/* Model List Grids */}
        <div style={{ padding: '14px', display: 'flex', flexDirection: 'column', gap: '14px' }}>
          {isLoadingModels ? (
            <div style={{ padding: '30px', textAlign: 'center', color: 'var(--text-muted)', fontSize: '11px' }}>
              LOADING MODEL INVENTORY...
            </div>
          ) : (
            <>
              {/* Deep Learning Architectures */}
              <div>
                <div style={{
                  fontSize: '10px',
                  color: 'var(--color-mars)',
                  fontWeight: 700,
                  letterSpacing: '0.06em',
                  marginBottom: '8px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                }}>
                  <span>// DEEP LEARNING ARCHITECTURES (PYTORCH)</span>
                  <span style={{ fontSize: '9px', color: 'var(--text-muted)' }}>4 MODELS</span>
                </div>
                <div style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))',
                  gap: '8px',
                }}>
                  {dlModels.map((model) => (
                    <ModelCard
                      key={model.id}
                      model={model}
                      isSelected={selectedModel?.id === model.id}
                      onSelect={handleSelectModel}
                      canRunInference={isTestSplit || model.has_checkpoint}
                      cannotRunReason={!isTestSplit && !model.has_checkpoint ? 'Local checkpoint required for train/val' : undefined}
                    />
                  ))}
                </div>
              </div>

              {/* Classical Machine Learning */}
              <div>
                <div style={{
                  fontSize: '10px',
                  color: 'var(--color-amber)',
                  fontWeight: 700,
                  letterSpacing: '0.06em',
                  marginBottom: '8px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                }}>
                  <span>// CLASSICAL MACHINE LEARNING (SCIKIT-LEARN 8,186-DIM FEATURE VECTOR)</span>
                  <span style={{ fontSize: '9px', color: 'var(--text-muted)' }}>5 MODELS</span>
                </div>
                <div style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(auto-fit, minmax(210px, 1fr))',
                  gap: '8px',
                }}>
                  {clModels.map((model) => (
                    <ModelCard
                      key={model.id}
                      model={model}
                      isSelected={selectedModel?.id === model.id}
                      onSelect={handleSelectModel}
                      canRunInference={isTestSplit || model.has_checkpoint}
                      cannotRunReason={!isTestSplit && !model.has_checkpoint ? 'Local checkpoint required for train/val' : undefined}
                    />
                  ))}
                </div>
              </div>
            </>
          )}
        </div>
      </div>

      {/* 3. Collapsible / Embedded Sample Drawer Strip */}
      <div style={{
        marginTop: '6px',
        backgroundColor: 'var(--color-surface)',
        border: '1px solid var(--border-structural)',
        borderRadius: '2px',
        overflow: 'hidden',
      }}>
        <div
          onClick={() => setIsDrawerOpen((o) => !o)}
          style={{
            padding: '8px 14px',
            backgroundColor: 'var(--color-charcoal)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            cursor: 'pointer',
            fontSize: '11px',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{ color: 'var(--color-mars)', fontWeight: 800 }}>■</span>
            <span style={{ fontWeight: 800, color: '#FFFFFF', letterSpacing: '0.04em' }}>
              04. ARCHIVE SAMPLE SELECTOR
            </span>
            <span style={{ color: 'var(--text-muted)', fontSize: '10px' }}>
              (1,305 TEST IMAGES • 24 TAXA)
            </span>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: 'var(--text-muted)' }}>
            <span style={{ fontSize: '10px' }}>{isDrawerOpen ? 'COLLAPSE DRAWER' : 'EXPAND ARCHIVE'}</span>
            {isDrawerOpen ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
          </div>
        </div>

        {isDrawerOpen && (
          <div style={{ borderTop: '1px solid var(--border-hairline)' }}>
            <SampleDrawer
              selectedSample={selectedSample}
              onSelectSample={handleSelectSample}
              defaultSplit="test"
            />
          </div>
        )}
      </div>

    </div>
  );
};
