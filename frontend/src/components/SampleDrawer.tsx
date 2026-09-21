import React, { useState, useEffect } from 'react';
import { Search, ChevronLeft, ChevronRight } from 'lucide-react';
import type { SampleItem, SampleListResponse } from '../types';
import { api } from '../services/api';

interface SampleDrawerProps {
  selectedSample: SampleItem | null;
  onSelectSample: (sample: SampleItem) => void;
  defaultSplit?: 'test' | 'train' | 'val';
}

export const SampleDrawer: React.FC<SampleDrawerProps> = ({
  selectedSample,
  onSelectSample,
  defaultSplit = 'test',
}) => {
  const [split, setSplit] = useState<'train' | 'val' | 'test'>(defaultSplit);
  const [page, setPage] = useState<number>(1);
  const [search, setSearch] = useState<string>('');
  const [searchInput, setSearchInput] = useState<string>('');
  const [classFilter, setClassFilter] = useState<string>('');
  const [data, setData] = useState<SampleListResponse | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [splitCounts, setSplitCounts] = useState<{ test: number; train: number; val: number }>({
    test: 1305,
    train: 3746,
    val: 1640,
  });
  const [activeClasses, setActiveClasses] = useState<{ id: string; name: string }[]>([]);

  useEffect(() => {
    let isMounted = true;
    api.getDatasetStats().then((s) => {
      if (isMounted) {
        setSplitCounts({
          test: s.splits.test,
          train: s.splits.train,
          val: s.splits.val,
        });
        if (s.classes) {
          setActiveClasses(s.classes.map((c) => ({ id: c.class_id, name: c.class_name })));
        }
      }
    }).catch(() => {});
    return () => {
      isMounted = false;
    };
  }, []);

  const pageSize = 12;

  useEffect(() => {
    let isMounted = true;
    const fetchSamples = async () => {
      setIsLoading(true);
      setError(null);
      try {
        const res = await api.getSamples({
          split,
          page,
          pageSize,
          search: search || undefined,
          classFilter: classFilter || undefined,
        });
        if (isMounted) {
          setData(res);
        }
      } catch (err: unknown) {
        if (isMounted) {
          setError(err instanceof Error ? err.message : 'Failed to load samples');
        }
      } finally {
        if (isMounted) setIsLoading(false);
      }
    };
    fetchSamples();
    return () => {
      isMounted = false;
    };
  }, [split, page, search, classFilter]);

  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setPage(1);
    setSearch(searchInput);
  };

  const handleSplitChange = (newSplit: 'train' | 'val' | 'test') => {
    setSplit(newSplit);
    setPage(1);
  };

  return (
    <div style={{
      backgroundColor: 'var(--color-surface)',
      fontFamily: 'var(--font-mono)',
      display: 'flex',
      flexDirection: 'column',
    }}>
      {/* Top Controls: Split Tabs, Search, Filter */}
      <div style={{
        padding: '8px 12px',
        backgroundColor: 'var(--color-charcoal)',
        borderBottom: '1px solid var(--border-hairline)',
        display: 'flex',
        flexWrap: 'wrap',
        alignItems: 'center',
        justifyContent: 'space-between',
        gap: '8px',
      }}>
        {/* Split Switcher */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
          {(['test', 'train', 'val'] as const).map((s) => (
            <button
              key={s}
              onClick={() => handleSplitChange(s)}
              style={{
                padding: '4px 10px',
                fontSize: '10px',
                fontFamily: 'var(--font-mono)',
                fontWeight: 700,
                textTransform: 'uppercase',
                borderRadius: '2px',
                border: split === s ? '1px solid var(--color-mars)' : '1px solid var(--border-hairline)',
                backgroundColor: split === s ? 'var(--color-mars)' : 'var(--color-surface)',
                color: split === s ? '#090B0E' : 'var(--text-secondary)',
                cursor: 'pointer',
              }}
            >
              {s.toUpperCase()} ({splitCounts[s].toLocaleString()})
            </button>
          ))}
        </div>

        {/* Class Filter & Search Bar */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <select
            value={classFilter}
            onChange={(e) => {
              setClassFilter(e.target.value);
              setPage(1);
            }}
            style={{
              backgroundColor: 'var(--color-obsidian)',
              border: '1px solid var(--border-structural)',
              borderRadius: '2px',
              padding: '4px 8px',
              fontSize: '10px',
              fontFamily: 'var(--font-mono)',
              color: 'var(--text-primary)',
              outline: 'none',
              cursor: 'pointer',
              maxWidth: '170px',
            }}
          >
            <option value="">ALL 24 TAXA</option>
            {activeClasses.map((cls) => (
              <option key={cls.id} value={cls.name}>
                {cls.id}: {cls.name.toUpperCase()}
              </option>
            ))}
          </select>

          <form onSubmit={handleSearchSubmit} style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
            <div style={{ position: 'relative', display: 'flex', alignItems: 'center' }}>
              <Search size={12} color="var(--text-muted)" style={{ position: 'absolute', left: '8px' }} />
              <input
                type="text"
                placeholder="SEARCH ID / FILE..."
                value={searchInput}
                onChange={(e) => setSearchInput(e.target.value)}
                style={{
                  backgroundColor: 'var(--color-obsidian)',
                  border: '1px solid var(--border-structural)',
                  borderRadius: '2px',
                  padding: '4px 8px 4px 24px',
                  fontSize: '10px',
                  fontFamily: 'var(--font-mono)',
                  color: 'var(--text-primary)',
                  outline: 'none',
                  width: '150px',
                }}
              />
            </div>
            <button type="submit" className="btn btn-secondary" style={{ padding: '4px 8px', fontSize: '10px' }}>
              FILTER
            </button>
          </form>
        </div>
      </div>

      {/* Grid of Samples */}
      <div style={{
        padding: '10px 12px',
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fill, minmax(130px, 1fr))',
        gap: '8px',
        maxHeight: '260px',
        overflowY: 'auto',
      }}>
        {isLoading ? (
          <div style={{ gridColumn: '1 / -1', display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '30px' }}>
            <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
              INDEXING ARCHIVE SAMPLES...
            </span>
          </div>
        ) : error ? (
          <div style={{ gridColumn: '1 / -1', textAlign: 'center', color: 'var(--color-error)', padding: '20px', fontSize: '11px' }}>
            {error}
          </div>
        ) : data?.samples.length === 0 ? (
          <div style={{ gridColumn: '1 / -1', textAlign: 'center', color: 'var(--text-muted)', padding: '20px', fontSize: '11px' }}>
            NO SAMPLES MATCH CRITERIA
          </div>
        ) : (
          data?.samples.map((sample) => {
            const isSelected = selectedSample?.sample_id === sample.sample_id;
            const thumbUrl = api.getImageUrl(sample.image_url);
            return (
              <div
                key={sample.sample_id}
                onClick={() => onSelectSample(sample)}
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  borderRadius: '2px',
                  border: isSelected ? '1px solid var(--color-mars)' : '1px solid var(--border-structural)',
                  backgroundColor: isSelected ? 'var(--color-charcoal)' : 'var(--color-surface)',
                  overflow: 'hidden',
                  cursor: 'pointer',
                  transition: 'border-color 0.12s ease',
                }}
              >
                {/* Thumbnail */}
                <div style={{ height: '70px', backgroundColor: '#000000', overflow: 'hidden' }}>
                  <img
                    src={thumbUrl}
                    alt={sample.filename}
                    loading="lazy"
                    style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                  />
                </div>

                {/* Details */}
                <div style={{ padding: '5px 6px', display: 'flex', flexDirection: 'column', gap: '2px' }}>
                  <div style={{ fontSize: '9px', color: isSelected ? 'var(--color-mars)' : 'var(--text-primary)', fontWeight: 700 }}>
                    {sample.sample_id}
                  </div>
                  <div style={{
                    fontSize: '9px',
                    color: 'var(--color-amber)',
                    whiteSpace: 'nowrap',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    textTransform: 'uppercase',
                  }}>
                    {sample.ground_truth_name}
                  </div>
                </div>
              </div>
            );
          })
        )}
      </div>

      {/* Pagination Footer */}
      {data && (
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '6px 12px',
          borderTop: '1px solid var(--border-hairline)',
          backgroundColor: 'var(--color-charcoal)',
          fontSize: '10px',
          color: 'var(--text-muted)',
        }}>
          <div>
            PAGE <span style={{ color: '#FFFFFF', fontWeight: 700 }}>{data.page}</span> / {data.total_pages} ({data.total.toLocaleString()} SAMPLES)
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <button
              disabled={page <= 1}
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              className="btn btn-secondary"
              style={{ padding: '2px 6px', fontSize: '9px' }}
            >
              <ChevronLeft size={11} /> PREV
            </button>
            <button
              disabled={page >= data.total_pages}
              onClick={() => setPage((p) => Math.min(data.total_pages, p + 1))}
              className="btn btn-secondary"
              style={{ padding: '2px 6px', fontSize: '9px' }}
            >
              NEXT <ChevronRight size={11} />
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
