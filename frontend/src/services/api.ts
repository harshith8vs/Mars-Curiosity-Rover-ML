import type {
  ClassifyResponse,
  DatasetStats,
  HealthResponse,
  ModelInfo,
  ModelPerformanceResponse,
  SampleItem,
  SampleListResponse,
} from '../types';

const RAW_BASE = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';
const SERVER_URL = RAW_BASE.replace(/\/+$/, '');
const API_BASE = SERVER_URL.endsWith('/api') ? SERVER_URL : `${SERVER_URL}/api`;

async function fetchJson<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(url, options);
  if (!res.ok) {
    let errorDetail = res.statusText;
    try {
      const errBody = await res.json();
      if (errBody && errBody.detail) {
        errorDetail = errBody.detail;
      }
    } catch {
      // ignore
    }
    throw new Error(errorDetail || `API request failed with HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  async getHealth(): Promise<HealthResponse> {
    return fetchJson<HealthResponse>(`${API_BASE}/health`);
  },

  async getModels(): Promise<ModelInfo[]> {
    const [models, perf] = await Promise.all([
      fetchJson<ModelInfo[]>(`${API_BASE}/models`),
      this.getPerformance().catch(() => null),
    ]);

    if (perf && perf.models) {
      const benchMap = new Map(perf.models.map((b) => [b.key, b]));
      return models.map((m) => ({
        ...m,
        benchmark: benchMap.get(m.id),
      }));
    }
    return models;
  },

  async getPerformance(): Promise<ModelPerformanceResponse> {
    return fetchJson<ModelPerformanceResponse>(`${API_BASE}/models/performance`);
  },

  async getDatasetStats(): Promise<DatasetStats> {
    return fetchJson<DatasetStats>(`${API_BASE}/dataset/stats`);
  },

  async getSamples(params?: {
    split?: 'train' | 'val' | 'test';
    page?: number;
    pageSize?: number;
    search?: string;
    classId?: number;
    classFilter?: string;
  }): Promise<SampleListResponse> {
    const q = new URLSearchParams();
    if (params?.split) q.set('split', params.split);
    if (params?.page) q.set('page', String(params.page));
    if (params?.pageSize) q.set('page_size', String(params.pageSize));
    const searchQuery = params?.search || params?.classFilter;
    if (searchQuery) q.set('q', searchQuery);
    if (params?.classId !== undefined) q.set('class_id', String(params.classId));
    return fetchJson<SampleListResponse>(`${API_BASE}/samples?${q.toString()}`);
  },

  async getSample(sampleId: string): Promise<SampleItem> {
    return fetchJson<SampleItem>(`${API_BASE}/samples/${sampleId}`);
  },

  async classify(sampleId: string, modelKey: string, preferLive: boolean = false): Promise<ClassifyResponse> {
    return fetchJson<ClassifyResponse>(`${API_BASE}/classify`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        sample_id: sampleId,
        model: modelKey,
        prefer_live: preferLive,
      }),
    });
  },

  getImageUrl(sampleIdOrUrl: string): string {
    if (sampleIdOrUrl.startsWith('http://') || sampleIdOrUrl.startsWith('https://')) {
      return sampleIdOrUrl;
    }
    if (sampleIdOrUrl.startsWith('/api/')) {
      return `${SERVER_URL}${sampleIdOrUrl}`;
    }
    if (sampleIdOrUrl.startsWith('api/')) {
      return `${SERVER_URL}/${sampleIdOrUrl}`;
    }
    return `${API_BASE}/samples/${sampleIdOrUrl}/image`;
  },
};
