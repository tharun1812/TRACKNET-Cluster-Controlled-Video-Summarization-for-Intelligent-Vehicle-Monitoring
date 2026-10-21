import type { Analytics, Camera, Candidate, Cluster, Session, SummaryResult, TrafficEvent, Track } from './types';

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`/api${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error((body as { error?: string }).error ?? `Request failed (${res.status})`);
  return body as T;
}

const post = <T,>(path: string, data?: unknown) =>
  request<T>(path, { method: 'POST', body: JSON.stringify(data ?? {}) });

export const api = {
  cameras: () => request<Camera[]>('/cameras'),
  clusters: () => request<Cluster[]>('/clusters'),
  analytics: () => request<Analytics>('/analytics'),
  events: () => request<TrafficEvent[]>('/events'),
  tracks: (camera?: string) => request<Track[]>(`/tracks?limit=200${camera ? `&camera=${camera}` : ''}`),
  search: (query: string, camera?: string) =>
    post<{ parsed: Record<string, string | null>; results: Track[] }>('/search', { query, camera: camera || undefined }),
  startSession: (trackId: string) => post<Session>('/sessions', { track_id: trackId }),
  candidates: (sid: string, direction: 'forward' | 'backward') =>
    request<{ candidates: Candidate[] }>(`/sessions/${sid}/candidates?direction=${direction}`),
  confirm: (sid: string, direction: string, candidateId: string) =>
    post<{ session: Session }>(`/sessions/${sid}/confirm`, { direction, candidate_id: candidateId }),
  skip: (sid: string, direction: string, candidateId: string) =>
    post<Session>(`/sessions/${sid}/skip`, { direction, candidate_id: candidateId }),
  auto: (sid: string) => post<{ added: string[]; session: Session }>(`/sessions/${sid}/auto`),
  vehicleSummary: (sid: string) => post<SummaryResult>(`/sessions/${sid}/summary`),
  clusterSummary: (cluster: string) => post<SummaryResult>(`/clusters/${cluster}/summary`),
  label: (chain: string[], labelType: 'correct' | 'incorrect' | 'uncertain') =>
    post<{ label_id: number }>('/labels', { track_chain: chain, label_type: labelType, annotated_by: 'dashboard' }),
};

export const imageUrl = (trackId: string) => `/api/tracks/${trackId}/image`;
