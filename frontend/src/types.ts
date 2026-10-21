export interface Track {
  track_id: string;
  camera_id: string;
  start_ts: number;
  end_ts: number;
  duration: number;
  vehicle_class: string;
  color: string;
  direction: string;
  match?: number;
}

export interface Candidate {
  candidate_id: string;
  camera_id: string;
  total: number;
  appearance: number;
  color: number;
  temporal: number;
  direction: number;
  graph: number;
  hops: number;
  gap: number;
  track: Track | null;
}

export interface Session {
  session_id: string;
  query_track_id: string;
  chain: Track[];
  visited_cameras: string[];
}

export interface Camera {
  camera_id: string;
  name: string;
  cluster: string;
  view: string;
  tracks: number;
  next: string[];
  previous: string[];
}

export interface Cluster {
  cluster: string;
  cameras: string[];
  tracks: number;
}

export interface Analytics {
  total_tracks: number;
  by_camera: Record<string, number>;
  by_class: Record<string, number>;
  by_color: Record<string, number>;
  by_direction: Record<string, number>;
}

export interface TrafficEvent {
  kind: string;
  camera_id: string;
  start_ts: number;
  end_ts: number;
  detail: Record<string, unknown>;
}

export interface SummaryResult {
  url: string;
  segments: number;
  summary_seconds: number;
  source_seconds: number;
  reduction: number | null;
}
