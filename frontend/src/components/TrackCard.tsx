import { useState } from 'react';
import { imageUrl } from '../api';
import type { Track } from '../types';

interface Props {
  track: Track;
  highlight?: boolean;
  children?: React.ReactNode;
}

export default function TrackCard({ track, highlight, children }: Props) {
  const [broken, setBroken] = useState(false);
  return (
    <article className={`overflow-hidden rounded-xl border bg-white ${highlight ? 'border-route ring-2 ring-blue-200' : 'border-slate-200'}`}>
      <div className="flex h-28 items-center justify-center bg-slate-100">
        {broken ? (
          <span className="text-sm text-slate-500">No image</span>
        ) : (
          <img src={imageUrl(track.track_id)} alt={`${track.color} ${track.vehicle_class} on ${track.camera_id}`}
               className="max-h-full max-w-full object-contain" onError={() => setBroken(true)} />
        )}
      </div>
      <div className="space-y-1 p-3 text-sm">
        <p className="font-semibold">{track.track_id}</p>
        <p className="text-slate-600">{track.color} {track.vehicle_class}, moving {track.direction}</p>
        <p className="text-slate-500">{track.camera_id} · {track.start_ts.toFixed(1)}s to {track.end_ts.toFixed(1)}s</p>
        {children}
      </div>
    </article>
  );
}
