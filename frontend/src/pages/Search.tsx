import { useEffect, useState } from 'react';
import { api } from '../api';
import type { Camera, Track } from '../types';
import { ErrorNotice } from '../components/Notice';
import TrackCard from '../components/TrackCard';

export default function Search({ onTrace }: { onTrace: (trackId: string) => void }) {
  const [query, setQuery] = useState('red car');
  const [camera, setCamera] = useState('');
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [results, setResults] = useState<Track[] | null>(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);

  useEffect(() => { api.cameras().then(setCameras).catch(() => undefined); }, []);

  async function run(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError('');
    try {
      setResults((await api.search(query, camera)).results);
    } catch (err) {
      setResults(null);
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-5">
      <form onSubmit={run} className="grid gap-3 rounded-xl border border-slate-200 bg-white p-4 md:grid-cols-[1fr_180px_auto]">
        <div>
          <label htmlFor="q" className="block text-sm font-semibold">Describe the vehicle</label>
          <input id="q" value={query} onChange={(e) => setQuery(e.target.value)} placeholder="white truck, red car on c021"
                 className="mt-1 w-full rounded-lg border border-slate-300 px-3 py-2" />
        </div>
        <div>
          <label htmlFor="cam" className="block text-sm font-semibold">Camera</label>
          <select id="cam" value={camera} onChange={(e) => setCamera(e.target.value)} className="mt-1 w-full rounded-lg border border-slate-300 bg-white px-3 py-2">
            <option value="">All cameras</option>
            {cameras.map((c) => <option key={c.camera_id} value={c.camera_id}>{c.camera_id}</option>)}
          </select>
        </div>
        <button disabled={busy} className="self-end rounded-lg bg-route px-4 py-2 font-semibold text-white disabled:opacity-60">
          {busy ? 'Searching…' : 'Search'}
        </button>
      </form>
      <p className="text-sm text-slate-500">Understands colour, vehicle type (car, truck, bus, bike), direction and camera ids such as c021.</p>

      {error && <ErrorNotice message={error} />}
      {results && results.length === 0 && <p className="text-slate-600">No vehicles match that description.</p>}
      {results && results.length > 0 && (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {results.map((t) => (
            <TrackCard key={t.track_id} track={t}>
              <p className="text-slate-500">Match {Math.round((t.match ?? 0) * 100)}%</p>
              <button onClick={() => onTrace(t.track_id)} className="mt-2 w-full rounded-lg border border-route px-3 py-1.5 font-semibold text-route hover:bg-blue-50">
                Trace across cameras
              </button>
            </TrackCard>
          ))}
        </div>
      )}
    </div>
  );
}
