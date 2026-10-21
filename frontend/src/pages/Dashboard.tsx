import { useEffect, useState } from 'react';
import { api } from '../api';
import type { Analytics, Camera, Cluster, TrafficEvent } from '../types';
import { Bar, ErrorNotice, Loading } from '../components/Notice';

function Breakdown({ title, data }: { title: string; data: Record<string, number> }) {
  const max = Math.max(1, ...Object.values(data));
  return (
    <section className="rounded-xl border border-slate-200 bg-white p-4">
      <h3 className="font-semibold">{title}</h3>
      <ul className="mt-3 space-y-2">
        {Object.entries(data).sort((a, b) => b[1] - a[1]).map(([k, v]) => (
          <li key={k}>
            <div className="flex justify-between text-sm"><span>{k}</span><span>{v}</span></div>
            <Bar value={v / max} />
          </li>
        ))}
      </ul>
    </section>
  );
}

export default function Dashboard() {
  const [cameras, setCameras] = useState<Camera[]>();
  const [clusters, setClusters] = useState<Cluster[]>();
  const [analytics, setAnalytics] = useState<Analytics>();
  const [events, setEvents] = useState<TrafficEvent[]>();
  const [error, setError] = useState('');

  useEffect(() => {
    Promise.all([api.cameras(), api.clusters(), api.analytics(), api.events()])
      .then(([c, cl, a, e]) => { setCameras(c); setClusters(cl); setAnalytics(a); setEvents(e); })
      .catch((e: Error) => setError(e.message));
  }, []);

  if (error) return <ErrorNotice message={`${error}. Is the API running? Start it with: python -m tracknet serve`} />;
  if (!cameras || !analytics || !clusters || !events) return <Loading label="Loading dashboard" />;
  if (analytics.total_tracks === 0) {
    return <p className="rounded-xl border border-dashed border-slate-300 bg-white p-6">No tracks yet. Run <code>python -m tracknet demo</code> or process a video, then reload.</p>;
  }

  return (
    <div className="space-y-6">
      <div className="grid gap-4 sm:grid-cols-3">
        <div className="rounded-xl bg-ink p-4 text-white"><p className="text-sm text-slate-300">Vehicle tracks</p><p className="text-3xl font-bold">{analytics.total_tracks}</p></div>
        <div className="rounded-xl bg-white p-4 border border-slate-200"><p className="text-sm text-slate-500">Cameras</p><p className="text-3xl font-bold">{cameras.length}</p></div>
        <div className="rounded-xl bg-white p-4 border border-slate-200"><p className="text-sm text-slate-500">Clusters</p><p className="text-3xl font-bold">{clusters.length}</p></div>
      </div>

      <section className="rounded-xl border border-slate-200 bg-white">
        <h2 className="px-4 pt-4 text-lg font-bold">Cameras</h2>
        <div className="overflow-x-auto p-4">
          <table className="w-full min-w-[560px] text-left text-sm">
            <thead className="text-slate-500"><tr><th className="pb-2">Camera</th><th>Cluster</th><th>Location</th><th>Leads to</th><th className="text-right">Tracks</th></tr></thead>
            <tbody>
              {cameras.map((c) => (
                <tr key={c.camera_id} className="border-t border-slate-100">
                  <td className="py-2 font-semibold">{c.camera_id}</td><td>{c.cluster}</td><td>{c.name}</td>
                  <td>{c.next.join(', ') || 'end of route'}</td><td className="text-right">{c.tracks}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <div className="grid gap-4 md:grid-cols-3">
        <Breakdown title="By vehicle type" data={analytics.by_class} />
        <Breakdown title="By colour" data={analytics.by_color} />
        <Breakdown title="By camera" data={analytics.by_camera} />
      </div>

      <section className="rounded-xl border border-slate-200 bg-white p-4">
        <h2 className="text-lg font-bold">Detected events</h2>
        {events.length === 0 ? <p className="mt-2 text-sm text-slate-500">No congestion or stationary vehicles found.</p> : (
          <ul className="mt-3 divide-y divide-slate-100 text-sm">
            {events.map((e, i) => (
              <li key={i} className="flex flex-wrap justify-between gap-2 py-2">
                <span><strong>{e.kind.replace('_', ' ')}</strong> on {e.camera_id}</span>
                <span className="text-slate-500">{e.start_ts.toFixed(0)}s to {e.end_ts.toFixed(0)}s{e.kind === 'congestion' ? `, up to ${String(e.detail.peak_vehicles)} vehicles` : ''}</span>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}
