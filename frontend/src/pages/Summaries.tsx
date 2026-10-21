import { useEffect, useState } from 'react';
import { api } from '../api';
import type { Cluster, SummaryResult } from '../types';
import { ErrorNotice, Loading } from '../components/Notice';
import SummaryPlayer from '../components/SummaryPlayer';

export default function Summaries() {
  const [clusters, setClusters] = useState<Cluster[]>();
  const [results, setResults] = useState<Record<string, SummaryResult>>({});
  const [busy, setBusy] = useState('');
  const [error, setError] = useState('');

  useEffect(() => { api.clusters().then(setClusters).catch((e: Error) => setError(e.message)); }, []);

  async function generate(cluster: string) {
    setBusy(cluster);
    setError('');
    try {
      const r = await api.clusterSummary(cluster);
      setResults((prev) => ({ ...prev, [cluster]: r }));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy('');
    }
  }

  if (!clusters) return error ? <ErrorNotice message={error} /> : <Loading />;
  return (
    <div className="space-y-5">
      <p className="max-w-2xl text-slate-600">A cluster summary keeps only the moments around congestion or stationary vehicles in a group of nearby cameras, and drops the rest of the footage.</p>
      {error && <ErrorNotice message={error} />}
      {clusters.map((c) => (
        <section key={c.cluster} className="rounded-xl border border-slate-200 bg-white p-4">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <h2 className="text-lg font-bold">Cluster {c.cluster} <span className="font-normal text-slate-500">({c.cameras.join(', ')}, {c.tracks} tracks)</span></h2>
            <button disabled={busy === c.cluster} onClick={() => generate(c.cluster)} className="rounded-lg bg-signal px-3 py-1.5 text-sm font-semibold text-ink disabled:opacity-60">
              {busy === c.cluster ? 'Building clip…' : 'Generate summary'}
            </button>
          </div>
          {results[c.cluster] && <div className="mt-4"><SummaryPlayer result={results[c.cluster]} /></div>}
        </section>
      ))}
    </div>
  );
}
