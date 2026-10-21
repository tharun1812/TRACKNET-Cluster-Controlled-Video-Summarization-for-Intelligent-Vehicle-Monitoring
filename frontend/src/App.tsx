import { useState } from 'react';
import Dashboard from './pages/Dashboard';
import Search from './pages/Search';
import Summaries from './pages/Summaries';
import Tracker from './pages/Tracker';

type View = 'dashboard' | 'search' | 'tracker' | 'summaries';
const TABS: { id: View; label: string }[] = [
  { id: 'dashboard', label: 'Overview' },
  { id: 'search', label: 'Find a vehicle' },
  { id: 'tracker', label: 'Trace path' },
  { id: 'summaries', label: 'Cluster summaries' },
];

export default function App() {
  const [view, setView] = useState<View>('dashboard');
  const [traceId, setTraceId] = useState<string | null>(null);

  return (
    <div className="mx-auto max-w-7xl px-4 py-6 sm:px-6">
      <header className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">TrackNet</h1>
          <p className="text-sm text-slate-600">Cluster-controlled vehicle tracing and video summarization</p>
        </div>
        <nav aria-label="Sections" className="flex flex-wrap gap-1 rounded-xl bg-white p-1 border border-slate-200">
          {TABS.map((t) => (
            <button key={t.id} onClick={() => setView(t.id)} aria-current={view === t.id ? 'page' : undefined}
                    className={`rounded-lg px-3 py-1.5 text-sm font-semibold ${view === t.id ? 'bg-ink text-white' : 'text-slate-600 hover:bg-slate-100'}`}>
              {t.label}
            </button>
          ))}
        </nav>
      </header>
      <main className="mt-6">
        {view === 'dashboard' && <Dashboard />}
        {view === 'search' && <Search onTrace={(id) => { setTraceId(id); setView('tracker'); }} />}
        {view === 'tracker' && <Tracker startTrackId={traceId} />}
        {view === 'summaries' && <Summaries />}
      </main>
    </div>
  );
}
