import { useCallback, useEffect, useState } from 'react';
import { api } from '../api';
import type { Candidate, Session, SummaryResult } from '../types';
import { Bar, ErrorNotice, Loading } from '../components/Notice';
import SummaryPlayer from '../components/SummaryPlayer';
import TrackCard from '../components/TrackCard';

type Direction = 'forward' | 'backward';

function CandidatePanel({ title, direction, candidates, onConfirm, onSkip }: {
  title: string; direction: Direction; candidates: Candidate[];
  onConfirm: (d: Direction, id: string) => void; onSkip: (d: Direction, id: string) => void;
}) {
  return (
    <section className="rounded-xl border border-slate-200 bg-white p-4">
      <h3 className="font-bold">{title}</h3>
      {candidates.length === 0 ? <p className="mt-2 text-sm text-slate-500">No more likely matches in this direction.</p> : (
        <div className="mt-3 grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
          {candidates.map((c) => c.track && (
            <TrackCard key={c.candidate_id} track={c.track}>
              <p className="font-semibold">Match {(c.total * 100).toFixed(0)}%</p>
              <dl className="space-y-1 text-xs text-slate-600">
                {([['Look', c.appearance], ['Colour', c.color], ['Timing', c.temporal], ['Direction', c.direction]] as const).map(([k, v]) => (
                  <div key={k}><dt className="flex justify-between"><span>{k}</span><span>{(v * 100).toFixed(0)}%</span></dt><dd><Bar value={v} /></dd></div>
                ))}
              </dl>
              <div className="mt-2 flex gap-2">
                <button onClick={() => onConfirm(direction, c.candidate_id)} className="flex-1 rounded-lg bg-route px-2 py-1.5 font-semibold text-white">Confirm</button>
                <button onClick={() => onSkip(direction, c.candidate_id)} className="flex-1 rounded-lg border border-slate-300 px-2 py-1.5 font-semibold">Skip</button>
              </div>
            </TrackCard>
          ))}
        </div>
      )}
    </section>
  );
}

export default function Tracker({ startTrackId }: { startTrackId: string | null }) {
  const [session, setSession] = useState<Session | null>(null);
  const [forward, setForward] = useState<Candidate[]>([]);
  const [backward, setBackward] = useState<Candidate[]>([]);
  const [summary, setSummary] = useState<SummaryResult | null>(null);
  const [error, setError] = useState('');
  const [note, setNote] = useState('');
  const [busy, setBusy] = useState(false);

  const refresh = useCallback(async (s: Session) => {
    const [f, b] = await Promise.all([api.candidates(s.session_id, 'forward'), api.candidates(s.session_id, 'backward')]);
    setForward(f.candidates);
    setBackward(b.candidates);
  }, []);

  const guard = useCallback(async (fn: () => Promise<void>) => {
    setBusy(true);
    setError('');
    try { await fn(); } catch (e) { setError((e as Error).message); } finally { setBusy(false); }
  }, []);

  useEffect(() => {
    if (!startTrackId) return;
    setSummary(null);
    setNote('');
    void guard(async () => {
      const s = await api.startSession(startTrackId);
      setSession(s);
      await refresh(s);
    });
  }, [startTrackId, guard, refresh]);

  if (!startTrackId) {
    return <p className="rounded-xl border border-dashed border-slate-300 bg-white p-6">Search for a vehicle first, then choose “Trace across cameras”.</p>;
  }
  if (!session) return error ? <ErrorNotice message={error} /> : <Loading label="Starting trace" />;

  const apply = (s: Session) => { setSession(s); setSummary(null); return refresh(s); };
  const confirm = (d: Direction, id: string) => guard(async () => apply((await api.confirm(session.session_id, d, id)).session));
  const skip = (d: Direction, id: string) => guard(async () => apply(await api.skip(session.session_id, d, id)));
  const auto = () => guard(async () => apply((await api.auto(session.session_id)).session));
  const makeSummary = () => guard(async () => setSummary(await api.vehicleSummary(session.session_id)));
  const label = (type: 'correct' | 'incorrect') => guard(async () => {
    await api.label(session.chain.map((t) => t.track_id), type);
    setNote(`Chain saved as “${type}”.`);
  });

  return (
    <div className="space-y-5">
      <section className="rounded-xl border border-slate-200 bg-white p-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h2 className="text-lg font-bold">Vehicle path ({session.chain.length} camera{session.chain.length === 1 ? '' : 's'})</h2>
          <div className="flex flex-wrap gap-2">
            <button disabled={busy} onClick={auto} className="rounded-lg border border-slate-300 px-3 py-1.5 text-sm font-semibold">Auto-trace</button>
            <button disabled={busy || session.chain.length < 2} onClick={() => label('correct')} className="rounded-lg border border-slate-300 px-3 py-1.5 text-sm font-semibold disabled:opacity-50">Mark correct</button>
            <button disabled={busy || session.chain.length < 2} onClick={() => label('incorrect')} className="rounded-lg border border-slate-300 px-3 py-1.5 text-sm font-semibold disabled:opacity-50">Mark incorrect</button>
            <button disabled={busy} onClick={makeSummary} className="rounded-lg bg-signal px-3 py-1.5 text-sm font-semibold text-ink">Generate summary clip</button>
          </div>
        </div>
        <ol className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {session.chain.map((t) => (
            <li key={t.track_id}><TrackCard track={t} highlight={t.track_id === session.query_track_id} /></li>
          ))}
        </ol>
        {note && <p role="status" className="mt-3 text-sm text-green-700">{note}</p>}
      </section>

      {error && <ErrorNotice message={error} />}
      {summary && <section className="rounded-xl border border-slate-200 bg-white p-4"><h2 className="mb-3 text-lg font-bold">Summary clip</h2><SummaryPlayer result={summary} /></section>}

      <div className="grid gap-5 xl:grid-cols-2">
        <CandidatePanel title="Where it came from (earlier)" direction="backward" candidates={backward} onConfirm={confirm} onSkip={skip} />
        <CandidatePanel title="Where it went next (later)" direction="forward" candidates={forward} onConfirm={confirm} onSkip={skip} />
      </div>
    </div>
  );
}
