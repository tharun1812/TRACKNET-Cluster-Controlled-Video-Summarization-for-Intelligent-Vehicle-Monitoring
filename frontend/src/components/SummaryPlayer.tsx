import type { SummaryResult } from '../types';

export default function SummaryPlayer({ result }: { result: SummaryResult }) {
  return (
    <div className="space-y-2">
      <video key={result.url} src={result.url} controls className="w-full max-w-2xl rounded-lg bg-black" />
      <p className="text-sm text-slate-600">
        {result.segments} segment{result.segments === 1 ? '' : 's'}, {result.summary_seconds}s of footage
        {result.reduction !== null && ` (${(result.reduction * 100).toFixed(0)}% shorter than the ${result.source_seconds}s of source video)`}.
      </p>
    </div>
  );
}
