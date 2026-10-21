export function ErrorNotice({ message }: { message: string }) {
  return <p role="alert" className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-800">{message}</p>;
}

export function Loading({ label = 'Loading' }: { label?: string }) {
  return <p className="text-sm text-slate-500" role="status">{label}…</p>;
}

export function Bar({ value, tone = 'bg-route' }: { value: number; tone?: string }) {
  return (
    <div className="h-1.5 w-full overflow-hidden rounded-full bg-slate-200">
      <div className={`h-full ${tone}`} style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%` }} />
    </div>
  );
}
