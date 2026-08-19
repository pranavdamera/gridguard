import type { ReactNode } from "react";
import type { DataMode, Health, Severity } from "@/lib/api";

export function Card({
  children,
  className = "",
}: {
  children: ReactNode;
  className?: string;
}) {
  return (
    <div
      className={`rounded-lg border border-slate-800 bg-slate-900/50 p-4 ${className}`}
    >
      {children}
    </div>
  );
}

export function CardTitle({
  children,
  hint,
}: {
  children: ReactNode;
  hint?: string;
}) {
  return (
    <div className="mb-3 flex items-baseline justify-between gap-3">
      <h2 className="text-sm font-semibold tracking-tight text-slate-200">
        {children}
      </h2>
      {hint && <span className="text-[11px] text-slate-500">{hint}</span>}
    </div>
  );
}

export function Stat({
  label,
  value,
  unit,
  tone = "neutral",
  hint,
}: {
  label: string;
  value: string | number;
  unit?: string;
  tone?: "neutral" | "good" | "warn" | "bad";
  hint?: string;
}) {
  const toneClass = {
    neutral: "text-slate-100",
    good: "text-emerald-400",
    warn: "text-amber-400",
    bad: "text-red-400",
  }[tone];

  return (
    <div className="rounded-lg border border-slate-800 bg-slate-900/50 px-3 py-2.5">
      <div className="text-[11px] uppercase tracking-wide text-slate-500">
        {label}
      </div>
      <div className={`mt-0.5 text-xl font-semibold tabular-nums ${toneClass}`}>
        {value}
        {unit && <span className="ml-1 text-xs font-normal text-slate-500">{unit}</span>}
      </div>
      {hint && <div className="mt-0.5 text-[11px] text-slate-600">{hint}</div>}
    </div>
  );
}

const SEVERITY_STYLE: Record<Severity, string> = {
  low: "bg-slate-800 text-slate-400 border-slate-700",
  medium: "bg-amber-950/60 text-amber-400 border-amber-900",
  high: "bg-red-950/60 text-red-400 border-red-900",
};

export function SeverityBadge({ severity }: { severity: Severity }) {
  return (
    <span
      className={`rounded border px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide ${SEVERITY_STYLE[severity]}`}
    >
      {severity}
    </span>
  );
}

const HEALTH_STYLE: Record<Health, string> = {
  healthy: "bg-emerald-950/60 text-emerald-400 border-emerald-900",
  warning: "bg-amber-950/60 text-amber-400 border-amber-900",
  critical: "bg-red-950/60 text-red-400 border-red-900",
  no_data: "bg-slate-800 text-slate-500 border-slate-700",
};

export function HealthBadge({ health }: { health: Health }) {
  const label = health === "no_data" ? "no data" : health;
  return (
    <span
      className={`rounded border px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide ${HEALTH_STYLE[health]}`}
    >
      {label}
    </span>
  );
}

/**
 * The provenance badge.
 *
 * Every surface that shows generation numbers carries one of these. It is the
 * single most important piece of UI in the application: without it a viewer
 * cannot tell a measured array from a simulation.
 */
export function DataModeBadge({
  mode,
  className = "",
}: {
  mode: DataMode;
  className?: string;
}) {
  const isReal = mode === "real";
  return (
    <span
      className={`inline-flex items-center gap-1 rounded border px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide ${
        isReal
          ? "border-emerald-900 bg-emerald-950/60 text-emerald-400"
          : "border-amber-900 bg-amber-950/60 text-amber-400"
      } ${className}`}
    >
      <span
        className={`h-1.5 w-1.5 rounded-full ${isReal ? "bg-emerald-400" : "bg-amber-400"}`}
      />
      {isReal ? "Measured data" : "Simulated data"}
    </span>
  );
}

/** Full-width provenance statement, shown above any chart of generation. */
export function ProvenanceNotice({
  mode,
  text,
  className = "",
}: {
  mode: DataMode;
  text: string;
  className?: string;
}) {
  const isReal = mode === "real";
  return (
    <div
      className={`rounded-lg border px-3 py-2 text-xs leading-relaxed ${
        isReal
          ? "border-emerald-900/60 bg-emerald-950/30 text-emerald-200/80"
          : "border-amber-900/60 bg-amber-950/30 text-amber-200/80"
      } ${className}`}
    >
      <span className="font-semibold uppercase tracking-wide">
        {isReal ? "Measured" : "Simulated"}
      </span>
      <span className="mx-2 text-slate-600">|</span>
      {text}
    </div>
  );
}

export function ErrorPanel({ error, hint }: { error: string; hint?: string }) {
  return (
    <div className="rounded-lg border border-red-900/60 bg-red-950/30 p-4 text-sm text-red-200/90">
      <p className="font-medium">Could not load data</p>
      <p className="mt-1 font-mono text-xs text-red-300/70">{error}</p>
      <p className="mt-2 text-xs text-red-200/70">
        {hint ??
          "Start the backend with `make api`, and build artifacts once with `make build-artifacts`."}
      </p>
    </div>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return (
    <div className="rounded-lg border border-dashed border-slate-800 p-6 text-center text-sm text-slate-500">
      {children}
    </div>
  );
}

export function formatKwh(value: number): string {
  if (Math.abs(value) >= 1000) return `${(value / 1000).toFixed(1)} MWh`;
  return `${value.toFixed(1)} kWh`;
}

export function formatDuration(minutes: number): string {
  if (minutes < 60) return `${Math.round(minutes)} min`;
  const hours = Math.floor(minutes / 60);
  const rest = Math.round(minutes % 60);
  return rest ? `${hours}h ${rest}min` : `${hours}h`;
}

export function formatTimestamp(value: string): string {
  const date = new Date(value.replace(" ", "T"));
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "numeric",
    minute: "2-digit",
    hour12: false,
  });
}
