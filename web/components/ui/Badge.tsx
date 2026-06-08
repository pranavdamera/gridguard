interface BadgeProps {
  severity: "low" | "medium" | "high" | string;
  className?: string;
}

const colors: Record<string, string> = {
  low: "bg-emerald-900/60 text-emerald-300 border-emerald-700",
  medium: "bg-amber-900/60 text-amber-300 border-amber-700",
  high: "bg-red-900/60 text-red-300 border-red-700",
};

export function SeverityBadge({ severity, className = "" }: BadgeProps) {
  const cls = colors[severity] ?? "bg-slate-700 text-slate-300 border-slate-600";
  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium border ${cls} ${className}`}>
      {severity}
    </span>
  );
}
