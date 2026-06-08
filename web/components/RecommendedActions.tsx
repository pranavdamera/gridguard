"use client";

import { Card, CardTitle } from "@/components/ui/Card";

const DEFAULT_ACTIONS = [
  "Inspect inverter logs for fault codes in the anomaly window",
  "Check for partial shading (construction, vegetation, equipment)",
  "Compare with adjacent string-level power readings",
  "Verify irradiance sensor vs satellite-derived GHI estimate",
  "Schedule on-site inspection if degradation persists beyond 48 h",
];

interface Props {
  actions?: string[];
}

export function RecommendedActions({ actions = DEFAULT_ACTIONS }: Props) {
  return (
    <Card>
      <CardTitle>Recommended Operator Actions</CardTitle>
      <ol className="space-y-2">
        {actions.map((action, i) => (
          <li key={i} className="flex gap-3 text-sm">
            <span className="text-cyan-600 font-mono text-xs mt-0.5 shrink-0">
              {String(i + 1).padStart(2, "0")}
            </span>
            <span className="text-slate-300">{action}</span>
          </li>
        ))}
      </ol>
    </Card>
  );
}
