"use client";

import { EventsResponse, HealthResponse } from "@/lib/api";
import { Card, CardTitle } from "@/components/ui/Card";

interface Props {
  health: HealthResponse | null;
  events: EventsResponse | null;
  anomalies?: unknown;
}

function Stat({ label, value, unit, accent = false }: { label: string; value: string | number; unit?: string; accent?: boolean }) {
  return (
    <div>
      <div className={`text-2xl font-mono font-semibold ${accent ? "text-red-400" : "text-cyan-300"}`}>
        {value}
        {unit && <span className="text-base ml-1 text-slate-400">{unit}</span>}
      </div>
      <div className="text-xs text-slate-500 mt-0.5">{label}</div>
    </div>
  );
}

export function FleetHealthCards({ health, events }: Props) {
  const highEvents = events?.events.filter(e => e.severity === "high").length ?? 0;
  const totalEvents = events?.total_events ?? 0;
  const totalLost = events?.total_lost_kwh ?? 0;

  return (
    <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
      <Card>
        <CardTitle>Fleet Status</CardTitle>
        <Stat label="API online" value={health?.status === "ok" ? "Online" : "Offline"} />
      </Card>
      <Card>
        <CardTitle>Anomaly Events</CardTitle>
        <Stat label="total detected" value={totalEvents} accent={totalEvents > 0} />
      </Card>
      <Card>
        <CardTitle>High Severity</CardTitle>
        <Stat label="events flagged" value={highEvents} accent={highEvents > 0} />
      </Card>
      <Card>
        <CardTitle>Estimated Lost</CardTitle>
        <Stat label="kWh (test period)" value={totalLost.toFixed(1)} unit="kWh" accent={totalLost > 10} />
      </Card>
    </div>
  );
}
