"use client";

import Link from "next/link";
import { EventRecord } from "@/lib/api";
import { SeverityBadge } from "@/components/ui/Badge";
import { Card, CardTitle } from "@/components/ui/Card";

interface Props {
  events: EventRecord[];
  highlightId?: number;
}

function fmtDate(ts: string) {
  const d = new Date(ts);
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
}

function fmtTime(ts: string) {
  return new Date(ts).toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", hour12: false });
}

export function EventTimeline({ events, highlightId }: Props) {
  if (!events.length) {
    return (
      <Card>
        <CardTitle>Anomaly Events</CardTitle>
        <div className="text-slate-500 text-sm py-4">No events in this window.</div>
      </Card>
    );
  }

  return (
    <Card>
      <CardTitle>Anomaly Events ({events.length})</CardTitle>
      <div className="space-y-2">
        {events.map((evt) => (
          <Link
            key={evt.event_id}
            href={`/events/${evt.event_id}`}
            className={`flex items-center gap-3 p-3 rounded border transition-colors group
              ${highlightId === evt.event_id
                ? "border-cyan-600 bg-cyan-900/20"
                : "border-slate-700 hover:border-slate-500 hover:bg-slate-700/40"
              }`}
          >
            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-2 mb-0.5">
                <SeverityBadge severity={evt.severity} />
                <span className="text-slate-300 text-sm font-medium truncate">
                  {fmtDate(evt.start_time)}
                </span>
                <span className="text-slate-500 text-xs">
                  {fmtTime(evt.start_time)} – {fmtTime(evt.end_time)}
                </span>
              </div>
              <div className="text-xs text-slate-500">
                {evt.duration_minutes} min &bull; {evt.total_lost_kwh.toFixed(1)} kWh lost
                {evt.site_id && <> &bull; {evt.site_id}</>}
              </div>
            </div>
            <div className="text-slate-600 group-hover:text-slate-400 text-xs">→</div>
          </Link>
        ))}
      </div>
    </Card>
  );
}
