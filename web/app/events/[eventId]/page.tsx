import Link from "next/link";
import { api } from "@/lib/api";
import { EventExplanationCard } from "@/components/EventExplanationCard";
import { RecommendedActions } from "@/components/RecommendedActions";
import { Card, CardTitle } from "@/components/ui/Card";
import { SeverityBadge } from "@/components/ui/Badge";
import { MethodologyNotice } from "@/components/MethodologyNotice";

export const revalidate = 60;

interface Props {
  params: Promise<{ eventId: string }>;
}

function fmtTs(ts: string) {
  return new Date(ts).toLocaleString("en-US", {
    weekday: "short", month: "short", day: "numeric",
    hour: "2-digit", minute: "2-digit", hour12: false,
  });
}

export default async function EventDetailPage({ params }: Props) {
  const { eventId } = await params;
  const id = parseInt(eventId, 10);

  let event = null;
  let explain = null;
  let error = null;

  try {
    event = await api.event(id);
    // Try to get SHAP explanation for the first anomaly in the event window
    if (event) {
      const anoms = await api.anomalies({
        start: event.start_time,
        end: event.end_time,
        only_anomalies: true,
        limit: 1,
      });
      if (anoms.records.length > 0) {
        explain = await api.explain(anoms.records[0].timestamp).catch(() => null);
      }
    }
  } catch (e) {
    error = (e as Error).message;
  }

  if (error || !event) {
    return (
      <div className="max-w-4xl mx-auto px-4 py-8">
        <Link href="/demo" className="text-xs text-slate-500 hover:text-slate-300 mb-6 inline-block">← Back to demo</Link>
        <div className="text-red-400 text-sm border border-red-800 bg-red-950/30 rounded p-4">
          {error ?? "Event not found."}
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto px-4 py-8 space-y-5">
      <div>
        <Link href="/demo" className="text-xs text-slate-500 hover:text-slate-300">← Back to demo</Link>
      </div>

      {/* Header */}
      <div className="flex items-start gap-4 flex-wrap">
        <div className="flex-1">
          <div className="flex items-center gap-3 mb-2">
            <h1 className="text-xl font-semibold text-slate-100">Event #{event.event_id}</h1>
            <SeverityBadge severity={event.severity} />
          </div>
          <p className="text-slate-500 text-sm">
            {event.site_id && <><span className="font-medium text-slate-400">{event.site_id}</span> · </>}
            {fmtTs(event.start_time)} – {fmtTs(event.end_time)}
          </p>
        </div>
      </div>

      <MethodologyNotice />

      {/* Summary answers */}
      <Card>
        <CardTitle>What happened?</CardTitle>
        <p className="text-slate-300 text-sm leading-relaxed">{event.explanation}</p>
      </Card>

      <div className="grid sm:grid-cols-2 gap-5">
        <EventExplanationCard event={event} explain={explain} />
        <RecommendedActions />
      </div>

      {/* Signal detail */}
      <Card>
        <CardTitle>Signal detail</CardTitle>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-sm">
          <div>
            <div className="text-slate-500 text-xs mb-1">Duration</div>
            <div className="text-slate-200 font-mono">{event.duration_minutes} min</div>
          </div>
          <div>
            <div className="text-slate-500 text-xs mb-1">Intervals flagged</div>
            <div className="text-slate-200 font-mono">{event.interval_count}</div>
          </div>
          <div>
            <div className="text-slate-500 text-xs mb-1">Peak deviation</div>
            <div className="text-red-400 font-mono">{event.max_residual_sigma.toFixed(1)}σ</div>
          </div>
          <div>
            <div className="text-slate-500 text-xs mb-1">Lost energy</div>
            <div className="text-amber-300 font-mono">{event.total_lost_kwh.toFixed(1)} kWh</div>
          </div>
        </div>
      </Card>

      {/* Related */}
      <div className="flex gap-3 flex-wrap text-sm">
        <Link href="/demo" className="text-slate-500 hover:text-slate-300 underline">← Full demo dashboard</Link>
        {event.site_id && (
          <Link href={`/sites/${event.site_id}`} className="text-slate-500 hover:text-slate-300 underline">
            View site: {event.site_id} →
          </Link>
        )}
      </div>
    </div>
  );
}
