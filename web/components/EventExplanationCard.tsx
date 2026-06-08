"use client";

import { EventRecord, ExplainResponse } from "@/lib/api";
import { Card, CardTitle } from "@/components/ui/Card";
import { SeverityBadge } from "@/components/ui/Badge";

interface Props {
  event: EventRecord;
  explain?: ExplainResponse | null;
}

function fmtTs(ts: string) {
  return new Date(ts).toLocaleString("en-US", {
    month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", hour12: false,
  });
}

export function EventExplanationCard({ event, explain }: Props) {
  const pctDrop = event.mean_predicted_kw > 0
    ? Math.round(100 * (1 - event.mean_actual_kw / event.mean_predicted_kw))
    : 0;

  return (
    <Card>
      <CardTitle>Why Flagged?</CardTitle>
      <div className="space-y-4">
        <div className="flex items-center gap-2">
          <SeverityBadge severity={event.severity} />
          <span className="text-slate-300 text-sm">Event #{event.event_id}</span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 text-sm">
          <div>
            <div className="text-slate-500 text-xs mb-0.5">Window</div>
            <div className="text-slate-200 font-mono text-xs">
              {fmtTs(event.start_time)} – {fmtTs(event.end_time)}
            </div>
          </div>
          <div>
            <div className="text-slate-500 text-xs mb-0.5">Output drop</div>
            <div className="text-red-400 font-semibold">{pctDrop}% below forecast</div>
          </div>
          <div>
            <div className="text-slate-500 text-xs mb-0.5">Lost energy</div>
            <div className="text-amber-300 font-semibold">{event.total_lost_kwh.toFixed(1)} kWh</div>
          </div>
          <div>
            <div className="text-slate-500 text-xs mb-0.5">Expected avg</div>
            <div className="text-cyan-300">{event.mean_predicted_kw.toFixed(1)} kW</div>
          </div>
          <div>
            <div className="text-slate-500 text-xs mb-0.5">Actual avg</div>
            <div className="text-slate-300">{event.mean_actual_kw.toFixed(1)} kW</div>
          </div>
          <div>
            <div className="text-slate-500 text-xs mb-0.5">Peak sigma</div>
            <div className="text-slate-300">{event.max_residual_sigma.toFixed(1)}σ</div>
          </div>
        </div>

        <div className="text-slate-400 text-sm border-l-2 border-slate-600 pl-3">
          {event.explanation}
        </div>

        {explain && explain.contributors.length > 0 && (
          <div>
            <div className="text-xs text-slate-500 uppercase tracking-wider mb-2">
              Feature contributions {explain.shap_available ? "(SHAP)" : "(importance proxy)"}
            </div>
            <div className="space-y-1.5">
              {explain.contributors.slice(0, 6).map((c) => {
                const bar = Math.min(100, Math.abs(c.shap_value) * 400);
                const color = c.direction === "positive" ? "bg-cyan-600" : "bg-red-700";
                return (
                  <div key={c.feature} className="flex items-center gap-2 text-xs">
                    <div className="w-28 text-slate-400 truncate">{c.feature}</div>
                    <div className="flex-1 bg-slate-700 rounded-full h-1.5">
                      <div className={`${color} h-1.5 rounded-full`} style={{ width: `${bar}%` }} />
                    </div>
                    <div className="w-16 text-right font-mono text-slate-400">
                      {c.shap_value > 0 ? "+" : ""}{c.shap_value.toFixed(3)}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>
    </Card>
  );
}
