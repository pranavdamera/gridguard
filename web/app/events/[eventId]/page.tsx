import Link from "next/link";
import { api, tryFetch } from "@/lib/api";
import { ChartLegend, GenerationChart } from "@/components/GenerationChart";
import {
  Card,
  CardTitle,
  DataModeBadge,
  Empty,
  ErrorPanel,
  ProvenanceNotice,
  SeverityBadge,
  Stat,
  formatDuration,
  formatKwh,
  formatTimestamp,
} from "@/components/ui/Primitives";

export const revalidate = 60;

const SCOPE_STYLE: Record<string, string> = {
  site_specific: "border-red-900 bg-red-950/30 text-red-300",
  regional: "border-sky-900 bg-sky-950/30 text-sky-300",
  indeterminate: "border-slate-700 bg-slate-900/60 text-slate-400",
};

const SCOPE_HEADLINE: Record<string, string> = {
  site_specific: "Likely site-specific",
  regional: "Likely regional weather",
  indeterminate: "Inconclusive",
};

export default async function EventDetailPage(props: {
  params: Promise<{ eventId: string }>;
}) {
  const { eventId } = await props.params;
  const { data, error } = await tryFetch(() => api.event(eventId));

  if (error || !data) {
    return (
      <div className="mx-auto max-w-5xl px-4 py-8">
        <Link href="/demo" className="text-xs text-slate-500 hover:text-slate-300">
          ← Fleet
        </Link>
        <div className="mt-3">
          <ErrorPanel error={error ?? "Event not found"} />
        </div>
      </div>
    );
  }

  const { event, spatial_context, neighbor_comparison, timeseries, recommended_actions, contributors } =
    data;
  const shortfall =
    event.mean_predicted_kw > 0
      ? 1 - event.mean_actual_kw / event.mean_predicted_kw
      : 0;

  return (
    <div className="mx-auto max-w-5xl px-4 py-8">
      <Link
        href={event.site_id ? `/sites/${event.site_id}` : "/demo"}
        className="text-xs text-slate-500 hover:text-slate-300"
      >
        ← {event.site_name ?? "Fleet"}
      </Link>

      <header className="mt-2 mb-4">
        <div className="flex flex-wrap items-center gap-2">
          <h1 className="text-xl font-semibold tracking-tight text-slate-100">
            Event investigation
          </h1>
          <SeverityBadge severity={event.severity} />
          {event.data_mode && <DataModeBadge mode={event.data_mode} />}
        </div>
        <p className="mt-1 text-sm text-slate-400">
          {event.site_name ?? event.site_id} · {formatTimestamp(event.start_time)} →{" "}
          {formatTimestamp(event.end_time)}
        </p>
      </header>

      {event.disclaimer && event.data_mode && (
        <ProvenanceNotice mode={event.data_mode} text={event.disclaimer} />
      )}

      {/* ---- Operational impact ---- */}
      <section className="mt-5 grid grid-cols-2 gap-3 sm:grid-cols-5">
        <Stat label="Duration" value={formatDuration(event.duration_minutes)} />
        <Stat
          label="Energy lost"
          value={formatKwh(event.total_lost_kwh).split(" ")[0]}
          unit={formatKwh(event.total_lost_kwh).split(" ")[1]}
          tone="warn"
        />
        <Stat label="Shortfall" value={`${(shortfall * 100).toFixed(0)}%`} tone="warn" />
        <Stat label="Actual" value={event.mean_actual_kw.toFixed(1)} unit="kW" />
        <Stat label="Expected" value={event.mean_predicted_kw.toFixed(1)} unit="kW" />
      </section>

      {/* ---- The reading, in plain language ---- */}
      <section className="mt-5">
        <Card>
          <CardTitle>What the model saw</CardTitle>
          <p className="text-sm leading-relaxed text-slate-300">{event.explanation}</p>
          {event.mean_expected_lower_kw !== null && (
            <dl className="mt-3 grid grid-cols-3 gap-2 text-xs">
              <div className="rounded border border-slate-800 px-2 py-1.5">
                <dt className="text-slate-500">Expected</dt>
                <dd className="tabular-nums text-slate-200">
                  {event.mean_predicted_kw.toFixed(1)} kW
                </dd>
              </div>
              <div className="rounded border border-cyan-900/60 px-2 py-1.5">
                <dt className="text-cyan-500/80">Lower bound of healthy range</dt>
                <dd className="tabular-nums text-cyan-300">
                  {event.mean_expected_lower_kw.toFixed(1)} kW
                </dd>
              </div>
              <div className="rounded border border-red-900/60 px-2 py-1.5">
                <dt className="text-red-500/80">Actual</dt>
                <dd className="tabular-nums text-red-300">
                  {event.mean_actual_kw.toFixed(1)} kW
                </dd>
              </div>
            </dl>
          )}
        </Card>
      </section>

      {/* ---- Chart ---- */}
      <section className="mt-5">
        <Card>
          <CardTitle hint="event window highlighted">Generation through the event</CardTitle>
          {timeseries.length ? (
            <>
              <GenerationChart
                points={timeseries}
                height={300}
                highlightStart={event.start_time}
                highlightEnd={event.end_time}
              />
              <ChartLegend />
            </>
          ) : (
            <Empty>No telemetry for this window.</Empty>
          )}
        </Card>
      </section>

      {/* ---- Spatial attribution ---- */}
      {spatial_context && (
        <section className="mt-5">
          <Card>
            <CardTitle hint={`within ${spatial_context.radius_km.toFixed(0)} km`}>
              Was it this site, or the weather?
            </CardTitle>

            <div
              className={`rounded-lg border px-3 py-2.5 ${
                SCOPE_STYLE[spatial_context.scope] ?? SCOPE_STYLE.indeterminate
              }`}
            >
              <div className="flex items-center gap-2">
                <span className="text-sm font-semibold">
                  {SCOPE_HEADLINE[spatial_context.scope] ?? spatial_context.scope}
                </span>
                <span className="rounded bg-black/30 px-1.5 py-0.5 text-[10px] uppercase tracking-wide">
                  {spatial_context.confidence} confidence
                </span>
              </div>
              <p className="mt-1.5 text-xs leading-relaxed opacity-90">
                {spatial_context.explanation}
              </p>
            </div>

            <div className="mt-3 grid grid-cols-2 gap-2 text-xs sm:grid-cols-4">
              <Evidence
                label="This site"
                value={`${(spatial_context.site_normalised_residual * 100).toFixed(1)}%`}
                hint="of nameplate vs expected"
              />
              <Evidence
                label="Neighbour median"
                value={
                  spatial_context.neighbor_residual_median !== null
                    ? `${(spatial_context.neighbor_residual_median * 100).toFixed(1)}%`
                    : "—"
                }
                hint="same window"
              />
              <Evidence
                label="Neighbours affected"
                value={`${spatial_context.neighbors_affected} / ${spatial_context.neighbor_count}`}
                hint="also below par"
              />
              <Evidence
                label="Excess deviation"
                value={
                  spatial_context.excess_deviation !== null
                    ? `${(spatial_context.excess_deviation * 100).toFixed(1)} pp`
                    : "—"
                }
                hint="worse than neighbourhood"
              />
            </div>

            {neighbor_comparison.length > 0 && (
              <div className="mt-3 overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead className="text-slate-500">
                    <tr className="border-b border-slate-800">
                      <th className="py-1.5 pr-2 font-medium">Neighbour</th>
                      <th className="px-2 py-1.5 text-right font-medium">Actual kW</th>
                      <th className="px-2 py-1.5 text-right font-medium">Expected kW</th>
                      <th className="px-2 py-1.5 text-right font-medium">Ratio</th>
                      <th className="px-2 py-1.5 text-right font-medium">Flagged</th>
                    </tr>
                  </thead>
                  <tbody>
                    {neighbor_comparison.map((n) => (
                      <tr key={n.site_id} className="border-b border-slate-800/60 last:border-0">
                        <td className="py-1.5 pr-2">
                          <Link
                            href={`/sites/${n.site_id}`}
                            className="text-slate-300 hover:text-cyan-300"
                          >
                            {n.name}
                          </Link>
                        </td>
                        <td className="px-2 py-1.5 text-right tabular-nums text-slate-400">
                          {n.actual_kw_mean.toFixed(1)}
                        </td>
                        <td className="px-2 py-1.5 text-right tabular-nums text-slate-400">
                          {n.expected_kw_mean.toFixed(1)}
                        </td>
                        <td className="px-2 py-1.5 text-right tabular-nums">
                          <span
                            className={
                              n.expected_ratio !== null && n.expected_ratio < 0.9
                                ? "text-amber-400"
                                : "text-emerald-400"
                            }
                          >
                            {n.expected_ratio !== null
                              ? `${(n.expected_ratio * 100).toFixed(0)}%`
                              : "—"}
                          </span>
                        </td>
                        <td className="px-2 py-1.5 text-right tabular-nums text-slate-400">
                          {n.anomaly_intervals}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            <p className="mt-2 text-[11px] leading-relaxed text-slate-600">
              This is not a fault classifier. It answers only whether the deviation
              was shared with nearby sites, and shows the evidence so you can
              disagree with it.
            </p>
          </Card>
        </section>
      )}

      {/* ---- Actions and attribution ---- */}
      <div className="mt-5 grid gap-5 lg:grid-cols-2">
        {recommended_actions.length > 0 && (
          <Card>
            <CardTitle>Suggested next steps</CardTitle>
            <ul className="space-y-2">
              {recommended_actions.map((action, i) => (
                <li key={i} className="flex gap-2 text-xs leading-relaxed text-slate-300">
                  <span className="mt-0.5 flex h-4 w-4 shrink-0 items-center justify-center rounded-full bg-slate-800 text-[10px] text-cyan-400">
                    {i + 1}
                  </span>
                  {action}
                </li>
              ))}
            </ul>
          </Card>
        )}

        {contributors.length > 0 && (
          <Card>
            <CardTitle hint="SHAP, on the expected-generation model">
              What drove the expectation
            </CardTitle>
            <ul className="space-y-1.5">
              {contributors.map((c) => (
                <li key={c.feature} className="flex items-center gap-2 text-xs">
                  <span className="w-40 shrink-0 truncate font-mono text-[11px] text-slate-400">
                    {c.feature}
                  </span>
                  <span className="tabular-nums text-slate-600">
                    {c.feature_value.toFixed(1)}
                  </span>
                  <span className="ml-auto flex items-center gap-1.5">
                    <span
                      className={`inline-block h-1.5 rounded ${
                        c.direction === "positive" ? "bg-emerald-500" : "bg-red-500"
                      }`}
                      style={{
                        width: `${Math.min(64, Math.abs(c.shap_value) * 2 + 6)}px`,
                      }}
                    />
                    <span
                      className={`w-14 text-right tabular-nums ${
                        c.direction === "positive" ? "text-emerald-400" : "text-red-400"
                      }`}
                    >
                      {c.shap_value > 0 ? "+" : ""}
                      {c.shap_value.toFixed(1)}
                    </span>
                  </span>
                </li>
              ))}
            </ul>
            <p className="mt-2 text-[11px] leading-relaxed text-slate-600">
              These explain why the model <em>expected</em> what it did — not why
              the array then fell short of that expectation.
            </p>
          </Card>
        )}
      </div>
    </div>
  );
}

function Evidence({
  label,
  value,
  hint,
}: {
  label: string;
  value: string;
  hint: string;
}) {
  return (
    <div className="rounded border border-slate-800 px-2 py-1.5">
      <div className="text-[10px] uppercase tracking-wide text-slate-500">{label}</div>
      <div className="mt-0.5 tabular-nums text-slate-200">{value}</div>
      <div className="text-[10px] text-slate-600">{hint}</div>
    </div>
  );
}
