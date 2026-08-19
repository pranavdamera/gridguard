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

export default async function SiteDetailPage(props: {
  params: Promise<{ siteId: string }>;
}) {
  const { siteId } = await props.params;

  const [{ data: site, error }, { data: series }, { data: events }, { data: metrics }] =
    await Promise.all([
      tryFetch(() => api.site(siteId)),
      tryFetch(() => api.timeseries(siteId, { limit: 900 })),
      tryFetch(() => api.events({ site_id: siteId, limit: 10 })),
      tryFetch(() => api.metrics(siteId)),
    ]);

  if (error || !site) {
    return (
      <div className="mx-auto max-w-5xl px-4 py-8">
        <Link href="/demo" className="text-xs text-slate-500 hover:text-slate-300">
          ← Fleet
        </Link>
        <div className="mt-3">
          <ErrorPanel error={error ?? "Site not found"} />
        </div>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-5xl px-4 py-8">
      <Link href="/demo" className="text-xs text-slate-500 hover:text-slate-300">
        ← Fleet
      </Link>

      <header className="mt-2 mb-4">
        <div className="flex flex-wrap items-center gap-2">
          <h1 className="text-xl font-semibold tracking-tight text-slate-100">
            {site.name}
          </h1>
          <DataModeBadge mode={site.data_mode} />
        </div>
        <p className="mt-1 text-sm text-slate-400">
          {site.capacity_kw} kW · {site.latitude.toFixed(4)}, {site.longitude.toFixed(4)}
          {site.tilt_deg !== null && (
            <> · tilt {site.tilt_deg}° / azimuth {site.azimuth_deg}°</>
          )}
          {site.source_system_id && <> · PVDAQ system {site.source_system_id}</>}
        </p>
      </header>

      <ProvenanceNotice mode={site.data_mode} text={site.disclaimer} />

      {site.notes && (
        <p className="mt-2 text-xs leading-relaxed text-slate-500">{site.notes}</p>
      )}

      {/* ---- Generation ---- */}
      <section className="mt-5">
        <Card>
          <CardTitle
            hint={
              series
                ? `${series.points.length} of ${series.total_points} intervals`
                : undefined
            }
          >
            Actual vs expected generation
          </CardTitle>
          {series && series.points.length > 0 ? (
            <>
              <GenerationChart points={series.points} height={320} />
              <ChartLegend />
              <p className="mt-2 text-[11px] leading-relaxed text-slate-500">
                The shaded band is the calibrated range of healthy output. Actual
                generation falling below it is what GridGuard flags — the band has
                a stated false-alarm rate, unlike a hand-picked threshold.
              </p>
            </>
          ) : (
            <Empty>No telemetry available for this site.</Empty>
          )}
        </Card>
      </section>

      {/* ---- Model performance ---- */}
      {metrics && (
        <section className="mt-5">
          <Card>
            <CardTitle
              hint={
                metrics.train_test_split_date
                  ? `held-out from ${metrics.train_test_split_date}`
                  : undefined
              }
            >
              Model performance on held-out data
            </CardTitle>

            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              <Stat label="Best model" value={metrics.best_model} />
              {metrics.models.slice(0, 1).map((m) => (
                <Stat key={m.model} label="MAE" value={m.mae_kw.toFixed(2)} unit="kW" />
              ))}
              {metrics.models.slice(0, 1).map((m) => (
                <Stat key={m.model} label="RMSE" value={m.rmse_kw.toFixed(2)} unit="kW" />
              ))}
              {metrics.models.slice(0, 1).map((m) => (
                <Stat key={m.model} label="R²" value={m.r2.toFixed(4)} />
              ))}
            </div>

            <div className="mt-4 overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead className="text-slate-500">
                  <tr className="border-b border-slate-800">
                    <th className="py-1.5 pr-2 font-medium">Forecast model (lag-aware)</th>
                    <th className="px-2 py-1.5 text-right font-medium">MAE kW</th>
                    <th className="px-2 py-1.5 text-right font-medium">RMSE kW</th>
                    <th className="px-2 py-1.5 text-right font-medium">R²</th>
                  </tr>
                </thead>
                <tbody>
                  {metrics.models.map((m) => (
                    <tr key={m.model} className="border-b border-slate-800/60 last:border-0">
                      <td className="py-1.5 pr-2 text-slate-300">{m.model}</td>
                      <td className="px-2 py-1.5 text-right tabular-nums text-slate-400">
                        {m.mae_kw.toFixed(3)}
                      </td>
                      <td className="px-2 py-1.5 text-right tabular-nums text-slate-400">
                        {m.rmse_kw.toFixed(3)}
                      </td>
                      <td className="px-2 py-1.5 text-right tabular-nums text-slate-400">
                        {m.r2.toFixed(4)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {metrics.weather_only_models.length > 0 && (
              <div className="mt-4 overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead className="text-slate-500">
                    <tr className="border-b border-slate-800">
                      <th className="py-1.5 pr-2 font-medium">
                        Expected-generation model (weather-only)
                      </th>
                      <th className="px-2 py-1.5 text-right font-medium">MAE kW</th>
                      <th className="px-2 py-1.5 text-right font-medium">RMSE kW</th>
                      <th className="px-2 py-1.5 text-right font-medium">R²</th>
                    </tr>
                  </thead>
                  <tbody>
                    {metrics.weather_only_models.map((m) => (
                      <tr key={m.model} className="border-b border-slate-800/60 last:border-0">
                        <td className="py-1.5 pr-2 text-slate-300">{m.model}</td>
                        <td className="px-2 py-1.5 text-right tabular-nums text-slate-400">
                          {m.mae_kw.toFixed(3)}
                        </td>
                        <td className="px-2 py-1.5 text-right tabular-nums text-slate-400">
                          {m.rmse_kw.toFixed(3)}
                        </td>
                        <td className="px-2 py-1.5 text-right tabular-nums text-slate-400">
                          {m.r2.toFixed(4)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            {metrics.evaluation_note && (
              <p className="mt-2 text-[11px] leading-relaxed text-slate-500">
                {metrics.evaluation_note}
              </p>
            )}

            {metrics.conformal_coverage && (
              <div className="mt-4">
                <p className="text-[11px] font-medium uppercase tracking-wide text-slate-500">
                  Conformal coverage on held-out healthy intervals
                </p>
                <div className="mt-1.5 flex flex-wrap gap-2">
                  {Object.entries(metrics.conformal_coverage).map(([bucket, value]) => (
                    <span
                      key={bucket}
                      className="rounded border border-slate-800 px-2 py-1 text-[11px] tabular-nums text-slate-400"
                    >
                      {bucket}:{" "}
                      <span
                        className={value >= 0.95 ? "text-emerald-400" : "text-amber-400"}
                      >
                        {(value * 100).toFixed(1)}%
                      </span>
                    </span>
                  ))}
                </div>
                <p className="mt-1.5 text-[11px] text-slate-600">
                  Target is ≥95%. Measured on intervals neither the model nor the
                  calibration saw.
                </p>
              </div>
            )}
          </Card>
        </section>
      )}

      {/* ---- Events ---- */}
      <section className="mt-5">
        <Card>
          <CardTitle hint={events ? `${events.total_events} total` : undefined}>
            Underperformance events
          </CardTitle>
          {!events || events.events.length === 0 ? (
            <Empty>No events detected for this site in the evaluation window.</Empty>
          ) : (
            <ul className="space-y-2">
              {events.events.map((event) => (
                <li key={event.global_event_id ?? event.event_id}>
                  <Link
                    href={`/events/${encodeURIComponent(
                      event.global_event_id ?? String(event.event_id),
                    )}`}
                    className="block rounded border border-slate-800 px-3 py-2 transition-colors hover:border-slate-700 hover:bg-slate-800/40"
                  >
                    <div className="flex flex-wrap items-center gap-2">
                      <SeverityBadge severity={event.severity} />
                      <span className="text-xs text-slate-400">
                        {formatTimestamp(event.start_time)}
                      </span>
                      <span className="text-xs text-slate-600">
                        {formatDuration(event.duration_minutes)}
                      </span>
                      <span className="ml-auto text-xs tabular-nums text-slate-400">
                        {formatKwh(event.total_lost_kwh)}
                      </span>
                    </div>
                    <p className="mt-1 line-clamp-2 text-[11px] leading-relaxed text-slate-500">
                      {event.explanation}
                    </p>
                  </Link>
                </li>
              ))}
            </ul>
          )}
        </Card>
      </section>
    </div>
  );
}
