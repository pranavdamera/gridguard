import Link from "next/link";
import { api, tryFetch } from "@/lib/api";
import { FleetMapPanel } from "@/components/FleetMapPanel";
import {
  Card,
  CardTitle,
  DataModeBadge,
  Empty,
  ErrorPanel,
  HealthBadge,
  SeverityBadge,
  Stat,
  formatDuration,
  formatKwh,
  formatTimestamp,
} from "@/components/ui/Primitives";

export const revalidate = 60;

export default async function DemoPage() {
  const [{ data: fleet, error }, { data: events }, { data: scenario }] =
    await Promise.all([
      tryFetch(() => api.fleetSummary()),
      tryFetch(() => api.events({ limit: 8 })),
      tryFetch(() => api.demoScenario()),
    ]);

  if (error || !fleet) {
    return (
      <div className="mx-auto max-w-6xl px-4 py-8">
        <ErrorPanel error={error ?? "No fleet data"} />
      </div>
    );
  }

  const needingAttention = fleet.sites_critical + fleet.sites_warning;
  const worst = fleet.sites.find((s) => s.health === "critical") ?? fleet.sites[0];

  return (
    <div className="mx-auto max-w-6xl px-4 py-8">
      <header className="mb-5">
        <h1 className="text-xl font-semibold tracking-tight text-slate-100">
          Fleet overview
        </h1>
        <p className="mt-1 text-sm text-slate-400">
          {fleet.total_sites} sites · {(fleet.total_capacity_kw / 1000).toFixed(2)} MW ·{" "}
          {fleet.real_sites} measured, {fleet.synthetic_sites} simulated
          {fleet.window_end && (
            <> · window ending {formatTimestamp(fleet.window_end)}</>
          )}
        </p>
      </header>

      {/* ---- Fleet KPIs ---- */}
      <section className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
        <Stat label="Total capacity" value={(fleet.total_capacity_kw / 1000).toFixed(2)} unit="MW" />
        <Stat
          label="Healthy"
          value={fleet.sites_healthy}
          tone={fleet.sites_healthy === fleet.total_sites ? "good" : "neutral"}
        />
        <Stat label="Warning" value={fleet.sites_warning} tone={fleet.sites_warning ? "warn" : "neutral"} />
        <Stat label="Critical" value={fleet.sites_critical} tone={fleet.sites_critical ? "bad" : "neutral"} />
        <Stat
          label="Estimated lost"
          value={formatKwh(fleet.total_lost_kwh).split(" ")[0]}
          unit={formatKwh(fleet.total_lost_kwh).split(" ")[1]}
          tone={fleet.total_lost_kwh > 0 ? "warn" : "neutral"}
        />
        <Stat label="Open events" value={fleet.active_events} />
      </section>

      {/* ---- Map ---- */}
      <section className="mt-6">
        <FleetMapPanel sites={fleet.sites} bounds={fleet.bounds} />
      </section>

      {/* ---- Guided walkthrough entry ---- */}
      {scenario?.event && (
        <section className="mt-6">
          <Card className="border-red-900/50 bg-red-950/10">
            <div className="flex flex-wrap items-center gap-2">
              <SeverityBadge severity={scenario.event.severity} />
              <DataModeBadge mode={scenario.data_mode} />
              <span className="text-xs text-slate-500">
                {formatTimestamp(scenario.event.start_time)}
              </span>
            </div>
            <h2 className="mt-2 text-sm font-semibold text-slate-100">
              {scenario.site_name} — underperformance detected
            </h2>
            <p className="mt-1.5 max-w-3xl text-xs leading-relaxed text-slate-400">
              {scenario.event.explanation}
            </p>
            {scenario.spatial_context && (
              <p className="mt-2 max-w-3xl rounded border border-slate-800 bg-slate-900/60 px-3 py-2 text-xs leading-relaxed text-slate-300">
                <span className="font-semibold uppercase tracking-wide text-cyan-400">
                  {scenario.spatial_context.scope.replace("_", " ")}
                </span>
                <span className="mx-2 text-slate-600">|</span>
                {scenario.spatial_context.explanation}
              </p>
            )}
            {scenario.jump_to_event_id && (
              <Link
                href={`/events/${encodeURIComponent(scenario.jump_to_event_id)}`}
                className="mt-3 inline-block rounded-md bg-cyan-600 px-3 py-1.5 text-xs font-medium text-white transition-colors hover:bg-cyan-500"
              >
                Investigate this event →
              </Link>
            )}
          </Card>
        </section>
      )}

      <div className="mt-6 grid gap-5 lg:grid-cols-5">
        {/* ---- Site table ---- */}
        <section className="lg:col-span-3">
          <Card>
            <CardTitle hint={`${needingAttention} needing attention`}>
              Sites, worst first
            </CardTitle>
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead className="text-slate-500">
                  <tr className="border-b border-slate-800">
                    <th className="py-1.5 pr-2 font-medium">Site</th>
                    <th className="px-2 py-1.5 font-medium">Health</th>
                    <th className="px-2 py-1.5 text-right font-medium">kW</th>
                    <th className="px-2 py-1.5 text-right font-medium">vs expected</th>
                    <th className="px-2 py-1.5 text-right font-medium">Lost</th>
                    <th className="px-2 py-1.5 font-medium">Scope</th>
                  </tr>
                </thead>
                <tbody>
                  {fleet.sites.map((site) => (
                    <tr
                      key={site.site_id}
                      className="border-b border-slate-800/60 last:border-0 hover:bg-slate-800/30"
                    >
                      <td className="py-2 pr-2">
                        <Link
                          href={`/sites/${site.site_id}`}
                          className="text-slate-300 hover:text-cyan-300"
                        >
                          {site.name}
                        </Link>
                        <span
                          className={`ml-1.5 rounded px-1 text-[9px] ${
                            site.data_mode === "real"
                              ? "bg-emerald-950 text-emerald-400"
                              : "bg-amber-950 text-amber-400"
                          }`}
                        >
                          {site.data_mode === "real" ? "measured" : "sim"}
                        </span>
                      </td>
                      <td className="px-2 py-2">
                        <HealthBadge health={site.health} />
                      </td>
                      <td className="px-2 py-2 text-right tabular-nums text-slate-400">
                        {site.capacity_kw}
                      </td>
                      <td className="px-2 py-2 text-right tabular-nums">
                        {site.expected_ratio !== null ? (
                          <span
                            className={
                              site.expected_ratio < 0.9
                                ? "text-amber-400"
                                : "text-slate-400"
                            }
                          >
                            {(site.expected_ratio * 100).toFixed(0)}%
                          </span>
                        ) : (
                          <span className="text-slate-600">—</span>
                        )}
                      </td>
                      <td className="px-2 py-2 text-right tabular-nums text-slate-400">
                        {site.lost_energy_kwh > 0
                          ? formatKwh(site.lost_energy_kwh)
                          : "—"}
                      </td>
                      <td className="px-2 py-2 text-slate-500">
                        {site.anomaly_scope
                          ? site.anomaly_scope.replace("_", " ")
                          : "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </section>

        {/* ---- Recent events ---- */}
        <section className="lg:col-span-2">
          <Card>
            <CardTitle hint={events ? `${events.total_events} total` : undefined}>
              Recent events
            </CardTitle>
            {!events || events.events.length === 0 ? (
              <Empty>No underperformance events in this window.</Empty>
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
                      <div className="flex items-center gap-2">
                        <SeverityBadge severity={event.severity} />
                        <span className="truncate text-xs text-slate-300">
                          {event.site_name ?? event.site_id}
                        </span>
                        <span className="ml-auto shrink-0 text-[11px] tabular-nums text-slate-500">
                          {formatKwh(event.total_lost_kwh)}
                        </span>
                      </div>
                      <div className="mt-1 text-[11px] text-slate-500">
                        {formatTimestamp(event.start_time)} ·{" "}
                        {formatDuration(event.duration_minutes)}
                      </div>
                    </Link>
                  </li>
                ))}
              </ul>
            )}
          </Card>

          {worst && (
            <Card className="mt-4">
              <CardTitle>Highest priority</CardTitle>
              <div className="flex items-center gap-2">
                <HealthBadge health={worst.health} />
                <DataModeBadge mode={worst.data_mode} />
              </div>
              <p className="mt-2 text-sm text-slate-200">{worst.name}</p>
              {worst.scope_explanation && (
                <p className="mt-1.5 text-xs leading-relaxed text-slate-400">
                  {worst.scope_explanation}
                </p>
              )}
              <Link
                href={`/sites/${worst.site_id}`}
                className="mt-3 inline-block text-xs text-cyan-400 hover:text-cyan-300"
              >
                Open site analytics →
              </Link>
            </Card>
          )}
        </section>
      </div>
    </div>
  );
}
