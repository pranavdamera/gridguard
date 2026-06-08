"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import {
  api,
  DemoScenarioResponse,
  EventsResponse,
  AnomalyResponse,
  MetricsResponse,
  SitesResponse,
  HealthResponse,
} from "@/lib/api";
import { FleetHealthCards } from "@/components/FleetHealthCards";
import { EventTimeline } from "@/components/EventTimeline";
import { ExpectedVsActualChart } from "@/components/ExpectedVsActualChart";
import { WeatherContextPanel } from "@/components/WeatherContextPanel";
import { EventExplanationCard } from "@/components/EventExplanationCard";
import { RecommendedActions } from "@/components/RecommendedActions";
import { ModelMetricsPanel } from "@/components/ModelMetricsPanel";
import { SiteMap } from "@/components/SiteMap";
import { MethodologyNotice } from "@/components/MethodologyNotice";
import { LoadingSpinner, ErrorState } from "@/components/ui/LoadingSpinner";

export default function DemoPage() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [demo, setDemo] = useState<DemoScenarioResponse | null>(null);
  const [events, setEvents] = useState<EventsResponse | null>(null);
  const [anomalies, setAnomalies] = useState<AnomalyResponse | null>(null);
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [sites, setSites] = useState<SitesResponse | null>(null);

  useEffect(() => {
    async function loadAll() {
      try {
        const [h, d, ev, an, m, s] = await Promise.all([
          api.health(),
          api.demoScenario(),
          api.events({ limit: 20 }),
          api.anomalies({ start: "2023-06-15", end: "2023-06-15T23:59", limit: 200, only_anomalies: false }),
          api.metrics().catch(() => null),
          api.sites(),
        ]);
        setHealth(h);
        setDemo(d);
        setEvents(ev);
        setAnomalies(an);
        setMetrics(m as MetricsResponse | null);
        setSites(s);
      } catch (e) {
        setError(
          (e as Error).message.includes("fetch")
            ? "Cannot reach the GridGuard API. Make sure the backend is running: `make api`"
            : (e as Error).message
        );
      } finally {
        setLoading(false);
      }
    }
    loadAll();
  }, []);

  if (loading) {
    return (
      <div className="max-w-6xl mx-auto px-4 py-12">
        <LoadingSpinner label="Loading demo data from GridGuard API…" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="max-w-6xl mx-auto px-4 py-12">
        <ErrorState message={error} />
        <div className="mt-4 text-sm text-slate-500">
          Run <code className="bg-slate-800 px-1 rounded">make demo-reset && make api</code> to start the backend, then reload.
        </div>
      </div>
    );
  }

  const demoEvent = demo?.event;
  const demoRecords = anomalies?.records ?? [];

  return (
    <div className="max-w-6xl mx-auto px-4 py-6 space-y-5">
      {/* Header */}
      <div className="flex items-start justify-between gap-4 flex-wrap">
        <div>
          <h1 className="text-xl font-semibold text-slate-100">Live Demo Dashboard</h1>
          <p className="text-sm text-slate-500 mt-0.5">
            GMU Fairfax Campus Solar Array · June 15, 2023 underperformance event
          </p>
        </div>
        {demoEvent && (
          <Link
            href={`/events/${demoEvent.event_id}`}
            className="px-4 py-2 bg-amber-700/30 border border-amber-700 text-amber-300 rounded text-sm hover:bg-amber-700/50 transition-colors"
          >
            ⚡ Jump to June 15 event →
          </Link>
        )}
      </div>

      <MethodologyNotice />

      {/* KPI row */}
      <FleetHealthCards health={health} events={events} anomalies={anomalies} />

      {/* Main layout: chart + events */}
      <div className="grid lg:grid-cols-3 gap-5">
        <div className="lg:col-span-2 space-y-5">
          <ExpectedVsActualChart
            records={demoRecords}
            title="June 15 — Expected vs Actual Generation (15-min intervals)"
          />
          <WeatherContextPanel records={demoRecords} />
        </div>
        <div className="space-y-5">
          <EventTimeline events={events?.events ?? []} highlightId={demoEvent?.event_id} />
        </div>
      </div>

      {/* Event explanation + actions */}
      {demoEvent && (
        <div className="grid lg:grid-cols-2 gap-5">
          <EventExplanationCard event={demoEvent} />
          <RecommendedActions actions={demo?.recommended_actions} />
        </div>
      )}

      {/* Fleet map + model metrics */}
      <div className="grid lg:grid-cols-2 gap-5">
        {sites && <SiteMap sites={sites.sites} activeSiteId="gmu_fairfax" />}
        {metrics && <ModelMetricsPanel metrics={metrics} />}
      </div>
    </div>
  );
}
