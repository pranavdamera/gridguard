import Link from "next/link";
import { api } from "@/lib/api";
import { Card, CardTitle } from "@/components/ui/Card";
import { MethodologyNotice } from "@/components/MethodologyNotice";

export const revalidate = 60;

interface Props {
  params: Promise<{ siteId: string }>;
}

export default async function SiteDetailPage({ params }: Props) {
  const { siteId } = await params;
  let site = null;
  let events = null;
  let error = null;

  try {
    [site, events] = await Promise.all([
      api.site(siteId),
      api.events({ site_id: siteId, limit: 10 }),
    ]);
  } catch (e) {
    error = (e as Error).message;
  }

  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      <div className="mb-2">
        <Link href="/sites" className="text-xs text-slate-500 hover:text-slate-300">← All sites</Link>
      </div>

      {error && (
        <div className="text-red-400 text-sm border border-red-800 bg-red-950/30 rounded p-4 mb-6">
          {error}
        </div>
      )}

      {site && (
        <>
          <div className="mb-6">
            <h1 className="text-xl font-semibold text-slate-100">{site.name}</h1>
            <p className="text-slate-500 text-sm mt-1">{site.region}</p>
          </div>

          <div className="mb-5">
            <MethodologyNotice />
          </div>

          <div className="grid sm:grid-cols-2 gap-4 mb-6">
            <Card>
              <CardTitle>Site Info</CardTitle>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-slate-500">Site ID</span>
                  <span className="font-mono text-slate-300">{site.site_id}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Capacity</span>
                  <span className="text-cyan-300 font-mono">{site.capacity_kw} kW</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Latitude</span>
                  <span className="font-mono text-slate-300">{site.latitude.toFixed(4)}° N</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Longitude</span>
                  <span className="font-mono text-slate-300">{Math.abs(site.longitude).toFixed(4)}° W</span>
                </div>
              </div>
              {site.notes && (
                <div className="text-slate-500 text-xs mt-4 leading-relaxed border-t border-slate-700 pt-3">
                  {site.notes}
                </div>
              )}
            </Card>

            <Card>
              <CardTitle>Quick Actions</CardTitle>
              <div className="space-y-2 text-sm">
                <Link
                  href={`/demo?site=${site.site_id}`}
                  className="block px-3 py-2 bg-cyan-900/30 border border-cyan-800 text-cyan-300 rounded text-xs hover:bg-cyan-900/50 transition-colors"
                >
                  View demo dashboard for this site →
                </Link>
                <div className="text-xs text-slate-600 pt-1">
                  Demo data is synthetic. Real-data adapters planned.
                </div>
              </div>
            </Card>
          </div>

          {events && events.events.length > 0 && (
            <Card>
              <CardTitle>Recent Events ({events.total_events} total)</CardTitle>
              <div className="space-y-2">
                {events.events.map((evt) => (
                  <Link
                    key={evt.event_id}
                    href={`/events/${evt.event_id}`}
                    className="flex items-center gap-3 p-3 rounded border border-slate-700 hover:border-slate-500 transition-colors text-sm"
                  >
                    <span className={`px-1.5 py-0.5 rounded text-xs font-medium ${
                      evt.severity === "high" ? "bg-red-900/60 text-red-300" :
                      evt.severity === "medium" ? "bg-amber-900/60 text-amber-300" :
                      "bg-emerald-900/60 text-emerald-300"
                    }`}>{evt.severity}</span>
                    <span className="text-slate-300">
                      {new Date(evt.start_time).toLocaleDateString()} — {evt.duration_minutes} min
                    </span>
                    <span className="ml-auto text-amber-300 font-mono text-xs">{evt.total_lost_kwh.toFixed(1)} kWh lost</span>
                  </Link>
                ))}
              </div>
            </Card>
          )}

          {events && events.events.length === 0 && (
            <div className="text-slate-500 text-sm">No anomaly events found for this site.</div>
          )}
        </>
      )}
    </div>
  );
}
