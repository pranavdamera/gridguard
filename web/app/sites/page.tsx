import Link from "next/link";
import { api } from "@/lib/api";
import { Card } from "@/components/ui/Card";
import { MethodologyNotice } from "@/components/MethodologyNotice";

export const revalidate = 60;

export default async function SitesPage() {
  let sites = null;
  let error = null;
  try {
    const res = await api.sites();
    sites = res.sites;
  } catch (e) {
    error = (e as Error).message;
  }

  return (
    <div className="max-w-5xl mx-auto px-4 py-8">
      <div className="mb-6">
        <h1 className="text-xl font-semibold text-slate-100">Fleet Overview</h1>
        <p className="text-sm text-slate-500 mt-1">
          {sites?.length ?? 0} illustrative DC/Northern Virginia campus PV assets
        </p>
      </div>

      <div className="mb-5">
        <MethodologyNotice />
      </div>

      {error && (
        <div className="text-red-400 text-sm border border-red-800 bg-red-950/30 rounded p-4 mb-6">
          {error} — make sure the backend is running.
        </div>
      )}

      {sites && (
        <div className="grid sm:grid-cols-2 gap-4">
          {sites.map((site) => (
            <Link key={site.site_id} href={`/sites/${site.site_id}`}>
              <Card className="hover:border-slate-500 transition-colors cursor-pointer h-full">
                <div className="flex items-start justify-between mb-3">
                  <div>
                    <div className="text-slate-200 font-medium text-sm">{site.name}</div>
                    <div className="text-slate-500 text-xs mt-0.5">{site.region}</div>
                  </div>
                  <div className="text-right">
                    <div className="text-cyan-300 font-mono text-sm">{site.capacity_kw} kW</div>
                    <div className="text-slate-600 text-xs">nameplate</div>
                  </div>
                </div>
                <div className="text-xs text-slate-600 font-mono">
                  {site.latitude.toFixed(4)}° N, {Math.abs(site.longitude).toFixed(4)}° W
                </div>
                {site.notes && (
                  <div className="text-slate-500 text-xs mt-2 leading-relaxed">{site.notes}</div>
                )}
              </Card>
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}
