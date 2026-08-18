import Link from "next/link";
import { api, tryFetch } from "@/lib/api";
import {
  Card,
  DataModeBadge,
  ErrorPanel,
} from "@/components/ui/Primitives";

export const revalidate = 300;

export default async function SitesPage() {
  const { data, error } = await tryFetch(() => api.sites());

  if (error || !data) {
    return (
      <div className="mx-auto max-w-4xl px-4 py-8">
        <ErrorPanel error={error ?? "No sites available"} />
      </div>
    );
  }

  const groups: { mode: "real" | "synthetic"; label: string; blurb: string }[] = [
    {
      mode: "real",
      label: "Measured arrays",
      blurb:
        "Real installations with published system metadata and on-site weather instruments.",
    },
    {
      mode: "synthetic",
      label: "Simulated arrays",
      blurb:
        "Illustrative DMV campus fleet used for controlled fault injection. Not measured data.",
    },
  ];

  return (
    <div className="mx-auto max-w-4xl px-4 py-8">
      <header className="mb-6">
        <h1 className="text-xl font-semibold tracking-tight text-slate-100">Sites</h1>
        <p className="mt-1 text-sm text-slate-400">
          {data.total} sites · {data.real_count} measured · {data.synthetic_count} simulated
        </p>
      </header>

      {groups.map((group) => {
        const sites = data.sites.filter((s) => s.data_mode === group.mode);
        if (!sites.length) return null;
        return (
          <section key={group.mode} className="mb-8">
            <div className="mb-2 flex items-center gap-2">
              <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-400">
                {group.label}
              </h2>
              <DataModeBadge mode={group.mode} />
            </div>
            <p className="mb-3 text-xs text-slate-500">{group.blurb}</p>

            <div className="grid gap-3 sm:grid-cols-2">
              {sites.map((site) => (
                <Link key={site.site_id} href={`/sites/${site.site_id}`}>
                  <Card className="h-full transition-colors hover:border-slate-700">
                    <h3 className="text-sm font-medium text-slate-200">{site.name}</h3>
                    <p className="mt-1 text-xs text-slate-500">
                      {site.capacity_kw} kW · {site.latitude.toFixed(3)},{" "}
                      {site.longitude.toFixed(3)}
                    </p>
                    <p className="mt-0.5 text-[11px] text-slate-600">
                      {site.capacity_basis}
                    </p>
                    {site.notes && (
                      <p className="mt-2 line-clamp-3 text-[11px] leading-relaxed text-slate-500">
                        {site.notes}
                      </p>
                    )}
                  </Card>
                </Link>
              ))}
            </div>
          </section>
        );
      })}
    </div>
  );
}
