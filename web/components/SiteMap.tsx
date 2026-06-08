"use client";

import { SiteRecord } from "@/lib/api";
import { Card, CardTitle } from "@/components/ui/Card";
import Link from "next/link";

interface Props {
  sites: SiteRecord[];
  activeSiteId?: string;
}

// Simple SVG scatter plot of site lat/lon — no external map tile dependency
function latLonToSvg(lat: number, lon: number, bounds: { minLat: number; maxLat: number; minLon: number; maxLon: number }, w: number, h: number) {
  const x = ((lon - bounds.minLon) / (bounds.maxLon - bounds.minLon)) * w;
  const y = ((bounds.maxLat - lat) / (bounds.maxLat - bounds.minLat)) * h;
  return { x: Math.max(8, Math.min(w - 8, x)), y: Math.max(8, Math.min(h - 8, y)) };
}

export function SiteMap({ sites, activeSiteId }: Props) {
  if (!sites.length) return null;

  const lats = sites.map(s => s.latitude);
  const lons = sites.map(s => s.longitude);
  const pad = 0.05;
  const bounds = {
    minLat: Math.min(...lats) - pad,
    maxLat: Math.max(...lats) + pad,
    minLon: Math.min(...lons) - pad,
    maxLon: Math.max(...lons) + pad,
  };
  const W = 480, H = 220;

  return (
    <Card>
      <CardTitle>Fleet Map — DMV Region</CardTitle>
      <div className="relative">
        <svg
          viewBox={`0 0 ${W} ${H}`}
          className="w-full h-auto bg-slate-900 rounded border border-slate-700"
          style={{ maxHeight: 240 }}
        >
          {/* Grid lines */}
          {[...Array(5)].map((_, i) => (
            <line key={i} x1={0} y1={(H / 4) * i} x2={W} y2={(H / 4) * i} stroke="#1e293b" strokeWidth={1} />
          ))}
          {[...Array(7)].map((_, i) => (
            <line key={i} x1={(W / 6) * i} y1={0} x2={(W / 6) * i} y2={H} stroke="#1e293b" strokeWidth={1} />
          ))}

          {sites.map((site) => {
            const { x, y } = latLonToSvg(site.latitude, site.longitude, bounds, W, H);
            const isActive = site.site_id === activeSiteId;
            const r = Math.max(5, Math.sqrt(site.capacity_kw) * 0.35);
            return (
              <g key={site.site_id}>
                <circle
                  cx={x}
                  cy={y}
                  r={r + 3}
                  fill={isActive ? "#06b6d4" : "transparent"}
                  opacity={0.15}
                />
                <circle
                  cx={x}
                  cy={y}
                  r={r}
                  fill={isActive ? "#06b6d4" : "#64748b"}
                  stroke={isActive ? "#22d3ee" : "#475569"}
                  strokeWidth={1.5}
                />
                <text x={x + r + 4} y={y + 4} fill="#94a3b8" fontSize={9}>
                  {site.name.replace("GMU ", "").replace(" Campus Solar Array", "").replace(" Solar", "")}
                </text>
              </g>
            );
          })}
        </svg>
        <div className="text-xs text-slate-600 mt-2">
          Circle size ≈ capacity. Click a site for details.
        </div>
      </div>
      <div className="mt-3 grid grid-cols-2 gap-1.5">
        {sites.map((s) => (
          <Link
            key={s.site_id}
            href={`/sites/${s.site_id}`}
            className="flex items-center gap-2 text-xs text-slate-400 hover:text-cyan-300 transition-colors"
          >
            <span className={`w-2 h-2 rounded-full ${s.site_id === activeSiteId ? "bg-cyan-400" : "bg-slate-600"}`} />
            {s.name} <span className="text-slate-600">({s.capacity_kw} kW)</span>
          </Link>
        ))}
      </div>
    </Card>
  );
}
