"use client";

/**
 * Interactive fleet map (MapLibre GL + OpenStreetMap raster tiles via CARTO).
 *
 * No API key and no paid tile service: CARTO's basemaps are free to use with
 * attribution, and MapLibre is the BSD-licensed fork of Mapbox GL. Attribution
 * for both OpenStreetMap and CARTO is rendered by the map's own control.
 *
 * Markers are sized by nameplate capacity and coloured by health, and the
 * marker ring encodes data mode — a dashed ring means simulated. Colour alone
 * is never the only carrier of that distinction: the popup and the site list
 * both state it in words.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import type { FleetBounds, Health, SiteStatusRecord } from "@/lib/api";
import type { Map as MapLibreMap, Marker as MapLibreMarker } from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";

interface Props {
  sites: SiteStatusRecord[];
  bounds?: FleetBounds | null;
  activeSiteId?: string;
  height?: number;
  onSelect?: (siteId: string) => void;
}

const HEALTH_COLOR: Record<Health, string> = {
  healthy: "#10b981",
  warning: "#f59e0b",
  critical: "#ef4444",
  no_data: "#64748b",
};

const HEALTH_LABEL: Record<Health, string> = {
  healthy: "Healthy",
  warning: "Warning",
  critical: "Critical",
  no_data: "No data",
};

/** Marker radius in pixels, scaled by capacity but bounded so a 500 kW site
 *  does not swamp a 60 kW one. */
function markerRadius(capacityKw: number): number {
  return Math.max(9, Math.min(26, Math.sqrt(capacityKw) * 0.85));
}

export function FleetMap({
  sites,
  bounds,
  activeSiteId,
  height = 420,
  onSelect,
}: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<MapLibreMap | null>(null);
  const markersRef = useRef<MapLibreMarker[]>([]);
  const [ready, setReady] = useState(false);
  const [failed, setFailed] = useState<string | null>(null);

  // Initialise the map once, on the client only. maplibre-gl touches `window`
  // at import time, so it is imported dynamically rather than at module scope.
  useEffect(() => {
    let cancelled = false;
    const container = containerRef.current;
    if (!container || mapRef.current) return;

    (async () => {
      try {
        // maplibre-gl exposes named exports and touches `window` at import
        // time, so it is imported dynamically inside the effect.
        const { Map, NavigationControl } = await import("maplibre-gl");
        if (cancelled || !containerRef.current) return;

        const map = new Map({
          container: containerRef.current,
          style: {
            version: 8,
            sources: {
              basemap: {
                type: "raster",
                tiles: [
                  "https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}@2x.png",
                  "https://b.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}@2x.png",
                  "https://c.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}@2x.png",
                ],
                tileSize: 256,
                attribution:
                  '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
              },
            },
            layers: [{ id: "basemap", type: "raster", source: "basemap" }],
          },
          center: [
            bounds?.center_longitude ?? -77.2,
            bounds?.center_latitude ?? 38.95,
          ],
          zoom: 8.2,
          attributionControl: { compact: true },
        });

        map.addControl(new NavigationControl({ showCompass: false }), "top-right");
        map.on("load", () => {
          if (!cancelled) setReady(true);
        });
        mapRef.current = map;
      } catch (e) {
        if (!cancelled) setFailed(e instanceof Error ? e.message : String(e));
      }
    })();

    return () => {
      cancelled = true;
      markersRef.current.forEach((m) => m.remove());
      markersRef.current = [];
      mapRef.current?.remove();
      mapRef.current = null;
    };
    // Bounds only seed the initial viewport; refitting happens below.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Fit the viewport to the fleet whenever the bounds change.
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !ready || !bounds) return;
    map.fitBounds(
      [
        [bounds.min_longitude, bounds.min_latitude],
        [bounds.max_longitude, bounds.max_latitude],
      ],
      { padding: 60, maxZoom: 12, duration: 0 },
    );
  }, [ready, bounds]);

  const buildMarker = useCallback(
    (site: SiteStatusRecord): HTMLElement => {
      const radius = markerRadius(site.capacity_kw);
      const color = HEALTH_COLOR[site.health];
      const isActive = site.site_id === activeSiteId;

      const el = document.createElement("button");
      el.type = "button";
      el.setAttribute(
        "aria-label",
        `${site.name}: ${HEALTH_LABEL[site.health]}, ${site.capacity_kw} kW, ${
          site.data_mode === "real" ? "measured data" : "simulated data"
        }`,
      );
      el.style.cssText = `
        width:${radius * 2}px;height:${radius * 2}px;border-radius:9999px;
        background:${color}33;border:2px ${site.data_mode === "real" ? "solid" : "dashed"} ${color};
        box-shadow:${isActive ? `0 0 0 4px ${color}55` : "none"};
        cursor:pointer;padding:0;display:flex;align-items:center;justify-content:center;
        transition:box-shadow .15s ease;
      `;

      const dot = document.createElement("span");
      dot.style.cssText = `width:${Math.max(4, radius * 0.42)}px;height:${Math.max(
        4,
        radius * 0.42,
      )}px;border-radius:9999px;background:${color};display:block;`;
      el.appendChild(dot);

      if (site.health === "critical") {
        el.animate(
          [
            { boxShadow: `0 0 0 0 ${color}88` },
            { boxShadow: `0 0 0 12px ${color}00` },
          ],
          { duration: 1800, iterations: Infinity },
        );
      }
      return el;
    },
    [activeSiteId],
  );

  // Rebuild markers whenever the fleet or selection changes.
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !ready) return;

    let cancelled = false;
    (async () => {
      const { Marker, Popup } = await import("maplibre-gl");
      if (cancelled) return;

      markersRef.current.forEach((m) => m.remove());
      markersRef.current = sites.map((site) => {
        const popup = new Popup({
          offset: 18,
          closeButton: false,
          maxWidth: "280px",
        }).setHTML(popupHtml(site));

        const marker = new Marker({ element: buildMarker(site) })
          .setLngLat([site.longitude, site.latitude])
          .setPopup(popup)
          .addTo(map);

        marker.getElement().addEventListener("click", () => onSelect?.(site.site_id));
        return marker;
      });
    })();

    return () => {
      cancelled = true;
    };
  }, [sites, ready, buildMarker, onSelect]);

  if (failed) {
    return (
      <div
        className="rounded-lg border border-slate-700 bg-slate-900/60 p-6 text-sm text-slate-400"
        style={{ height }}
      >
        <p className="font-medium text-slate-300">Map unavailable</p>
        <p className="mt-1 text-xs">
          The interactive map could not load ({failed}). Site coordinates and health
          are still listed below.
        </p>
      </div>
    );
  }

  return (
    <div className="relative overflow-hidden rounded-lg border border-slate-700">
      <div ref={containerRef} style={{ height }} className="w-full bg-slate-900" />
      {!ready && (
        <div className="absolute inset-0 flex items-center justify-center bg-slate-900/70 text-xs text-slate-500">
          Loading map…
        </div>
      )}
      <MapLegend />
    </div>
  );
}

function MapLegend() {
  return (
    <div className="pointer-events-none absolute bottom-2 left-2 rounded-md border border-slate-700 bg-slate-900/90 px-3 py-2 text-[11px] leading-relaxed text-slate-400 backdrop-blur-sm">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
        {(Object.keys(HEALTH_COLOR) as Health[]).map((health) => (
          <span key={health} className="flex items-center gap-1.5">
            <span
              className="inline-block h-2.5 w-2.5 rounded-full"
              style={{ background: HEALTH_COLOR[health] }}
            />
            {HEALTH_LABEL[health]}
          </span>
        ))}
      </div>
      <div className="mt-1 border-t border-slate-800 pt-1 text-slate-500">
        Solid ring = measured data · dashed ring = simulated · size ≈ capacity
      </div>
    </div>
  );
}

function escapeHtml(value: string): string {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function popupHtml(site: SiteStatusRecord): string {
  const color = HEALTH_COLOR[site.health];
  const ratio =
    site.expected_ratio !== null
      ? `${(site.expected_ratio * 100).toFixed(0)}% of expected`
      : "no expectation available";
  const mode =
    site.data_mode === "real"
      ? "Measured telemetry (NREL PVDAQ)"
      : "Simulated demonstration data";

  return `
    <div style="font-family:system-ui,-apple-system,sans-serif;min-width:210px">
      <div style="font-weight:600;font-size:13px;color:#0f172a;margin-bottom:2px">
        ${escapeHtml(site.name)}
      </div>
      <div style="font-size:11px;color:#475569;margin-bottom:6px">${escapeHtml(mode)}</div>
      <div style="display:flex;align-items:center;gap:6px;font-size:12px;color:#0f172a">
        <span style="width:8px;height:8px;border-radius:9999px;background:${color};display:inline-block"></span>
        <strong>${HEALTH_LABEL[site.health]}</strong>
        <span style="color:#64748b">· ${site.capacity_kw} kW</span>
      </div>
      <div style="font-size:11px;color:#475569;margin-top:6px;line-height:1.5">
        Running at ${escapeHtml(ratio)}<br/>
        ${site.lost_energy_kwh.toFixed(1)} kWh estimated shortfall<br/>
        ${site.active_events} event${site.active_events === 1 ? "" : "s"} in window
      </div>
      <a href="/sites/${encodeURIComponent(site.site_id)}"
         style="display:inline-block;margin-top:8px;font-size:11px;color:#0891b2;text-decoration:underline">
        Open site analytics →
      </a>
    </div>
  `;
}

/** Compact list beneath the map, so the fleet is usable without the map. */
export function FleetSiteList({
  sites,
  activeSiteId,
}: {
  sites: SiteStatusRecord[];
  activeSiteId?: string;
}) {
  return (
    <ul className="mt-3 grid grid-cols-1 gap-1.5 sm:grid-cols-2">
      {sites.map((site) => (
        <li key={site.site_id}>
          <Link
            href={`/sites/${site.site_id}`}
            className={`flex items-center gap-2 rounded px-2 py-1.5 text-xs transition-colors hover:bg-slate-800/60 ${
              site.site_id === activeSiteId ? "bg-slate-800/60" : ""
            }`}
          >
            <span
              className="h-2.5 w-2.5 shrink-0 rounded-full"
              style={{ background: HEALTH_COLOR[site.health] }}
            />
            <span className="truncate text-slate-300">{site.name}</span>
            <span className="ml-auto shrink-0 text-slate-600">
              {site.capacity_kw} kW
            </span>
            <span
              className={`shrink-0 rounded px-1 text-[10px] ${
                site.data_mode === "real"
                  ? "bg-emerald-950 text-emerald-400"
                  : "bg-amber-950 text-amber-400"
              }`}
            >
              {site.data_mode === "real" ? "measured" : "simulated"}
            </span>
          </Link>
        </li>
      ))}
    </ul>
  );
}
