"use client";

/**
 * Client wrapper around the map, so the server component that renders the fleet
 * page stays a server component. Holds only the selection state.
 */

import { useState } from "react";
import type { FleetBounds, SiteStatusRecord } from "@/lib/api";
import { FleetMap, FleetSiteList } from "@/components/FleetMap";
import { Card, CardTitle } from "@/components/ui/Primitives";

export function FleetMapPanel({
  sites,
  bounds,
}: {
  sites: SiteStatusRecord[];
  bounds: FleetBounds | null;
}) {
  const [selected, setSelected] = useState<string | undefined>(undefined);

  return (
    <Card>
      <CardTitle hint="OpenStreetMap via CARTO">
        Fleet map — DMV region
      </CardTitle>
      <FleetMap
        sites={sites}
        bounds={bounds}
        activeSiteId={selected}
        onSelect={setSelected}
        height={420}
      />
      <FleetSiteList sites={sites} activeSiteId={selected} />
    </Card>
  );
}
