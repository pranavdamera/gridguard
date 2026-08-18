"use client";

/**
 * Actual generation against the model's expected range.
 *
 * The shaded band is the calibrated conformal interval, not a cosmetic error
 * bar: at the configured miscoverage level, healthy intervals are expected to
 * fall inside it. That is what makes "below the band" a statement with a
 * false-alarm rate attached, rather than a judgement call about how far below a
 * line is far enough.
 */

import {
  Area,
  ComposedChart,
  Line,
  ReferenceArea,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { TimeseriesPoint } from "@/lib/api";

interface Props {
  points: TimeseriesPoint[];
  height?: number;
  highlightStart?: string;
  highlightEnd?: string;
}

interface Row {
  t: string;
  label: string;
  actual: number;
  expected: number;
  lower: number | null;
  bandBase: number | null;
  bandHeight: number | null;
  anomaly: boolean;
}

function toRows(points: TimeseriesPoint[]): Row[] {
  return points.map((p) => {
    const lower = p.expected_lower_kw;
    const upper = p.expected_upper_kw;
    // Recharts stacks an invisible base plus a visible height to draw a band
    // between two arbitrary series.
    const bandBase = lower ?? null;
    const bandHeight = lower !== null && upper !== null ? Math.max(0, upper - lower) : null;
    return {
      t: p.timestamp,
      label: p.timestamp.slice(5, 16).replace("T", " "),
      actual: p.actual_kw,
      expected: p.predicted_kw,
      lower,
      bandBase,
      bandHeight,
      anomaly: p.is_anomaly,
    };
  });
}

export function GenerationChart({
  points,
  height = 300,
  highlightStart,
  highlightEnd,
}: Props) {
  const rows = toRows(points);
  if (!rows.length) {
    return (
      <div className="flex items-center justify-center rounded border border-dashed border-slate-800 text-sm text-slate-600"
           style={{ height }}>
        No telemetry in this window
      </div>
    );
  }

  const highlightFrom = highlightStart
    ? rows.find((r) => r.t >= highlightStart)?.label
    : undefined;
  const highlightTo = highlightEnd
    ? [...rows].reverse().find((r) => r.t <= highlightEnd)?.label
    : undefined;

  return (
    <ResponsiveContainer width="100%" height={height}>
      <ComposedChart data={rows} margin={{ top: 8, right: 8, bottom: 4, left: -8 }}>
        <XAxis
          dataKey="label"
          tick={{ fill: "#64748b", fontSize: 10 }}
          stroke="#1e293b"
          minTickGap={48}
        />
        <YAxis
          tick={{ fill: "#64748b", fontSize: 10 }}
          stroke="#1e293b"
          width={48}
          label={{
            value: "kW",
            angle: -90,
            position: "insideLeft",
            fill: "#64748b",
            fontSize: 10,
            offset: 16,
          }}
        />

        {highlightFrom && highlightTo && (
          <ReferenceArea
            x1={highlightFrom}
            x2={highlightTo}
            fill="#ef4444"
            fillOpacity={0.08}
            stroke="#ef4444"
            strokeOpacity={0.25}
          />
        )}

        {/* Calibrated expected range, drawn as base + height. */}
        <Area
          dataKey="bandBase"
          stackId="band"
          stroke="none"
          fill="transparent"
          isAnimationActive={false}
        />
        <Area
          dataKey="bandHeight"
          stackId="band"
          stroke="none"
          fill="#0891b2"
          fillOpacity={0.16}
          isAnimationActive={false}
          name="Calibrated expected range"
        />

        <Line
          dataKey="expected"
          stroke="#22d3ee"
          strokeWidth={1.5}
          strokeDasharray="4 3"
          dot={false}
          isAnimationActive={false}
          name="Expected"
        />
        <Line
          dataKey="actual"
          stroke="#e2e8f0"
          strokeWidth={1.8}
          dot={false}
          isAnimationActive={false}
          name="Actual"
        />

        <Tooltip
          contentStyle={{
            background: "#0f172a",
            border: "1px solid #1e293b",
            borderRadius: 6,
            fontSize: 12,
          }}
          labelStyle={{ color: "#94a3b8", fontSize: 11 }}
          formatter={(value, name) => {
            const key = String(name ?? "");
            // The band is drawn as an invisible base plus a visible height, so
            // the base series is suppressed from the tooltip.
            if (key === "bandBase") return ["", ""];
            const numeric = typeof value === "number" ? `${value.toFixed(1)} kW` : "—";
            if (key === "bandHeight") return [numeric, "Range width"];
            return [numeric, key];
          }}
        />
      </ComposedChart>
    </ResponsiveContainer>
  );
}

export function ChartLegend() {
  return (
    <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-slate-500">
      <span className="flex items-center gap-1.5">
        <span className="inline-block h-0.5 w-4 bg-slate-200" /> Actual
      </span>
      <span className="flex items-center gap-1.5">
        <span
          className="inline-block h-0.5 w-4"
          style={{
            backgroundImage:
              "repeating-linear-gradient(90deg,#22d3ee 0 4px,transparent 4px 7px)",
          }}
        />
        Expected
      </span>
      <span className="flex items-center gap-1.5">
        <span className="inline-block h-2.5 w-4 rounded-sm bg-cyan-700/30" />
        Calibrated expected range
      </span>
    </div>
  );
}
