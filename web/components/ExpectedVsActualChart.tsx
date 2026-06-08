"use client";

import {
  ResponsiveContainer,
  ComposedChart,
  Area,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from "recharts";
import { AnomalyRecord } from "@/lib/api";
import { Card, CardTitle } from "@/components/ui/Card";

interface Props {
  records: AnomalyRecord[];
  title?: string;
}

function fmtTime(ts: string) {
  const d = new Date(ts);
  return `${d.getHours().toString().padStart(2, "0")}:${d.getMinutes().toString().padStart(2, "0")}`;
}

const CustomTooltip = ({ active, payload, label }: {active?: boolean; payload?: {color: string; name: string; value: number}[]; label?: string}) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-slate-900 border border-slate-600 rounded p-3 text-xs">
      <div className="text-slate-400 mb-1.5">{label}</div>
      {payload.map((p, i) => (
        <div key={i} style={{ color: p.color }} className="flex justify-between gap-4">
          <span>{p.name}</span>
          <span className="font-mono">{p.value?.toFixed(2)} kW</span>
        </div>
      ))}
    </div>
  );
};

export function ExpectedVsActualChart({ records, title = "Expected vs Actual Generation" }: Props) {
  const data = records.map((r) => ({
    time: fmtTime(r.timestamp),
    actual: r.actual_kw,
    expected: r.predicted_kw,
    anomaly: r.is_anomaly ? r.actual_kw : null,
  }));

  return (
    <Card>
      <CardTitle>{title}</CardTitle>
      <ResponsiveContainer width="100%" height={260}>
        <ComposedChart data={data} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
          <XAxis
            dataKey="time"
            tick={{ fill: "#94a3b8", fontSize: 11 }}
            tickLine={false}
            interval="preserveStartEnd"
          />
          <YAxis
            tick={{ fill: "#94a3b8", fontSize: 11 }}
            tickLine={false}
            axisLine={false}
            unit=" kW"
            width={56}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend wrapperStyle={{ fontSize: 12, color: "#94a3b8" }} />
          <Area
            type="monotone"
            dataKey="expected"
            name="Expected"
            stroke="#06b6d4"
            fill="#06b6d4"
            fillOpacity={0.08}
            strokeWidth={1.5}
            dot={false}
          />
          <Line
            type="monotone"
            dataKey="actual"
            name="Actual"
            stroke="#e2e8f0"
            strokeWidth={1.5}
            dot={false}
          />
          <Line
            type="monotone"
            dataKey="anomaly"
            name="Anomaly"
            stroke="#f87171"
            strokeWidth={0}
            dot={{ fill: "#f87171", r: 3 }}
            connectNulls={false}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </Card>
  );
}
