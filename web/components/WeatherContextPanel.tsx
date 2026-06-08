"use client";

import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
} from "recharts";
import { AnomalyRecord } from "@/lib/api";
import { Card, CardTitle } from "@/components/ui/Card";

interface Props {
  records: AnomalyRecord[];
}

export function WeatherContextPanel({ records }: Props) {
  // We only have irradiance indirectly from predicted_kw, but let's show residual sigma as weather proxy
  const data = records.map((r) => ({
    time: new Date(r.timestamp).toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", hour12: false }),
    sigma: Math.abs(r.residual_sigma),
    lost: r.lost_energy_kwh * 4, // kW equivalent
  }));

  return (
    <Card>
      <CardTitle>Anomaly Signal — Residual Sigma</CardTitle>
      <ResponsiveContainer width="100%" height={160}>
        <AreaChart data={data} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
          <XAxis dataKey="time" tick={{ fill: "#94a3b8", fontSize: 10 }} tickLine={false} interval="preserveStartEnd" />
          <YAxis tick={{ fill: "#94a3b8", fontSize: 10 }} tickLine={false} axisLine={false} width={32} />
          <Tooltip
            contentStyle={{ background: "#1e293b", border: "1px solid #334155", borderRadius: 6, fontSize: 12 }}
            labelStyle={{ color: "#94a3b8" }}
          />
          <Area
            type="monotone"
            dataKey="sigma"
            name="|residual σ|"
            stroke="#f59e0b"
            fill="#f59e0b"
            fillOpacity={0.15}
            strokeWidth={1.5}
            dot={false}
          />
        </AreaChart>
      </ResponsiveContainer>
      <div className="text-xs text-slate-600 mt-1">
        Residual σ = (actual − expected) / per-hour training std. Values {">"} 2σ trigger anomaly flag.
      </div>
    </Card>
  );
}
