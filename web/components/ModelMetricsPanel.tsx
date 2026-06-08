"use client";

import { MetricsResponse } from "@/lib/api";
import { Card, CardTitle } from "@/components/ui/Card";

interface Props {
  metrics: MetricsResponse;
}

export function ModelMetricsPanel({ metrics }: Props) {
  return (
    <Card>
      <CardTitle>Model Performance (test set)</CardTitle>
      <div className="overflow-x-auto">
        <table className="w-full text-xs text-left">
          <thead>
            <tr className="border-b border-slate-700">
              {["Model", "MAE (kW)", "RMSE (kW)", "MAPE (%)", "R²"].map((h) => (
                <th key={h} className="py-1.5 pr-4 text-slate-500 font-medium">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {metrics.models.map((m) => (
              <tr key={m.model} className={`border-b border-slate-800 ${m.model === metrics.best_model ? "text-cyan-300" : "text-slate-400"}`}>
                <td className="py-1.5 pr-4 font-medium">
                  {m.model}
                  {m.model === metrics.best_model && (
                    <span className="ml-1.5 text-xs text-cyan-600">best</span>
                  )}
                </td>
                <td className="py-1.5 pr-4 font-mono">{m.mae_kw.toFixed(3)}</td>
                <td className="py-1.5 pr-4 font-mono">{m.rmse_kw.toFixed(3)}</td>
                <td className="py-1.5 pr-4 font-mono">{m.mape_pct.toFixed(1)}</td>
                <td className="py-1.5 font-mono">{m.r2.toFixed(4)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}
