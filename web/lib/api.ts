/**
 * GridGuard API client.
 * All calls go through NEXT_PUBLIC_API_BASE_URL — never hardcoded localhost.
 */

const BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

// ---------------------------------------------------------------------------
// Types (mirrors FastAPI Pydantic schemas)
// ---------------------------------------------------------------------------

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  data_rows: number;
  version: string;
  residual_calibration_loaded: boolean;
}

export interface SiteRecord {
  site_id: string;
  name: string;
  region: string;
  latitude: number;
  longitude: number;
  capacity_kw: number;
  notes: string;
}

export interface SitesResponse {
  sites: SiteRecord[];
  total: number;
}

export interface EventRecord {
  event_id: number;
  start_time: string;
  end_time: string;
  duration_minutes: number;
  interval_count: number;
  total_lost_kwh: number;
  max_residual_sigma: number;
  mean_actual_kw: number;
  mean_predicted_kw: number;
  severity: "low" | "medium" | "high";
  explanation: string;
  site_id: string | null;
}

export interface EventsResponse {
  total_events: number;
  total_lost_kwh: number;
  events: EventRecord[];
}

export interface AnomalyRecord {
  timestamp: string;
  actual_kw: number;
  predicted_kw: number;
  residual_kw: number;
  residual_sigma: number;
  lost_energy_kwh: number;
  is_anomaly: boolean;
}

export interface AnomalyResponse {
  total_anomalies: number;
  total_lost_kwh: number;
  records: AnomalyRecord[];
}

export interface ModelMetrics {
  model: string;
  mae_kw: number;
  rmse_kw: number;
  mape_pct: number;
  r2: number;
}

export interface MetricsResponse {
  models: ModelMetrics[];
  best_model: string;
}

export interface ExplainContributor {
  feature: string;
  feature_value: number;
  shap_value: number;
  direction: "positive" | "negative";
}

export interface ExplainResponse {
  timestamp: string;
  predicted_kw: number;
  contributors: ExplainContributor[];
  model_name: string;
  shap_available: boolean;
}

export interface DemoScenarioResponse {
  data_label: string;
  demo_date: string;
  site_id: string;
  site_name: string;
  event: EventRecord | null;
  anomaly_records: AnomalyRecord[];
  jump_to_event_id: number | null;
  recommended_actions: string[];
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

async function apiFetch<T>(path: string, params?: Record<string, string | number | boolean>): Promise<T> {
  const url = new URL(`${BASE}${path}`);
  if (params) {
    Object.entries(params).forEach(([k, v]) => {
      if (v !== undefined && v !== null) url.searchParams.set(k, String(v));
    });
  }
  const res = await fetch(url.toString(), { next: { revalidate: 30 } });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`API ${path} returned ${res.status}: ${text}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => apiFetch<HealthResponse>("/health"),
  sites: () => apiFetch<SitesResponse>("/sites"),
  site: (siteId: string) => apiFetch<SiteRecord>(`/sites/${siteId}`),
  events: (params?: { site_id?: string; start?: string; end?: string; limit?: number; severity?: string }) =>
    apiFetch<EventsResponse>("/events", params as Record<string, string | number>),
  event: (eventId: number) => apiFetch<EventRecord>(`/events/${eventId}`),
  anomalies: (params?: { site_id?: string; start?: string; end?: string; limit?: number; only_anomalies?: boolean }) =>
    apiFetch<AnomalyResponse>("/anomalies", params as Record<string, string | number | boolean>),
  forecast: (params?: { site_id?: string; start?: string; end?: string; limit?: number }) =>
    apiFetch<{ timestamp: string; predicted_kw: number; model_name: string }[]>("/forecast", params as Record<string, string | number>),
  metrics: () => apiFetch<MetricsResponse>("/metrics"),
  explain: (timestamp: string) => apiFetch<ExplainResponse>("/explain", { timestamp }),
  demoScenario: () => apiFetch<DemoScenarioResponse>("/demo/scenario"),
};
