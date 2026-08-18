/**
 * GridGuard API client.
 *
 * Every response type carries `data_mode`. That is deliberate: it makes it
 * impossible to render generation numbers without knowing whether they are
 * measured telemetry or a simulation.
 */

const BASE =
  process.env.NEXT_PUBLIC_API_URL ??
  process.env.NEXT_PUBLIC_API_BASE_URL ??
  "http://localhost:8000";

export type DataMode = "real" | "synthetic";
export type Severity = "low" | "medium" | "high";
export type Health = "healthy" | "warning" | "critical" | "no_data";
export type Scope = "site_specific" | "regional" | "indeterminate";

// ---------------------------------------------------------------------------
// Types (mirror the FastAPI Pydantic schemas)
// ---------------------------------------------------------------------------

export interface HealthResponse {
  status: string;
  version: string;
  sites_loaded: number;
  sites_with_models: number;
  sites_with_calibration: number;
  data_rows: number;
  manifest_present: boolean;
  artifact_built_at: string | null;
  artifact_commit: string | null;
  detection_method: string;
  warnings: string[];
}

export interface SiteRecord {
  site_id: string;
  name: string;
  region: string;
  data_mode: DataMode;
  latitude: number;
  longitude: number;
  elevation_m: number;
  capacity_kw: number;
  capacity_basis: string;
  tilt_deg: number | null;
  azimuth_deg: number | null;
  source_system_id: number | null;
  disclaimer: string;
  notes: string;
}

export interface SitesResponse {
  sites: SiteRecord[];
  total: number;
  real_count: number;
  synthetic_count: number;
}

export interface TimeseriesPoint {
  timestamp: string;
  actual_kw: number;
  predicted_kw: number;
  expected_lower_kw: number | null;
  expected_upper_kw: number | null;
  irradiance_wm2: number | null;
  temperature_c: number | null;
  is_anomaly: boolean;
}

export interface TimeseriesResponse {
  site_id: string;
  data_mode: DataMode;
  disclaimer: string;
  interval_minutes: number;
  points: TimeseriesPoint[];
  total_points: number;
  truncated: boolean;
}

export interface SiteStatusRecord {
  site_id: string;
  name: string;
  data_mode: DataMode;
  latitude: number;
  longitude: number;
  capacity_kw: number;
  health: Health;
  actual_kwh: number;
  expected_kwh: number;
  expected_ratio: number | null;
  lost_energy_kwh: number;
  loss_fraction: number | null;
  anomaly_intervals: number;
  assessable_intervals: number;
  active_events: number;
  max_severity: string;
  latest_timestamp: string | null;
  latest_actual_kw: number | null;
  latest_expected_kw: number | null;
  latest_expected_lower_kw: number | null;
  anomaly_scope: Scope | null;
  scope_confidence: string | null;
  scope_explanation: string | null;
  disclaimer: string;
}

export interface FleetBounds {
  min_latitude: number;
  max_latitude: number;
  min_longitude: number;
  max_longitude: number;
  center_latitude: number;
  center_longitude: number;
}

export interface FleetSummaryResponse {
  total_sites: number;
  total_capacity_kw: number;
  sites_healthy: number;
  sites_warning: number;
  sites_critical: number;
  sites_no_data: number;
  total_actual_kwh: number;
  total_expected_kwh: number;
  total_lost_kwh: number;
  fleet_expected_ratio: number | null;
  active_events: number;
  real_sites: number;
  synthetic_sites: number;
  window_start: string | null;
  window_end: string | null;
  sites: SiteStatusRecord[];
  bounds: FleetBounds | null;
}

export interface SpatialContextRecord {
  scope: Scope;
  confidence: string;
  explanation: string;
  site_normalised_residual: number;
  neighbor_count: number;
  neighbors_considered: string[];
  neighbor_residual_median: number | null;
  distance_weighted_residual: number | null;
  neighbors_affected: number;
  regional_anomaly_score: number;
  excess_deviation: number | null;
  radius_km: number;
}

export interface EventRecord {
  event_id: number;
  global_event_id: string | null;
  site_id: string | null;
  site_name: string | null;
  data_mode: DataMode | null;
  start_time: string;
  end_time: string;
  duration_minutes: number;
  interval_count: number;
  total_lost_kwh: number;
  max_residual_sigma: number | null;
  mean_actual_kw: number;
  mean_predicted_kw: number;
  mean_expected_lower_kw: number | null;
  severity: Severity;
  explanation: string;
  disclaimer: string | null;
}

export interface EventsResponse {
  total_events: number;
  total_lost_kwh: number;
  events: EventRecord[];
}

export interface ExplainContributor {
  feature: string;
  feature_value: number;
  shap_value: number;
  direction: "positive" | "negative";
}

export interface NeighborComparison {
  site_id: string;
  name: string;
  capacity_kw: number;
  actual_kw_mean: number;
  expected_kw_mean: number;
  expected_ratio: number | null;
  anomaly_intervals: number;
}

export interface EventDetailResponse {
  event: EventRecord;
  spatial_context: SpatialContextRecord | null;
  neighbor_comparison: NeighborComparison[];
  timeseries: TimeseriesPoint[];
  recommended_actions: string[];
  contributors: ExplainContributor[];
}

export interface AnomalyRecord {
  timestamp: string;
  site_id: string | null;
  actual_kw: number;
  predicted_kw: number;
  expected_lower_kw: number | null;
  residual_kw: number;
  residual_sigma: number | null;
  lost_energy_kwh: number;
  is_anomaly: boolean;
}

export interface AnomalyResponse {
  total_anomalies: number;
  total_lost_kwh: number;
  detection_method: string;
  records: AnomalyRecord[];
}

export interface ModelMetrics {
  model: string;
  mae_kw: number;
  rmse_kw: number;
  mape_pct: number | null;
  r2: number;
  n_samples: number | null;
}

export interface MetricsResponse {
  site_id: string;
  data_mode: DataMode;
  best_model: string;
  models: ModelMetrics[];
  weather_only_models: ModelMetrics[];
  evaluation_note: string;
  detection_metrics: Record<string, unknown> | null;
  conformal_coverage: Record<string, number> | null;
  train_test_split_date: string | null;
  data_start: string | null;
  data_end: string | null;
}

export interface ProvenanceRecord {
  site_id: string;
  data_mode: DataMode;
  dataset: string;
  system_identifier: string;
  site_name: string;
  latitude: number | null;
  longitude: number | null;
  elevation_m: number | null;
  capacity_kw: number | null;
  capacity_basis: string;
  tilt_deg: number | null;
  azimuth_deg: number | null;
  interval_minutes: number | null;
  native_interval_minutes: number | null;
  start: string;
  end: string;
  timezone_note: string;
  row_count: number | null;
  weather_source: string;
  irradiance_kind: string;
  irradiance_channel: string;
  irradiance_scale_factor: number | null;
  irradiance_scale_basis: string;
  source_url: string;
  license_note: string;
  retrieved_at: string;
  processing_notes: string[];
  known_limitations: string[];
}

export interface DataProvenanceResponse {
  real_datasets: ProvenanceRecord[];
  synthetic_datasets: ProvenanceRecord[];
  real_data_note: string;
  synthetic_data_note: string;
}

export interface MethodologyResponse {
  detection_method: string;
  conformal_alpha: number;
  conformal_guarantee: string;
  conformal_limitations: string[];
  feature_modes: Record<
    string,
    { purpose: string; features: string[]; rationale: string }
  >;
  fault_taxonomy: {
    fault_type: string;
    description: string;
    is_generation_loss: boolean;
  }[];
  spatial_method: Record<string, unknown>;
  physics_model: Record<string, unknown> | null;
  artifact: Record<string, unknown> | null;
}

export interface DemoScenarioResponse {
  data_label: string;
  data_mode: DataMode;
  disclaimer: string;
  demo_date: string;
  site_id: string;
  site_name: string;
  event: EventRecord | null;
  spatial_context: SpatialContextRecord | null;
  anomaly_records: AnomalyRecord[];
  jump_to_event_id: string | null;
  recommended_actions: string[];
  real_data_site_id: string | null;
  real_data_note: string;
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

export class ApiError extends Error {
  constructor(
    public readonly path: string,
    public readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

type Params = Record<string, string | number | boolean | undefined | null>;

async function apiFetch<T>(path: string, params?: Params): Promise<T> {
  const url = new URL(`${BASE}${path}`);
  if (params) {
    for (const [key, value] of Object.entries(params)) {
      if (value !== undefined && value !== null) {
        url.searchParams.set(key, String(value));
      }
    }
  }

  let res: Response;
  try {
    res = await fetch(url.toString(), { next: { revalidate: 60 } });
  } catch (cause) {
    // A connection failure almost always means the backend is not running, so
    // say that rather than surfacing a bare "fetch failed".
    throw new ApiError(
      path,
      0,
      `Could not reach the GridGuard API at ${BASE}. Is the backend running? (${String(cause)})`,
    );
  }

  if (!res.ok) {
    const body = await res.text().catch(() => "");
    let detail = body;
    try {
      detail = (JSON.parse(body) as { detail?: string }).detail ?? body;
    } catch {
      /* body was not JSON; use it as-is */
    }
    throw new ApiError(path, res.status, detail || res.statusText);
  }

  return (await res.json()) as T;
}

export const api = {
  health: () => apiFetch<HealthResponse>("/health"),

  sites: (params?: { data_mode?: DataMode }) =>
    apiFetch<SitesResponse>("/sites", params),
  site: (siteId: string) => apiFetch<SiteRecord>(`/sites/${siteId}`),
  timeseries: (
    siteId: string,
    params?: { start?: string; end?: string; limit?: number },
  ) => apiFetch<TimeseriesResponse>(`/sites/${siteId}/timeseries`, params),

  fleetSummary: (params?: { data_mode?: DataMode; start?: string; end?: string }) =>
    apiFetch<FleetSummaryResponse>("/fleet/summary", params),

  events: (params?: {
    site_id?: string;
    data_mode?: DataMode;
    severity?: Severity;
    start?: string;
    end?: string;
    limit?: number;
  }) => apiFetch<EventsResponse>("/events", params),
  event: (eventId: string) =>
    apiFetch<EventDetailResponse>(`/events/${encodeURIComponent(eventId)}`),

  anomalies: (params?: {
    site_id?: string;
    start?: string;
    end?: string;
    only_anomalies?: boolean;
    limit?: number;
  }) => apiFetch<AnomalyResponse>("/anomalies", params),

  metrics: (siteId?: string) =>
    apiFetch<MetricsResponse>("/metrics", { site_id: siteId }),
  methodology: () => apiFetch<MethodologyResponse>("/methodology"),
  dataProvenance: () => apiFetch<DataProvenanceResponse>("/data"),
  demoScenario: () => apiFetch<DemoScenarioResponse>("/demo/scenario"),
};

/** Run a request and return `null` instead of throwing, with the error text. */
export async function tryFetch<T>(
  fn: () => Promise<T>,
): Promise<{ data: T; error: null } | { data: null; error: string }> {
  try {
    return { data: await fn(), error: null };
  } catch (e) {
    return { data: null, error: e instanceof Error ? e.message : String(e) };
  }
}
