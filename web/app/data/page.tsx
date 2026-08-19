import { api, tryFetch } from "@/lib/api";
import type { ProvenanceRecord } from "@/lib/api";
import {
  Card,
  CardTitle,
  DataModeBadge,
  ErrorPanel,
} from "@/components/ui/Primitives";

export const revalidate = 300;

export const metadata = {
  title: "Data provenance — GridGuard",
  description:
    "Exactly which GridGuard numbers are measured photovoltaic telemetry and which are simulated, with sources, instruments, processing steps and known limitations.",
};

export default async function DataPage() {
  const { data, error } = await tryFetch(() => api.dataProvenance());

  if (error || !data) {
    return (
      <div className="mx-auto max-w-4xl px-4 py-8">
        <ErrorPanel error={error ?? "No provenance available"} />
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-4xl px-4 py-8">
      <header className="mb-6">
        <h1 className="text-xl font-semibold tracking-tight text-slate-100">
          Data provenance
        </h1>
        <p className="mt-1.5 max-w-2xl text-sm leading-relaxed text-slate-400">
          GridGuard deliberately mixes measured telemetry with simulated
          demonstration data, because each answers a question the other cannot.
          This page states which is which, down to the instrument.
        </p>
      </header>

      <div className="grid gap-4 sm:grid-cols-2">
        <Card className="border-emerald-900/50">
          <DataModeBadge mode="real" />
          <p className="mt-2 text-xs leading-relaxed text-slate-400">
            <strong className="text-slate-300">
              Used for forecast accuracy.
            </strong>{" "}
            Real arrays, real weather, real sensor noise. The target — actual
            generation — is genuine, so error metrics on it describe real
            predictive skill.
          </p>
        </Card>
        <Card className="border-amber-900/50">
          <DataModeBadge mode="synthetic" />
          <p className="mt-2 text-xs leading-relaxed text-slate-400">
            <strong className="text-slate-300">
              Used for detection accuracy.
            </strong>{" "}
            Measured telemetry carries no annotated faults, so ground truth has to
            be constructed. Faults are injected with known labels — including into
            the measured data — and the detector is scored against them.
          </p>
        </Card>
      </div>

      <section className="mt-8">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-400">
          Measured datasets
        </h2>
        <p className="mt-1 text-xs leading-relaxed text-slate-500">
          {data.real_data_note}
        </p>
        <div className="mt-3 space-y-4">
          {data.real_datasets.map((record) => (
            <ProvenanceCard key={record.site_id} record={record} />
          ))}
          {data.real_datasets.length === 0 && (
            <p className="text-xs text-slate-600">No measured datasets loaded.</p>
          )}
        </div>
      </section>

      <section className="mt-8">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-400">
          Simulated datasets
        </h2>
        <p className="mt-1 rounded border border-amber-900/50 bg-amber-950/20 px-3 py-2 text-xs leading-relaxed text-amber-200/80">
          {data.synthetic_data_note}
        </p>
        <div className="mt-3 space-y-4">
          {data.synthetic_datasets.map((record) => (
            <ProvenanceCard key={record.site_id} record={record} />
          ))}
        </div>
      </section>

      <section className="mt-8">
        <Card>
          <CardTitle>Why the legacy PVDAQ API is not used</CardTitle>
          <p className="text-xs leading-relaxed text-slate-400">
            Earlier versions of this project pulled from{" "}
            <code className="rounded bg-slate-800 px-1 text-[11px]">
              developer.nrel.gov/api/pvdaq/v3
            </code>
            , which has since been decommissioned. The underlying PVDAQ data
            remains public and is distributed through the Open Energy Data
            Initiative data lake, which GridGuard reads over anonymous HTTPS.
            Because the arrays used here publish their own irradiance,
            temperature and wind, no NSRDB request — and therefore no API key —
            is needed anywhere in the shipped pipeline.
          </p>
        </Card>
      </section>
    </div>
  );
}

function ProvenanceCard({ record }: { record: ProvenanceRecord }) {
  return (
    <Card>
      <div className="flex flex-wrap items-center gap-2">
        <h3 className="text-sm font-semibold text-slate-200">
          {record.site_name || record.site_id}
        </h3>
        <DataModeBadge mode={record.data_mode} />
        {record.system_identifier && (
          <span className="font-mono text-[10px] text-slate-500">
            {record.system_identifier}
          </span>
        )}
      </div>

      <p className="mt-1 text-xs text-slate-400">{record.dataset}</p>

      <dl className="mt-3 grid grid-cols-2 gap-x-4 gap-y-1.5 text-[11px] sm:grid-cols-3">
        <Field label="Period" value={`${record.start.slice(0, 10)} → ${record.end.slice(0, 10)}`} />
        <Field label="Interval" value={`${record.interval_minutes} min`} />
        <Field label="Rows" value={record.row_count?.toLocaleString() ?? "—"} />
        <Field
          label="Capacity"
          value={record.capacity_kw ? `${record.capacity_kw} kW` : "—"}
          hint={record.capacity_basis}
        />
        <Field
          label="Orientation"
          value={
            record.tilt_deg !== null
              ? `tilt ${record.tilt_deg}° / az ${record.azimuth_deg}°`
              : "—"
          }
        />
        <Field
          label="Coordinates"
          value={
            record.latitude !== null
              ? `${record.latitude.toFixed(4)}, ${record.longitude?.toFixed(4)}`
              : "—"
          }
        />
      </dl>

      <div className="mt-3 space-y-2 text-[11px] leading-relaxed">
        <Detail label="Weather source" value={record.weather_source} />
        <Detail
          label="Irradiance"
          value={`${record.irradiance_kind}${
            record.irradiance_channel ? ` — ${record.irradiance_channel}` : ""
          }`}
        />
        {record.irradiance_scale_basis && (
          <Detail label="Units" value={record.irradiance_scale_basis} />
        )}
        <Detail label="Timezone" value={record.timezone_note} />
        {record.license_note && <Detail label="Licence" value={record.license_note} />}
        {record.source_url && (
          <div>
            <span className="text-slate-500">Source: </span>
            <a
              href={record.source_url}
              target="_blank"
              rel="noopener noreferrer"
              className="text-cyan-400 underline hover:text-cyan-300"
            >
              {record.source_url}
            </a>
          </div>
        )}
      </div>

      {record.processing_notes.length > 0 && (
        <details className="mt-3 text-[11px]">
          <summary className="cursor-pointer text-slate-500 hover:text-slate-400">
            Processing steps ({record.processing_notes.length})
          </summary>
          <ul className="mt-1.5 space-y-1 pl-4 text-slate-400">
            {record.processing_notes.map((note, i) => (
              <li key={i} className="list-disc leading-relaxed">
                {note}
              </li>
            ))}
          </ul>
        </details>
      )}

      {record.known_limitations.length > 0 && (
        <div className="mt-3 rounded border border-slate-800 bg-slate-950/40 px-3 py-2">
          <p className="text-[10px] font-semibold uppercase tracking-wide text-slate-500">
            Known limitations
          </p>
          <ul className="mt-1 space-y-1 pl-4 text-[11px] leading-relaxed text-slate-400">
            {record.known_limitations.map((limit, i) => (
              <li key={i} className="list-disc">
                {limit}
              </li>
            ))}
          </ul>
        </div>
      )}
    </Card>
  );
}

function Field({
  label,
  value,
  hint,
}: {
  label: string;
  value: string;
  hint?: string;
}) {
  return (
    <div>
      <dt className="text-slate-500">{label}</dt>
      <dd className="text-slate-300">{value}</dd>
      {hint && <dd className="text-[10px] text-slate-600">{hint}</dd>}
    </div>
  );
}

function Detail({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <span className="text-slate-500">{label}: </span>
      <span className="text-slate-400">{value}</span>
    </div>
  );
}
