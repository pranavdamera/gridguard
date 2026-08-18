import { api, tryFetch } from "@/lib/api";
import { Card, CardTitle, ErrorPanel } from "@/components/ui/Primitives";

export const revalidate = 300;

export const metadata = {
  title: "Methodology — GridGuard",
  description:
    "How GridGuard forecasts expected solar generation, calibrates uncertainty with conformal prediction, evaluates detection against injected faults, and attributes deviation spatially.",
};

export default async function MethodologyPage() {
  const { data, error } = await tryFetch(() => api.methodology());

  if (error || !data) {
    return (
      <div className="mx-auto max-w-4xl px-4 py-8">
        <ErrorPanel error={error ?? "No methodology available"} />
      </div>
    );
  }

  const artifact = data.artifact as
    | {
        built_at?: string;
        git_commit?: string;
        python_version?: string;
        package_versions?: Record<string, string>;
        sites_built?: string[];
      }
    | null;

  const spatial = data.spatial_method as Record<string, string | number | string[]>;

  return (
    <div className="mx-auto max-w-4xl px-4 py-8">
      <header className="mb-6">
        <h1 className="text-xl font-semibold tracking-tight text-slate-100">
          Methodology
        </h1>
        <p className="mt-1.5 max-w-2xl text-sm leading-relaxed text-slate-400">
          Everything on this page is read from the artifact manifest produced by
          the build that created the models currently being served — not written
          by hand, so it cannot drift away from what the models actually do.
        </p>
      </header>

      {/* ---- Feature modes ---- */}
      <section>
        <Card>
          <CardTitle>Two feature sets, and why the split matters</CardTitle>
          <div className="space-y-4">
            {Object.entries(data.feature_modes).map(([mode, detail]) => (
              <div key={mode}>
                <h3 className="font-mono text-xs text-cyan-400">{mode}</h3>
                <p className="mt-1 text-xs text-slate-300">{detail.purpose}</p>
                <p className="mt-1 text-xs leading-relaxed text-slate-500">
                  {detail.rationale}
                </p>
                <div className="mt-1.5 flex flex-wrap gap-1">
                  {detail.features.map((f) => (
                    <span
                      key={f}
                      className="rounded bg-slate-800 px-1.5 py-0.5 font-mono text-[10px] text-slate-400"
                    >
                      {f}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </Card>
      </section>

      {/* ---- Conformal ---- */}
      <section className="mt-5">
        <Card>
          <CardTitle hint={`α = ${data.conformal_alpha}`}>
            Calibrated uncertainty ({data.detection_method})
          </CardTitle>
          <p className="rounded border border-cyan-900/50 bg-cyan-950/20 px-3 py-2 text-xs leading-relaxed text-cyan-100/80">
            {data.conformal_guarantee}
          </p>
          <h3 className="mt-3 text-[11px] font-semibold uppercase tracking-wide text-slate-500">
            What the guarantee does not say
          </h3>
          <ul className="mt-1.5 space-y-1.5 pl-4 text-xs leading-relaxed text-slate-400">
            {data.conformal_limitations.map((limit, i) => (
              <li key={i} className="list-disc">
                {limit}
              </li>
            ))}
          </ul>
        </Card>
      </section>

      {/* ---- Physics ---- */}
      {data.physics_model && (
        <section className="mt-5">
          <Card>
            <CardTitle hint="pvlib">Physics baseline</CardTitle>
            <p className="text-xs leading-relaxed text-slate-400">
              {String(
                (data.physics_model as Record<string, unknown>).assumption_note ?? "",
              )}
            </p>
            <div className="mt-3 grid grid-cols-2 gap-2 text-[11px] sm:grid-cols-4">
              {Object.entries(
                ((data.physics_model as Record<string, unknown>).assumptions ??
                  {}) as Record<string, number>,
              ).map(([key, value]) => (
                <div key={key} className="rounded border border-slate-800 px-2 py-1.5">
                  <div className="font-mono text-[10px] text-slate-500">{key}</div>
                  <div className="tabular-nums text-slate-300">{value}</div>
                </div>
              ))}
            </div>
            <p className="mt-2 text-[11px] text-slate-600">
              Free parameters fitted from data:{" "}
              {String(
                (data.physics_model as Record<string, unknown>).free_parameters ?? 0,
              )}{" "}
              (an overall derate). Everything else is fixed physics or published
              system metadata.
            </p>
          </Card>
        </section>
      )}

      {/* ---- Fault taxonomy ---- */}
      <section className="mt-5">
        <Card>
          <CardTitle hint={`${data.fault_taxonomy.length} classes`}>
            How detection is evaluated
          </CardTitle>
          <p className="text-xs leading-relaxed text-slate-400">
            Measured PV telemetry carries no trustworthy fault labels, so
            detection is scored against faults injected into the held-out test
            split — including of the measured data, where the generation and
            weather are genuine and only the failures are simulated. Three of
            these classes are <em>not</em> generation losses; a detector that
            flags them is producing false alarms, and they are scored that way.
          </p>
          <div className="mt-3 space-y-1.5">
            {data.fault_taxonomy.map((fault) => (
              <div
                key={fault.fault_type}
                className="flex gap-3 rounded border border-slate-800 px-3 py-1.5"
              >
                <span
                  className={`mt-0.5 h-2 w-2 shrink-0 rounded-full ${
                    fault.is_generation_loss ? "bg-red-500" : "bg-slate-600"
                  }`}
                  title={
                    fault.is_generation_loss
                      ? "Generation loss — should be detected"
                      : "Not a loss — should not be flagged"
                  }
                />
                <div>
                  <span className="font-mono text-[11px] text-slate-300">
                    {fault.fault_type}
                  </span>
                  <p className="text-[11px] leading-relaxed text-slate-500">
                    {fault.description}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </Card>
      </section>

      {/* ---- Spatial ---- */}
      <section className="mt-5">
        <Card>
          <CardTitle hint={`radius ${spatial.neighbor_radius_km} km`}>
            Spatial attribution
          </CardTitle>
          <dl className="space-y-2 text-xs">
            {(
              [
                ["distance", "Distance"],
                ["normalisation", "Normalisation"],
                ["weighting", "Weighting"],
                ["constraint", "Constraint"],
              ] as const
            ).map(([key, label]) => (
              <div key={key}>
                <dt className="text-[11px] font-medium uppercase tracking-wide text-slate-500">
                  {label}
                </dt>
                <dd className="leading-relaxed text-slate-400">{String(spatial[key])}</dd>
              </div>
            ))}
          </dl>
          <p className="mt-3 rounded border border-slate-800 bg-slate-950/40 px-3 py-2 text-[11px] leading-relaxed text-slate-400">
            <strong className="text-slate-300">Limitation. </strong>
            {String(spatial.limitation)}
          </p>
        </Card>
      </section>

      {/* ---- Build provenance ---- */}
      {artifact && (
        <section className="mt-5">
          <Card>
            <CardTitle>Artifact build</CardTitle>
            <dl className="grid grid-cols-2 gap-x-4 gap-y-1.5 text-[11px] sm:grid-cols-3">
              <div>
                <dt className="text-slate-500">Built at</dt>
                <dd className="text-slate-300">{artifact.built_at ?? "—"}</dd>
              </div>
              <div>
                <dt className="text-slate-500">Commit</dt>
                <dd className="font-mono text-slate-300">
                  {artifact.git_commit?.slice(0, 12) ?? "—"}
                </dd>
              </div>
              <div>
                <dt className="text-slate-500">Python</dt>
                <dd className="text-slate-300">{artifact.python_version ?? "—"}</dd>
              </div>
            </dl>
            {artifact.package_versions && (
              <div className="mt-3 flex flex-wrap gap-1.5">
                {Object.entries(artifact.package_versions).map(([name, version]) => (
                  <span
                    key={name}
                    className="rounded bg-slate-800 px-1.5 py-0.5 font-mono text-[10px] text-slate-400"
                  >
                    {name} {version}
                  </span>
                ))}
              </div>
            )}
            <p className="mt-2 text-[11px] text-slate-600">
              Every metric shown anywhere in this application comes from this
              build. The backend loads prebuilt artifacts and never trains.
            </p>
          </Card>
        </section>
      )}
    </div>
  );
}
