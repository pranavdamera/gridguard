import Link from "next/link";
import { api, tryFetch } from "@/lib/api";
import { Card, DataModeBadge } from "@/components/ui/Primitives";

export const revalidate = 120;

export default async function LandingPage() {
  const [{ data: fleet }, { data: sites }] = await Promise.all([
    tryFetch(() => api.fleetSummary()),
    tryFetch(() => api.sites()),
  ]);

  const realCount = sites?.real_count ?? 3;
  const syntheticCount = sites?.synthetic_count ?? 7;
  const capacity = fleet?.total_capacity_kw;

  return (
    <div className="mx-auto max-w-5xl px-4 py-12">
      {/* ---- What is this, in one screen ---- */}
      <section>
        <p className="text-xs font-medium uppercase tracking-[0.18em] text-cyan-500">
          Open-source spatial intelligence for distributed solar
        </p>
        <h1 className="mt-3 text-3xl font-semibold tracking-tight text-slate-50 sm:text-4xl">
          GridGuard forecasts what a healthy solar array{" "}
          <span className="text-cyan-400">should</span> produce, then explains
          why it didn&apos;t.
        </h1>
        <p className="mt-4 max-w-2xl text-[15px] leading-relaxed text-slate-400">
          It learns expected generation from weather, puts a{" "}
          <strong className="text-slate-300">calibrated uncertainty band</strong>{" "}
          around that expectation, flags intervals that fall outside it, and uses{" "}
          <strong className="text-slate-300">nearby sites</strong> to separate a
          real equipment fault from a cloud that fooled the weather model.
        </p>

        <div className="mt-6 flex flex-wrap gap-3">
          <Link
            href="/demo"
            className="rounded-md bg-cyan-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-cyan-500"
          >
            Open the live fleet →
          </Link>
          <Link
            href="/data"
            className="rounded-md border border-slate-700 px-4 py-2 text-sm text-slate-300 transition-colors hover:border-slate-600 hover:text-slate-100"
          >
            What data is real?
          </Link>
          <Link
            href="/methodology"
            className="rounded-md border border-slate-700 px-4 py-2 text-sm text-slate-300 transition-colors hover:border-slate-600 hover:text-slate-100"
          >
            How the model works
          </Link>
        </div>
      </section>

      {/* ---- Real vs simulated, stated up front ---- */}
      <section className="mt-10 grid gap-4 sm:grid-cols-2">
        <Card>
          <DataModeBadge mode="real" />
          <h2 className="mt-2 text-sm font-semibold text-slate-200">
            {realCount} arrays with measured telemetry
          </h2>
          <p className="mt-1.5 text-xs leading-relaxed text-slate-400">
            NIST&apos;s campus arrays in Gaithersburg, Maryland — a year of
            15-minute generation from NREL&apos;s PVDAQ collection, paired with
            irradiance, temperature and wind measured by instruments{" "}
            <em>at the array itself</em>. Forecast accuracy is reported on this
            data.
          </p>
        </Card>

        <Card>
          <DataModeBadge mode="synthetic" />
          <h2 className="mt-2 text-sm font-semibold text-slate-200">
            {syntheticCount} simulated DMV campus sites
          </h2>
          <p className="mt-1.5 text-xs leading-relaxed text-slate-400">
            A physically-modelled fleet used for controlled fault injection,
            because measured data carries no fault labels. Institution names are
            illustrative — no named institution supplied telemetry to this
            project.
          </p>
        </Card>
      </section>

      {/* ---- Why it matters ---- */}
      <section className="mt-10">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-400">
          The problem it solves
        </h2>
        <div className="mt-3 grid gap-4 sm:grid-cols-3">
          <Explainer
            step="1"
            title="Underperformance is invisible"
            body="A solar array that quietly loses 20% still produces a plausible-looking curve. Without a model of what it should have produced, nothing looks wrong."
          />
          <Explainer
            step="2"
            title="Alerts without calibration get ignored"
            body="A threshold picked by eye has an unknown false-alarm rate. GridGuard's bound is conformal: at α = 0.05, at most ~5% of healthy intervals should breach it, with no distributional assumption."
          />
          <Explainer
            step="3"
            title="A drop is not always a fault"
            body="If neighbouring arrays under the same sky dropped too, it was the weather, not the hardware. GridGuard checks the neighbourhood before pointing at equipment."
          />
        </div>
      </section>

      {/* ---- Fleet snapshot ---- */}
      {fleet && (
        <section className="mt-10">
          <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-400">
            Current fleet
          </h2>
          <div className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-4">
            <Snapshot label="Sites" value={fleet.total_sites} />
            <Snapshot
              label="Capacity"
              value={capacity ? `${(capacity / 1000).toFixed(2)} MW` : "—"}
            />
            <Snapshot
              label="Needing attention"
              value={fleet.sites_critical + fleet.sites_warning}
              tone={fleet.sites_critical > 0 ? "bad" : "neutral"}
            />
            <Snapshot label="Open events" value={fleet.active_events} />
          </div>
        </section>
      )}

      <p className="mt-10 border-t border-slate-800 pt-4 text-xs leading-relaxed text-slate-600">
        GridGuard is a research and engineering demonstration, not an operational
        monitoring product. It detects and localises deviation; it does not
        diagnose which component failed, and its detection performance is
        characterised against injected faults rather than field-validated
        failures.
      </p>
    </div>
  );
}

function Explainer({
  step,
  title,
  body,
}: {
  step: string;
  title: string;
  body: string;
}) {
  return (
    <Card>
      <div className="flex items-center gap-2">
        <span className="flex h-5 w-5 items-center justify-center rounded-full bg-slate-800 text-[11px] font-semibold text-cyan-400">
          {step}
        </span>
        <h3 className="text-sm font-medium text-slate-200">{title}</h3>
      </div>
      <p className="mt-2 text-xs leading-relaxed text-slate-400">{body}</p>
    </Card>
  );
}

function Snapshot({
  label,
  value,
  tone = "neutral",
}: {
  label: string;
  value: string | number;
  tone?: "neutral" | "bad";
}) {
  return (
    <div className="rounded-lg border border-slate-800 bg-slate-900/50 px-3 py-2.5">
      <div className="text-[11px] uppercase tracking-wide text-slate-500">
        {label}
      </div>
      <div
        className={`mt-0.5 text-lg font-semibold tabular-nums ${
          tone === "bad" ? "text-red-400" : "text-slate-100"
        }`}
      >
        {value}
      </div>
    </div>
  );
}
