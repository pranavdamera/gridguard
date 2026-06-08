import Link from "next/link";

interface Props {
  label?: string;
}

export function MethodologyNotice({ label }: Props) {
  return (
    <div className="flex items-start gap-2 text-xs text-slate-500 border border-slate-700 rounded px-3 py-2 bg-slate-800/40">
      <span className="text-amber-500 shrink-0">⚠</span>
      <span>
        {label ?? (
          <>
            All generation data shown is{" "}
            <strong className="text-slate-400">deterministic synthetic demo telemetry</strong>{" "}
            for a GMU-style campus PV asset. Not real sensor data.{" "}
            Real-data adapters (NREL PVDAQ, NSRDB, CSV upload) are planned.{" "}
            <Link href="/methodology" className="underline hover:text-slate-300">Methodology →</Link>
          </>
        )}
      </span>
    </div>
  );
}
