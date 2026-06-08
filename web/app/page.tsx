import Link from "next/link";

const FEATURES = [
  {
    title: "Weather-aware forecasting",
    body: "XGBoost and Ridge models predict expected generation from irradiance, temperature, and time features — calibrated to the DC/Northern Virginia climate.",
  },
  {
    title: "Lag-leakage-free anomaly detection",
    body: "The expected-generation baseline uses weather/time features only. No lag features — so persistent degradation isn't hidden by a model that adapted to low output.",
  },
  {
    title: "Event grouping",
    body: "Consecutive anomalous intervals are collapsed into human-readable events with duration, lost kWh, severity, and plain-English explanations.",
  },
  {
    title: "SHAP explainability",
    body: "Feature contributions are computed per event so operators know which signals — irradiance, temperature, time-of-day — drove the anomaly flag.",
  },
  {
    title: "Fleet-level visibility",
    body: "Seven illustrative DMV campus PV sites. Each site has capacity, coordinates, and can be monitored independently.",
  },
  {
    title: "Research-grade transparency",
    body: "Open model card, honest methodology page, and clear synthetic-data labeling. Built for professor outreach, internship portfolios, and peer review.",
  },
];

export default function LandingPage() {
  return (
    <div className="max-w-5xl mx-auto px-4 py-16">
      {/* Hero */}
      <div className="text-center mb-16">
        <div className="inline-block text-xs font-mono border border-slate-700 rounded px-3 py-1 text-slate-500 mb-6">
          Deterministic synthetic demo — GMU-style campus PV assets
        </div>
        <h1 className="text-4xl sm:text-5xl font-bold tracking-tight text-slate-100 mb-5 leading-tight">
          Solar asset intelligence
          <br />
          <span className="text-cyan-400">for campus-scale PV</span>
        </h1>
        <p className="text-lg text-slate-400 max-w-2xl mx-auto mb-8 leading-relaxed">
          GridGuard detects underperformance events in DC/Northern Virginia solar installations
          using weather-aware forecasting, residual anomaly scoring, and operator-ready explanations.
        </p>
        <div className="flex flex-col sm:flex-row items-center justify-center gap-3">
          <Link
            href="/demo"
            className="px-6 py-2.5 bg-cyan-600 hover:bg-cyan-500 text-white rounded text-sm font-medium transition-colors"
          >
            View live demo →
          </Link>
          <Link
            href="/methodology"
            className="px-6 py-2.5 border border-slate-600 hover:border-slate-400 text-slate-300 rounded text-sm font-medium transition-colors"
          >
            Methodology
          </Link>
        </div>
      </div>

      {/* Demo scenario callout */}
      <div className="border border-amber-800/60 bg-amber-950/20 rounded-lg p-5 mb-12 flex gap-4">
        <div className="text-amber-400 text-xl shrink-0">⚡</div>
        <div>
          <div className="text-amber-300 font-medium text-sm mb-1">Primary demo scenario</div>
          <div className="text-slate-400 text-sm">
            GMU Fairfax Campus Solar Array — <strong className="text-slate-300">June 15, 2023</strong>.
            The system detects a 70% output reduction from 09:00–12:15, estimates lost kWh,
            and shows exactly which features drove the anomaly flag.{" "}
            <Link href="/demo" className="text-amber-400 underline hover:text-amber-300">Jump to demo →</Link>
          </div>
        </div>
      </div>

      {/* Features grid */}
      <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4 mb-16">
        {FEATURES.map((f) => (
          <div key={f.title} className="border border-slate-800 rounded-lg p-5 bg-slate-900/50">
            <h3 className="text-slate-100 font-medium text-sm mb-2">{f.title}</h3>
            <p className="text-slate-500 text-xs leading-relaxed">{f.body}</p>
          </div>
        ))}
      </div>

      {/* Architecture overview */}
      <div className="border-t border-slate-800 pt-10">
        <h2 className="text-slate-400 text-xs uppercase tracking-widest font-semibold mb-6">
          Architecture
        </h2>
        <div className="grid sm:grid-cols-3 gap-4 text-sm">
          <div className="border border-slate-800 rounded p-4">
            <div className="text-cyan-400 font-mono text-xs mb-1">Backend</div>
            <div className="text-slate-300 font-medium mb-1">FastAPI + XGBoost</div>
            <div className="text-slate-500 text-xs">Python · scikit-learn · SHAP · uvicorn</div>
          </div>
          <div className="border border-slate-800 rounded p-4">
            <div className="text-cyan-400 font-mono text-xs mb-1">Frontend</div>
            <div className="text-slate-300 font-medium mb-1">Next.js + Tailwind</div>
            <div className="text-slate-500 text-xs">TypeScript · Recharts · Vercel-ready</div>
          </div>
          <div className="border border-slate-800 rounded p-4">
            <div className="text-cyan-400 font-mono text-xs mb-1">Data</div>
            <div className="text-slate-300 font-medium mb-1">Synthetic demo bundle</div>
            <div className="text-slate-500 text-xs">NREL PVDAQ / NSRDB adapters planned</div>
          </div>
        </div>
      </div>
    </div>
  );
}
