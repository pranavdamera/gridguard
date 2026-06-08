import { Card, CardTitle } from "@/components/ui/Card";

export const metadata = {
  title: "Methodology — GridGuard",
};

export default function MethodologyPage() {
  return (
    <div className="max-w-3xl mx-auto px-4 py-10 space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-slate-100 mb-2">Methodology</h1>
        <p className="text-slate-400 text-sm">
          Transparent documentation of what GridGuard does, what is real vs synthetic,
          and the key modeling decisions that determine credibility.
        </p>
      </div>

      <Card>
        <CardTitle>Data</CardTitle>
        <div className="prose prose-invert prose-sm max-w-none text-slate-400 space-y-3">
          <p>
            All generation data shown in this demo is{" "}
            <strong className="text-slate-200">deterministic synthetic telemetry</strong> for
            a GMU Fairfax-style 250 kW campus PV asset. The synthetic model uses:
          </p>
          <ul className="list-disc pl-5 space-y-1">
            <li>Clear-sky irradiance from simplified sun geometry (latitude-parameterised)</li>
            <li>Log-normal cloud attenuation with autocorrelation</li>
            <li>Northern Virginia seasonal and diurnal temperature cycle</li>
            <li>Panel efficiency degradation with temperature (−0.4%/°C above 25°C)</li>
            <li>Random injected fault days (~5% of days) at 40–80% output reduction</li>
            <li>
              <strong className="text-amber-300">Scripted demo event:</strong> June 15, 2023 09:00–12:15 — 70% AC
              power reduction during peak irradiance. This is deterministically injected, not random.
            </li>
          </ul>
          <p>
            Real-data adapters are planned for: NREL PVDAQ, NSRDB satellite irradiance, CSV
            upload from campus telemetry systems, and PJM metering data.
          </p>
        </div>
      </Card>

      <Card>
        <CardTitle>Two feature modes — the most important modeling decision</CardTitle>
        <div className="text-slate-400 text-sm space-y-3">
          <p>
            GridGuard uses <strong className="text-slate-200">strictly separate feature sets</strong> for
            forecasting vs anomaly detection. This is not cosmetic — it prevents a class of temporal leakage
            that would make the anomaly detector meaningless.
          </p>

          <div className="grid sm:grid-cols-2 gap-4 mt-3">
            <div className="border border-slate-700 rounded p-3">
              <div className="text-cyan-400 text-xs font-semibold uppercase tracking-wider mb-2">
                Weather-only (anomaly detection)
              </div>
              <div className="text-xs space-y-1">
                <div className="text-slate-300">Used to answer: <em>what should a healthy system produce?</em></div>
                <div className="text-slate-500 mt-2">Features: irradiance, irradiance², temperature, wind speed, 1h rolling irradiance, hour sin/cos, day-of-year sin/cos, month</div>
                <div className="text-emerald-400 mt-2">✓ No lag features</div>
              </div>
            </div>
            <div className="border border-slate-700 rounded p-3">
              <div className="text-amber-400 text-xs font-semibold uppercase tracking-wider mb-2">
                Lag-aware (operational forecasting)
              </div>
              <div className="text-xs space-y-1">
                <div className="text-slate-300">Used to answer: <em>what will the system produce next?</em></div>
                <div className="text-slate-500 mt-2">Features: all weather-only features + ac_power_lag1 (prev 15 min) + ac_power_lag4 (1 h ago)</div>
                <div className="text-red-400 mt-2">⚠ Not used for fault detection</div>
              </div>
            </div>
          </div>

          <p>
            <strong className="text-slate-200">Why this matters:</strong> If lag features are included in the
            expected-generation model, a degraded inverter that has been producing low power for hours will
            have low lags. The model then predicts low power as expected, the residual is small, and the
            anomaly is missed. Using weather-only features means the model always asks &ldquo;given today&apos;s
            sun and temperature, what should a healthy system produce?&rdquo;
          </p>
        </div>
      </Card>

      <Card>
        <CardTitle>Anomaly detection method</CardTitle>
        <div className="text-slate-400 text-sm space-y-2">
          <p>
            Residual thresholding with training-set calibration:
          </p>
          <ol className="list-decimal pl-5 space-y-1">
            <li>Temporal split: training data before 2023-01-01, test data after. Never random shuffle.</li>
            <li>Within training data, reserve the last 15% as a validation set for early stopping.</li>
            <li>
              Train <code className="text-xs bg-slate-700 px-1 rounded">anomaly_detector.pkl</code> (XGBoost,
              weather-only features) on <em>healthy training intervals only</em> — injected fault rows are
              excluded so the model cannot learn "low output is normal."
            </li>
            <li>Compute residuals (actual − predicted) on healthy training intervals.</li>
            <li>Fit per-hour-of-day (mean, std) from those residuals. Save to artifact file.</li>
            <li>At inference: residual_σ = (residual − mean_train_hour) / std_train_hour.</li>
            <li>Flag as anomaly if residual_σ {"<"} −2.0 during daylight (irradiance {">"} 50 W/m²).</li>
          </ol>
          <p className="mt-2">
            The per-hour calibration prevents false positives at peak irradiance (where absolute
            residuals are naturally larger) and misses at dawn/dusk.
          </p>
          <p>
            <strong className="text-slate-200">Next step:</strong> Replace sigma thresholding with
            conformal prediction intervals for coverage guarantees. Implementation stub is in{" "}
            <code className="text-xs bg-slate-700 px-1 rounded">src/gridguard/anomaly/conformal.py</code>.
          </p>
        </div>
      </Card>

      <Card>
        <CardTitle>Models trained</CardTitle>
        <div className="text-slate-400 text-sm space-y-4">
          <div>
            <div className="text-slate-300 font-medium mb-1.5">
              Anomaly detector — <code className="text-xs bg-slate-700 px-1 rounded">anomaly_detector.pkl</code>
            </div>
            <ul className="space-y-1">
              <li><strong className="text-emerald-400">XGBoost, weather-only features, healthy data only.</strong> Used for /anomalies and /events. Cannot see lag features; cannot mask degradation.</li>
            </ul>
          </div>
          <div>
            <div className="text-slate-300 font-medium mb-1.5">Forecasting models — lag-aware, for /forecast and /metrics comparison</div>
            <ul className="space-y-1">
              <li><strong className="text-slate-300">Persistence</strong> — carry last observation. Naive benchmark.</li>
              <li><strong className="text-slate-300">Ridge</strong> — linear, L2-regularized. Interpretable baseline.</li>
              <li><strong className="text-slate-300">Random Forest</strong> — ensemble decision trees. Solid nonlinear baseline.</li>
              <li><strong className="text-slate-300">XGBoost (lag-aware)</strong> — gradient-boosted trees. Best short-horizon forecast accuracy.</li>
            </ul>
          </div>
          <p>
            All models use a chronological train/validation/test split. XGBoost uses the validation set
            for early stopping — <em>never the test set</em>. Random splits are never used.
          </p>
        </div>
      </Card>

      <Card>
        <CardTitle>Explainability</CardTitle>
        <div className="text-slate-400 text-sm space-y-2">
          <p>
            SHAP (SHapley Additive exPlanations) values are computed per anomaly interval using
            the best available model. For tree models (XGBoost, Random Forest) this uses the
            TreeExplainer — exact Shapley values, not approximations.
          </p>
          <p>
            If SHAP is unavailable, the dashboard falls back to model feature importances, clearly labeled.
          </p>
        </div>
      </Card>

      <Card>
        <CardTitle>What this is not</CardTitle>
        <div className="text-slate-400 text-sm space-y-1">
          <ul className="list-disc pl-5 space-y-1">
            <li>Not a real-time system. Data is static demo telemetry, regenerated on demand.</li>
            <li>Not connected to live sensor data. No SCADA, no live inverter feeds.</li>
            <li>Not production-grade. No auth, rate limiting, or database persistence in this MVP.</li>
            <li>Not claiming research-validated results on real installations (yet).</li>
          </ul>
        </div>
      </Card>

      <div className="text-xs text-slate-600 pt-4 border-t border-slate-800">
        Full source code: <a href="https://github.com/pranav-damera/gridguard" className="underline">github.com/pranav-damera/gridguard</a>
      </div>
    </div>
  );
}
