import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "GridGuard — Solar Fleet Intelligence",
  description:
    "Open-source spatial intelligence and anomaly detection for distributed solar. Forecasts expected generation, calibrates uncertainty with conformal prediction, and separates equipment faults from regional weather.",
};

const NAV = [
  { href: "/demo", label: "Fleet" },
  { href: "/sites", label: "Sites" },
  { href: "/data", label: "Data" },
  { href: "/methodology", label: "Methodology" },
];

function NavBar() {
  return (
    <nav className="relative z-10 border-b border-slate-800 bg-slate-900/80 backdrop-blur-sm">
      <div className="mx-auto flex h-12 max-w-6xl items-center gap-6 px-4">
        <Link
          href="/"
          className="text-sm font-semibold tracking-tight text-slate-100 transition-colors hover:text-cyan-300"
        >
          <span className="text-cyan-400">Grid</span>Guard
        </Link>
        <div className="flex items-center gap-4 text-sm text-slate-400">
          {NAV.map((item) => (
            <Link
              key={item.href}
              href={item.href}
              className="transition-colors hover:text-slate-200"
            >
              {item.label}
            </Link>
          ))}
        </div>
        {/*
          The global provenance reminder. GridGuard serves measured and
          simulated data side by side, so the distinction is stated in the
          chrome as well as on every card that shows numbers.
        */}
        <Link
          href="/data"
          className="ml-auto hidden text-xs text-slate-500 transition-colors hover:text-slate-300 sm:block"
        >
          <span className="mr-1 inline-block h-1.5 w-1.5 rounded-full bg-emerald-400 align-middle" />
          measured
          <span className="mx-1.5 text-slate-700">+</span>
          <span className="mr-1 inline-block h-1.5 w-1.5 rounded-full bg-amber-400 align-middle" />
          simulated data →
        </Link>
      </div>
    </nav>
  );
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="h-full antialiased">
      <body className="flex min-h-full flex-col bg-[#0d1117] text-slate-200">
        <NavBar />
        <main className="relative z-10 flex-1">{children}</main>
        <footer className="relative z-10 border-t border-slate-800 px-4 py-4 text-center text-xs leading-relaxed text-slate-600">
          <p>
            GridGuard — open-source solar fleet intelligence ·{" "}
            <Link href="/data" className="underline hover:text-slate-400">
              data provenance
            </Link>{" "}
            ·{" "}
            <a
              href="https://github.com/pranavdamera/gridguard"
              className="underline hover:text-slate-400"
              target="_blank"
              rel="noopener noreferrer"
            >
              GitHub
            </a>
          </p>
          <p className="mx-auto mt-1.5 max-w-2xl">
            A research and engineering demonstration. Detection performance is
            characterised against injected faults, not field-validated failures.
            Not a basis for dispatch, maintenance, or financial decisions.
          </p>
        </footer>
      </body>
    </html>
  );
}
