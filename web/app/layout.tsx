import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "GridGuard — Solar Asset Intelligence",
  description:
    "Research-grade solar generation forecasting and underperformance detection for campus-scale PV assets in the DC/Northern Virginia region.",
};

function NavBar() {
  return (
    <nav className="relative z-10 border-b border-slate-800 bg-slate-900/80 backdrop-blur-sm">
      <div className="max-w-6xl mx-auto px-4 flex items-center h-12 gap-6">
        <Link href="/" className="text-sm font-semibold tracking-tight text-slate-100 hover:text-cyan-300 transition-colors">
          <span className="text-cyan-400">Grid</span>Guard
        </Link>
        <div className="flex items-center gap-4 text-sm text-slate-400">
          <Link href="/demo" className="hover:text-slate-200 transition-colors">Demo</Link>
          <Link href="/sites" className="hover:text-slate-200 transition-colors">Sites</Link>
          <Link href="/methodology" className="hover:text-slate-200 transition-colors">Methodology</Link>
        </div>
        <div className="ml-auto text-xs text-slate-600">
          Synthetic demo data — not real sensor data
        </div>
      </div>
    </nav>
  );
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="h-full antialiased">
      <body className="min-h-full flex flex-col bg-[#0d1117] text-slate-200">
        <NavBar />
        <main className="relative z-10 flex-1">{children}</main>
        <footer className="relative z-10 border-t border-slate-800 py-4 text-center text-xs text-slate-600">
          GridGuard — Open-source solar asset intelligence &bull; Synthetic GMU-style PV demo &bull;{" "}
          <a href="https://github.com/pranav-damera/gridguard" className="underline hover:text-slate-400">GitHub</a>
        </footer>
      </body>
    </html>
  );
}
