import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Emit .next/standalone so the Docker runtime stage can run a traced
  // server.js with no node_modules install. Vercel ignores this setting, so
  // the hosted deployment is unaffected.
  output: "standalone",

  // Vercel deployment: allow API calls to backend without CORS issues
  async rewrites() {
    const apiBase = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";
    // Only add rewrite if API is on a different origin (not needed for same-origin)
    return [
      {
        source: "/api/proxy/:path*",
        destination: `${apiBase}/:path*`,
      },
    ];
  },
};

export default nextConfig;
