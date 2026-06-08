import type { NextConfig } from "next";

const nextConfig: NextConfig = {
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
