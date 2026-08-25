# GridGuard frontend image.
#
# Uses Next's standalone output, which traces the files each route actually
# needs and emits a self-contained server.js, so the runtime stage carries no
# node_modules install of its own.

# ---------------------------------------------------------------------------
# deps
# ---------------------------------------------------------------------------
FROM node:20-slim AS deps

WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
FROM node:20-slim AS build

WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .

# NEXT_PUBLIC_* values are inlined at build time, not read at runtime, so the
# API origin has to be fixed here. Inside compose that is the service name;
# every page is a server component, so this URL is resolved from the container
# network and never from the viewer's browser.
ARG NEXT_PUBLIC_API_URL=http://api:8000
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL
ENV NEXT_TELEMETRY_DISABLED=1

RUN npm run build

# ---------------------------------------------------------------------------
# runtime
# ---------------------------------------------------------------------------
FROM node:20-slim AS runtime

WORKDIR /app
ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    PORT=3000 \
    HOSTNAME=0.0.0.0

# server.js serves public/ and .next/static only if they sit beside it; the
# standalone output does not copy either, on the assumption a CDN handles them.
COPY --from=build /app/.next/standalone ./
COPY --from=build /app/.next/static ./.next/static
COPY --from=build /app/public ./public

RUN useradd --system --uid 1001 nextjs && chown -R nextjs:nextjs /app
USER nextjs

EXPOSE 3000

CMD ["node", "server.js"]
