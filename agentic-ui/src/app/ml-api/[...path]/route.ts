import { NextRequest, NextResponse } from "next/server";

const ENV_BASES: Record<string, string> = {
  prod: "https://lucid-v3-prod.westeurope.cloudapp.azure.com",
  test: "https://test.lucid.zypl.ai/api",
  local: "https://test.lucid.zypl.ai/api", // same as test
};

async function handler(req: NextRequest, { params }: { params: Promise<{ path: string[] }> }) {
  const { path } = await params;
  const env = req.headers.get("x-lucid-env") ?? "prod";
  const base = ENV_BASES[env] ?? ENV_BASES.prod;

  const url = new URL(`${base}/${path.join("/")}`);
  req.nextUrl.searchParams.forEach((value, key) => url.searchParams.set(key, value));

  const headers = new Headers(req.headers);
  headers.delete("host");

  const upstream = await fetch(url.toString(), {
    method: req.method,
    headers,
    body: req.method !== "GET" && req.method !== "HEAD" ? req.body : undefined,
    // @ts-expect-error Node fetch duplex
    duplex: "half",
  });

  return new NextResponse(upstream.body, {
    status: upstream.status,
    headers: upstream.headers,
  });
}

export const GET = handler;
export const POST = handler;
export const PUT = handler;
export const PATCH = handler;
export const DELETE = handler;
