"use client";

import { createContext, useContext, useMemo, ReactNode } from "react";
import { Client } from "@langchain/langgraph-sdk";

/**
 * Normalizes a URL by ensuring it has a protocol.
 * If no protocol is provided, defaults to https://
 */
function normalizeUrl(url: string): string {
  if (!url) return url;
  
  // Remove any trailing slashes
  url = url.trim().replace(/\/+$/, "");
  
  // Check if URL already has a protocol
  if (/^https?:\/\//i.test(url)) {
    return url;
  }
  
  // Default to https:// for production URLs
  return `https://${url}`;
}

interface ClientContextValue {
  client: Client;
}

const ClientContext = createContext<ClientContextValue | null>(null);

interface ClientProviderProps {
  children: ReactNode;
  deploymentUrl: string;
  apiKey: string;
}

export function ClientProvider({
  children,
  deploymentUrl,
  apiKey,
}: ClientProviderProps) {
  const client = useMemo(() => {
    // Normalize URL to ensure it has a protocol
    const normalizedUrl = normalizeUrl(deploymentUrl);
    
    return new Client({
      apiUrl: normalizedUrl,
      defaultHeaders: {
        "Content-Type": "application/json",
        "X-Api-Key": apiKey,
      },
    });
  }, [deploymentUrl, apiKey]);

  const value = useMemo(() => ({ client }), [client]);

  return (
    <ClientContext.Provider value={value}>{children}</ClientContext.Provider>
  );
}

export function useClient(): Client {
  const context = useContext(ClientContext);

  if (!context) {
    throw new Error("useClient must be used within a ClientProvider");
  }
  return context.client;
}
