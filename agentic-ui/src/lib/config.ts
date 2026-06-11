export interface StandaloneConfig {
  deploymentUrl: string;
  assistantId: string;
  langsmithApiKey?: string;
}

export const AGENT_MODES = [
  { id: "financial_copilot", label: "Financial Copilot", color: "text-[#2F6868]" },
] as const;

const CONFIG_STORAGE_KEY = "standalone-chat-config";

import { ENV_CONFIG, getAppEnv, getLocalDeploymentUrl } from "@/app/utils/mlConfig";

function getDefaultDeploymentUrl(): string {
  return "http://127.0.0.1:2024";
}

const DEFAULT_ASSISTANT_ID = "financial_copilot";
const DEFAULT_LANGSMITH_API_KEY = "";

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

export function getConfig(): StandaloneConfig | null {
  if (typeof window === "undefined") {
    return null;
  }

  try {
    const stored = localStorage.getItem(CONFIG_STORAGE_KEY);
    if (stored) {
      const config = JSON.parse(stored) as StandaloneConfig;
      // Always use env-based deployment URL — ignores any stale localStorage value
      return {
        ...config,
        deploymentUrl: normalizeUrl(getDefaultDeploymentUrl()),
      };
    }
  } catch (error) {
    console.error("Failed to parse stored config:", error);
  }

  // Return env-aware defaults if nothing is stored
  return {
    deploymentUrl: normalizeUrl(getDefaultDeploymentUrl()),
    assistantId: DEFAULT_ASSISTANT_ID,
    langsmithApiKey: DEFAULT_LANGSMITH_API_KEY || undefined,
  };
}

export function saveConfig(config: StandaloneConfig): void {
  if (typeof window === "undefined") {
    return;
  }

  try {
    // Normalize the URL before saving
    const normalizedConfig = {
      ...config,
      deploymentUrl: normalizeUrl(config.deploymentUrl),
    };
    localStorage.setItem(CONFIG_STORAGE_KEY, JSON.stringify(normalizedConfig));
  } catch (error) {
    console.error("Failed to save config:", error);
  }
}

/**
 * Get normalized deployment URL from config.
 * Ensures the URL has a protocol.
 */
export function getNormalizedDeploymentUrl(): string {
  const config = getConfig();
  if (config?.deploymentUrl) {
    return normalizeUrl(config.deploymentUrl);
  }
  return normalizeUrl(getDefaultDeploymentUrl());
}

// ── Per-thread mode persistence ──────────────────────────────────────────────

const THREAD_MODES_KEY = "thread-agent-modes";

/**
 * Get the saved mode for a specific thread.
 * Returns null if no mode was saved for this thread.
 */
export function getThreadMode(threadId: string): string | null {
  if (typeof window === "undefined") return null;
  try {
    const stored = localStorage.getItem(THREAD_MODES_KEY);
    if (stored) {
      const modes = JSON.parse(stored) as Record<string, string>;
      return modes[threadId] ?? null;
    }
  } catch (error) {
    console.error("Failed to read thread modes:", error);
  }
  return null;
}

/**
 * Save the mode for a specific thread.
 */
export function saveThreadMode(threadId: string, mode: string): void {
  if (typeof window === "undefined") return;
  try {
    const stored = localStorage.getItem(THREAD_MODES_KEY);
    const modes: Record<string, string> = stored ? JSON.parse(stored) : {};
    modes[threadId] = mode;
    localStorage.setItem(THREAD_MODES_KEY, JSON.stringify(modes));
  } catch (error) {
    console.error("Failed to save thread mode:", error);
  }
}

/** Default mode for new threads */
export const DEFAULT_MODE = "financial_copilot";

// ── Financial Copilot ─────────────────────────────────────────────────────────

export const TERMINAL_SERVER_URL =
  process.env.NEXT_PUBLIC_TERMINAL_SERVER_URL ?? "http://localhost:8001";

export const FINANCIAL_ASSISTANT_ID = "financial_copilot";
