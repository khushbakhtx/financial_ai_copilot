"use client";

import React, {
  useEffect,
  useRef,
  useState,
  useCallback,
  useId,
} from "react";

const TERMINAL_SERVER =
  process.env.NEXT_PUBLIC_TERMINAL_SERVER_URL ?? "http://localhost:8001";

interface TerminalEvent {
  kind: "stdout" | "stderr" | "system" | "start" | "end";
  text: string;
  agent: string;
  ts: string;
}

interface LogEntry extends TerminalEvent {
  id: string;
  /** true while the CSS enter animation is in its first frame */
  fresh: boolean;
}

// Strip ANSI escape codes from sandbox output
function stripAnsi(str: string): string {
  // eslint-disable-next-line no-control-regex
  return str.replace(/\x1b\[[0-9;]*[mGKHF]/g, "").replace(/\r/g, "");
}

// ── Per-kind visual config ────────────────────────────────────────────────────

const KIND_CONFIG = {
  start: {
    dot: "bg-[#1c3c3c]",
    dotPulse: true,
    textClass: "font-semibold text-[#1c3c3c]",
    prefixIcon: (
      <span className="mr-1.5 inline-block h-1.5 w-1.5 translate-y-[-1px] rounded-full bg-[#1c3c3c]" />
    ),
    bg: "bg-[#1c3c3c]/[0.04]",
    border: "border-l-2 border-[#1c3c3c]/30",
  },
  end: {
    dot: "bg-emerald-500",
    dotPulse: false,
    textClass: "text-emerald-700",
    prefixIcon: (
      <svg
        className="mr-1.5 inline-block h-3 w-3 translate-y-[-1px] text-emerald-500"
        viewBox="0 0 12 12"
        fill="none"
      >
        <path
          d="M2 6l3 3 5-5"
          stroke="currentColor"
          strokeWidth="1.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
    ),
    bg: "bg-emerald-50/60",
    border: "border-l-2 border-emerald-300/60",
  },
  stdout: {
    dot: "bg-[#1c3c3c]/40",
    dotPulse: false,
    textClass: "text-foreground/80",
    prefixIcon: null,
    bg: "",
    border: "",
  },
  stderr: {
    dot: "bg-red-400",
    dotPulse: false,
    textClass: "text-red-700",
    prefixIcon: (
      <svg
        className="mr-1.5 inline-block h-3 w-3 translate-y-[-1px] text-red-400"
        viewBox="0 0 12 12"
        fill="none"
      >
        <path
          d="M6 4v3M6 8.5v.5"
          stroke="currentColor"
          strokeWidth="1.5"
          strokeLinecap="round"
        />
        <rect x="1" y="1" width="10" height="10" rx="2" stroke="currentColor" strokeWidth="1" />
      </svg>
    ),
    bg: "bg-red-50/50",
    border: "border-l-2 border-red-300/60",
  },
  system: {
    dot: "bg-blue-400",
    dotPulse: false,
    textClass: "text-blue-700/80",
    prefixIcon: null,
    bg: "bg-blue-50/40",
    border: "border-l-2 border-blue-200",
  },
} as const;

// ── Single log row ────────────────────────────────────────────────────────────

const LogRow = React.memo(function LogRow({ entry }: { entry: LogEntry }) {
  const cfg = KIND_CONFIG[entry.kind];
  const text = stripAnsi(entry.text);
  if (!text.trim()) return null;

  return (
    <div
      className={[
        "group flex items-start gap-2.5 rounded-md px-3 py-1.5 transition-all duration-300",
        cfg.bg,
        cfg.border,
        entry.fresh
          ? "translate-y-1 opacity-0"
          : "translate-y-0 opacity-100",
      ].join(" ")}
      style={{ transitionProperty: "opacity, transform" }}
    >
      {/* Timeline dot */}
      <div className="mt-[5px] flex-shrink-0">
        <span
          className={[
            "block h-1.5 w-1.5 rounded-full",
            cfg.dot,
            cfg.dotPulse ? "animate-pulse" : "",
          ].join(" ")}
        />
      </div>

      {/* Content */}
      <div className="min-w-0 flex-1">
        {/* Agent badge on start events */}
        {entry.kind === "start" && entry.agent && (
          <span className="mb-0.5 block text-[10px] font-medium uppercase tracking-wider text-[#1c3c3c]/50">
            {entry.agent}
          </span>
        )}

        <p className={["break-words font-mono text-[11.5px] leading-relaxed", cfg.textClass].join(" ")}>
          {cfg.prefixIcon}
          {text}
        </p>
      </div>

      {/* Timestamp — visible on hover */}
      <span className="mt-[3px] flex-shrink-0 font-mono text-[9px] text-muted-foreground/0 transition-all group-hover:text-muted-foreground/60">
        {entry.ts}
      </span>
    </div>
  );
});

// ── Running indicator shown between start and end ─────────────────────────────

function RunningIndicator({ agent }: { agent: string }) {
  return (
    <div className="flex items-center gap-2.5 px-3 py-1.5">
      <div className="mt-[5px] flex-shrink-0">
        <span className="block h-1.5 w-1.5 animate-pulse rounded-full bg-[#1c3c3c]/40" />
      </div>
      <div className="flex items-center gap-1.5">
        <span className="flex gap-[3px]">
          {[0, 1, 2].map((i) => (
            <span
              key={i}
              className="block h-1 w-1 rounded-full bg-[#1c3c3c]/30"
              style={{
                animation: "sandboxBounce 1.2s ease-in-out infinite",
                animationDelay: `${i * 0.18}s`,
              }}
            />
          ))}
        </span>
        {agent && (
          <span className="font-mono text-[10px] text-muted-foreground/60">
            {agent}
          </span>
        )}
      </div>
    </div>
  );
}

// ── Empty state ───────────────────────────────────────────────────────────────

function EmptyState({ connected }: { connected: boolean }) {
  return (
    <div className="flex flex-1 flex-col items-center justify-center gap-3 px-6 py-12">
      <div className="flex h-10 w-10 items-center justify-center rounded-xl border border-border bg-sidebar">
        <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
          <rect x="2" y="2" width="6" height="6" rx="1.5" fill="#1c3c3c" opacity="0.7" />
          <rect x="10" y="2" width="6" height="6" rx="1.5" fill="#1c3c3c" opacity="0.3" />
          <rect x="2" y="10" width="6" height="6" rx="1.5" fill="#1c3c3c" opacity="0.3" />
          <rect x="10" y="10" width="6" height="6" rx="1.5" fill="#1c3c3c" opacity="0.7" />
        </svg>
      </div>
      <div className="text-center">
        <p className="text-xs font-medium text-foreground/70">
          {connected ? "Sandbox ready" : "Connecting…"}
        </p>
        <p className="mt-0.5 text-[11px] text-muted-foreground">
          {connected
            ? "Execution output will appear here"
            : "Waiting for terminal server"}
        </p>
      </div>
    </div>
  );
}

// ── Main panel ────────────────────────────────────────────────────────────────

export const TerminalPanel = React.memo(function TerminalPanel() {
  const [entries, setEntries] = useState<LogEntry[]>([]);
  const [connected, setConnected] = useState(false);
  const [running, setRunning] = useState<string | null>(null);
  const esRef = useRef<EventSource | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const uid = useId();
  const counter = useRef(0);

  // Auto-scroll to bottom on new entries
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const isNearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 120;
    if (isNearBottom) {
      el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
    }
  }, [entries]);

  // Remove "fresh" flag after one animation frame so the enter animation fires
  const addEntry = useCallback((event: TerminalEvent) => {
    const id = `${uid}-${++counter.current}`;
    const entry: LogEntry = { ...event, id, fresh: true };

    setEntries((prev) => {
      const next = [...prev, entry];
      return next.slice(-500); // cap at 500 entries
    });

    // Flip fresh → false after paint so CSS transition fires
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        setEntries((prev) =>
          prev.map((e) => (e.id === id ? { ...e, fresh: false } : e))
        );
      });
    });

    if (event.kind === "start") setRunning(event.agent || event.text);
    if (event.kind === "end") setRunning(null);
  }, [uid]);

  const connect = useCallback(() => {
    esRef.current?.close();
    const es = new EventSource(`${TERMINAL_SERVER}/terminal/stream`);
    esRef.current = es;

    es.onopen = () => {
      setConnected(true);
      addEntry({
        kind: "system",
        text: "Connected to sandbox",
        agent: "",
        ts: new Date().toLocaleTimeString(),
      });
    };

    es.onmessage = (e) => {
      try {
        const event: TerminalEvent = JSON.parse(e.data);
        addEntry(event);
      } catch {
        // malformed — ignore
      }
    };

    es.onerror = () => {
      setConnected(false);
      setRunning(null);
      esRef.current?.close();
      esRef.current = null;
      setTimeout(connect, 3000);
    };
  }, [addEntry]);

  useEffect(() => {
    const t = setTimeout(connect, 300);
    return () => {
      clearTimeout(t);
      esRef.current?.close();
    };
  }, [connect]);

  const handleClear = () => {
    setEntries([]);
    setRunning(null);
  };

  const eventCount = entries.filter(
    (e) => e.kind === "stdout" || e.kind === "stderr"
  ).length;

  return (
    <>
      {/* Bounce keyframes injected once */}
      <style>{`
        @keyframes sandboxBounce {
          0%, 80%, 100% { transform: translateY(0); opacity: 0.4; }
          40% { transform: translateY(-3px); opacity: 1; }
        }
      `}</style>

      <div className="flex h-full flex-col bg-background">
        {/* ── Header ── */}
        <div className="flex h-10 flex-shrink-0 items-center justify-between border-b border-border bg-sidebar px-4">
          <div className="flex items-center gap-2.5">
            <div className="flex h-5 w-5 items-center justify-center rounded bg-[#1c3c3c]">
              <svg width="10" height="10" viewBox="0 0 10 10" fill="none">
                <rect x="1" y="1" width="3" height="3" rx="0.5" fill="white" opacity="0.9" />
                <rect x="6" y="1" width="3" height="3" rx="0.5" fill="white" opacity="0.4" />
                <rect x="1" y="6" width="3" height="3" rx="0.5" fill="white" opacity="0.4" />
                <rect x="6" y="6" width="3" height="3" rx="0.5" fill="white" opacity="0.9" />
              </svg>
            </div>
            <span className="text-xs font-semibold text-foreground">Sandbox</span>
            {eventCount > 0 && (
              <span className="rounded-full bg-[#1c3c3c]/[0.07] px-1.5 py-0.5 text-[10px] font-medium text-[#1c3c3c]">
                {eventCount}
              </span>
            )}
          </div>

          <div className="flex items-center gap-3">
            {/* Live indicator */}
            <div className="flex items-center gap-1.5">
              <span
                className={`h-1.5 w-1.5 rounded-full transition-all duration-500 ${
                  running
                    ? "animate-pulse bg-[#1c3c3c]"
                    : connected
                    ? "bg-emerald-500"
                    : "bg-gray-300"
                }`}
              />
              <span className="text-[10px] text-muted-foreground">
                {running ? "running" : connected ? "idle" : "offline"}
              </span>
            </div>
            <button
              onClick={handleClear}
              className="rounded px-2 py-0.5 text-[10px] text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
            >
              clear
            </button>
          </div>
        </div>

        {/* Teal gradient accent */}
        <div className="h-px w-full bg-gradient-to-r from-[#1c3c3c]/30 via-[#1c3c3c]/10 to-transparent" />

        {/* ── Log body ── */}
        <div
          ref={scrollRef}
          className="flex min-h-0 flex-1 flex-col overflow-y-auto px-1 py-2"
          style={{ scrollbarWidth: "thin", scrollbarColor: "#e5e7eb transparent" }}
        >
          {entries.length === 0 ? (
            <EmptyState connected={connected} />
          ) : (
            <div className="flex flex-col gap-0.5">
              {entries.map((entry) => (
                <LogRow key={entry.id} entry={entry} />
              ))}
              {running && <RunningIndicator agent={running} />}
            </div>
          )}
        </div>
      </div>
    </>
  );
});
