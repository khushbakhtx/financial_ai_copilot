"use client";

import React, { useState, useEffect, useRef, useCallback } from "react";
import {
  ChevronDown,
  ChevronUp,
  X,
  CheckCircle2,
  XCircle,
  Loader2,
  BrainCircuit,
  Clock3,
} from "lucide-react";
import { cn } from "@/lib/utils";
import type { ZganTask, ZganLog } from "@/app/hooks/useZganMonitor";

// ── helpers ──────────────────────────────────────────────────────────────────

function formatElapsed(startedAt: Date): string {
  const totalSeconds = Math.floor((Date.now() - startedAt.getTime()) / 1000);
  const h = Math.floor(totalSeconds / 3600);
  const m = Math.floor((totalSeconds % 3600) / 60);
  const s = totalSeconds % 60;
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

function formatTime(date: Date): string {
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

// ── sub-components ────────────────────────────────────────────────────────────

interface TaskRowProps {
  task: ZganTask;
  isExpanded: boolean;
  onToggle: () => void;
  onDismiss: () => void;
  elapsedTick: number; // forces re-render every second
}

const TaskRow = React.memo<TaskRowProps>(({ task, isExpanded, onToggle, onDismiss, elapsedTick }) => {
  const logsEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll logs to bottom when new entries arrive
  useEffect(() => {
    if (isExpanded) {
      logsEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [task.logs.length, isExpanded]);

  const statusIcon =
    task.status === "done" ? (
      <CheckCircle2 size={14} className="text-emerald-400 shrink-0" />
    ) : task.status === "failed" ? (
      <XCircle size={14} className="text-red-400 shrink-0" />
    ) : task.status === "queued" ? (
      <Clock3 size={14} className="text-amber-400 shrink-0" />
    ) : (
      <Loader2 size={14} className="animate-spin text-[#4a9a9a] shrink-0" />
    );

  const statusLabel =
    task.status === "done" ? "Complete"
    : task.status === "failed" ? "Failed"
    : task.status === "queued" ? "Queued"
    : "Running";

  const elapsed = formatElapsed(task.startedAt);
  // suppress unused-warning — elapsedTick is intentionally consumed to force re-render
  void elapsedTick;

  return (
    <div className="flex flex-col">
      {/* ── row header ── */}
      <div className="flex items-center gap-2 px-3 py-2.5">
        {statusIcon}
        <button
          type="button"
          onClick={onToggle}
          className="flex min-w-0 flex-1 items-center gap-2 text-left"
        >
          <span className="truncate text-xs font-medium text-foreground leading-none">
            {task.label}
          </span>
          <span className="shrink-0 text-[10px] text-muted-foreground leading-none">
            {statusLabel} · {elapsed}
          </span>
        </button>
        <div className="flex items-center gap-1">
          <button
            type="button"
            onClick={onToggle}
            className="rounded p-0.5 text-muted-foreground transition-colors hover:text-foreground"
            aria-label={isExpanded ? "Collapse" : "Expand"}
          >
            {isExpanded ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
          </button>
          {(task.status === "done" || task.status === "failed") && (
            <button
              type="button"
              onClick={onDismiss}
              className="rounded p-0.5 text-muted-foreground transition-colors hover:text-foreground"
              aria-label="Dismiss"
            >
              <X size={12} />
            </button>
          )}
        </div>
      </div>

      {/* ── expandable logs ── */}
      {isExpanded && (
        <div className="max-h-48 overflow-y-auto border-t border-border/60 px-3 pb-3 pt-2">
          <div className="flex flex-col gap-1">
            {task.logs.map((log: ZganLog, i: number) => (
              <div
                key={i}
                className={cn(
                  "flex gap-2 text-[10px] leading-relaxed",
                  // fade-in for entries beyond the first
                  i > 0 && "animate-in fade-in duration-300"
                )}
              >
                <span className="shrink-0 font-mono text-muted-foreground/70">
                  {formatTime(log.timestamp)}
                </span>
                <span
                  className={cn(
                    "break-words",
                    log.type === "success" && "text-emerald-400",
                    log.type === "error" && "text-red-400",
                    log.type === "info" && "text-muted-foreground"
                  )}
                >
                  {log.message}
                </span>
              </div>
            ))}
            <div ref={logsEndRef} />
          </div>
        </div>
      )}
    </div>
  );
});

TaskRow.displayName = "TaskRow";

// ── main panel ────────────────────────────────────────────────────────────────

interface ZganMonitorPanelProps {
  tasks: ZganTask[];
  onDismiss: (tcId: string) => void;
}

export const ZganMonitorPanel = React.memo<ZganMonitorPanelProps>(({ tasks, onDismiss }) => {
  const [isPanelExpanded, setIsPanelExpanded] = useState(true);
  const [expandedTaskIds, setExpandedTaskIds] = useState<Set<string>>(new Set());
  // Tick every second to refresh elapsed time display
  const [tick, setTick] = useState(0);

  useEffect(() => {
    const timer = setInterval(() => setTick((t) => t + 1), 1000);
    return () => clearInterval(timer);
  }, []);

  // Auto-expand new tasks
  useEffect(() => {
    setExpandedTaskIds((prev) => {
      const next = new Set(prev);
      tasks.forEach((t) => next.add(t.tcId));
      return next;
    });
  }, [tasks]);

  // Auto-expand panel when a new task arrives
  useEffect(() => {
    if (tasks.length > 0) setIsPanelExpanded(true);
  }, [tasks.length]);

  const toggleTask = useCallback((tcId: string) => {
    setExpandedTaskIds((prev) => {
      const next = new Set(prev);
      if (next.has(tcId)) next.delete(tcId);
      else next.add(tcId);
      return next;
    });
  }, []);

  if (tasks.length === 0) return null;

  const runningCount = tasks.filter((t) => t.status === "running" || t.status === "queued").length;
  const hasRunning = runningCount > 0;

  return (
    <div
      className={cn(
        "fixed right-5 top-50 z-50",
        "flex w-72 flex-col overflow-hidden",
        "rounded-xl border border-border bg-background/95 shadow-lg backdrop-blur-sm",
        "transition-all duration-200 ease-in-out"
      )}
    >
      {/* ── panel header ── */}
      <button
        type="button"
        onClick={() => setIsPanelExpanded((v) => !v)}
        className="flex w-full items-center gap-2.5 px-3 py-2.5 text-left hover:bg-accent/50 transition-colors"
      >
        <BrainCircuit size={14} className={cn(hasRunning && "text-[#4a9a9a]", !hasRunning && "text-muted-foreground")} />
        <span className="flex-1 text-xs font-semibold tracking-tight text-foreground">
          Monitor
        </span>
        {hasRunning && (
          <span className="flex h-4 min-w-4 items-center justify-center rounded-full bg-[#2F6868] px-1 text-[10px] font-medium text-white">
            {runningCount}
          </span>
        )}
        {!hasRunning && (
          <CheckCircle2 size={13} className="text-emerald-400" />
        )}
        {isPanelExpanded ? (
          <ChevronUp size={13} className="text-muted-foreground" />
        ) : (
          <ChevronDown size={13} className="text-muted-foreground" />
        )}
      </button>

      {/* ── task list ── */}
      {isPanelExpanded && (
        <div className="flex flex-col divide-y divide-border/60 border-t border-border/60">
          {tasks.map((task) => (
            <TaskRow
              key={task.tcId}
              task={task}
              isExpanded={expandedTaskIds.has(task.tcId)}
              onToggle={() => toggleTask(task.tcId)}
              onDismiss={() => onDismiss(task.tcId)}
              elapsedTick={tick}
            />
          ))}
        </div>
      )}
    </div>
  );
});

ZganMonitorPanel.displayName = "ZganMonitorPanel";
