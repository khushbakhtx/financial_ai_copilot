"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { authenticate, checkZganTaskStatus } from "@/app/utils/mlApi";

export interface ZganLog {
  timestamp: Date;
  message: string;
  type: "info" | "success" | "error";
}

export type ZganTaskStatus = "queued" | "running" | "done" | "failed";

export interface ZganTask {
  tcId: string;
  task_id: string;
  zgan_model_id: string | null;
  label: string;
  startedAt: Date;
  status: ZganTaskStatus;
  logs: ZganLog[];
}

interface UseZganMonitorOptions {
  threadId: string | null;
  onComplete: (task: ZganTask) => void;
}

const POLL_INTERVAL_MS = 30_000;
const INITIAL_POLL_DELAY_MS = 8_000;

function log(message: string, type: ZganLog["type"] = "info"): ZganLog {
  return { timestamp: new Date(), message, type };
}

export function useZganMonitor({ threadId, onComplete }: UseZganMonitorOptions) {
  const [activeTasks, setActiveTasks] = useState<ZganTask[]>([]);
  const intervalRefs = useRef<Map<string, ReturnType<typeof setInterval>>>(new Map());
  const timeoutRefs = useRef<Map<string, ReturnType<typeof setTimeout>>>(new Map());
  // Guard against registering the same tool call twice (effect runs on every processedMessages change)
  const seenTcIds = useRef<Set<string>>(new Set());
  const tokenRef = useRef<string | null>(null);
  // Guard against calling onComplete more than once per task
  const notifiedRef = useRef<Set<string>>(new Set());

  const getToken = useCallback(async (forceRefresh = false): Promise<string | null> => {
    if (!forceRefresh && tokenRef.current) return tokenRef.current;
    tokenRef.current = await authenticate();
    return tokenRef.current;
  }, []);

  const stopPolling = useCallback((tcId: string) => {
    const iv = intervalRefs.current.get(tcId);
    if (iv) { clearInterval(iv); intervalRefs.current.delete(tcId); }
    const to = timeoutRefs.current.get(tcId);
    if (to) { clearTimeout(to); timeoutRefs.current.delete(tcId); }
  }, []);

  const updateTask = useCallback(
    (tcId: string, patch: Partial<ZganTask>, extraLog?: ZganLog) => {
      setActiveTasks((prev) =>
        prev.map((t) =>
          t.tcId !== tcId
            ? t
            : { ...t, ...patch, logs: extraLog ? [...t.logs, extraLog] : t.logs }
        )
      );
    },
    []
  );

  const startPolling = useCallback(
    (tcId: string, task_id: string) => {
      let retrying401 = false;

      const poll = async () => {
        const token = await getToken(retrying401);
        retrying401 = false;

        if (!token) {
          updateTask(tcId, {}, log("Could not get auth token — will retry", "info"));
          return;
        }

        const result = await checkZganTaskStatus(task_id, token);

        if (!result) {
          // 401 or network error — force token refresh and retry once after a short delay
          tokenRef.current = null;
          retrying401 = true;
          updateTask(tcId, {}, log("Auth expired — refreshing token…", "info"));
          setTimeout(() => poll(), 3_000);
          return;
        }

        const apiStatus = result.status;

        if (apiStatus === "done") {
          stopPolling(tcId);
          setActiveTasks((prev) =>
            prev.map((t) =>
              t.tcId !== tcId
                ? t
                : { ...t, status: "done", logs: [...t.logs, log("Training complete — model is ready!", "success")] }
            )
          );
        } else if (apiStatus === "failed") {
          stopPolling(tcId);
          const errorDetail =
            (result.result as Record<string, unknown> | undefined)?.error as string | undefined;
          const errorMsg = errorDetail ? `Training failed: ${errorDetail}` : "Training failed.";
          setActiveTasks((prev) =>
            prev.map((t) =>
              t.tcId !== tcId
                ? t
                : { ...t, status: "failed", logs: [...t.logs, log(errorMsg, "error")] }
            )
          );
        } else if (apiStatus === "queued") {
          updateTask(
            tcId,
            { status: "queued" },
            log("Job is queued — waiting to start…", "info")
          );
        } else {
          // "running" or any other intermediate
          updateTask(
            tcId,
            { status: "running" },
            log("Training in progress…", "info")
          );
        }
      };

      const to = setTimeout(poll, INITIAL_POLL_DELAY_MS);
      const iv = setInterval(poll, POLL_INTERVAL_MS);
      timeoutRefs.current.set(tcId, to);
      intervalRefs.current.set(tcId, iv);
    },
    [getToken, stopPolling, updateTask]
  );

  /**
   * Register a task only after create_zgan confirms status "started" with a valid task_id.
   * Calling this with the same tcId twice is a no-op.
   */
  const addTask = useCallback(
    (tcId: string, task_id: string, zgan_model_id: string | null, label: string) => {
      if (seenTcIds.current.has(tcId)) return;
      seenTcIds.current.add(tcId);

      setActiveTasks((prev) => [
        ...prev,
        {
          tcId,
          task_id,
          zgan_model_id,
          label,
          startedAt: new Date(),
          status: "queued",
          logs: [log(`Job accepted (${task_id.slice(0, 8)}…) — polling every 30s`, "info")],
        },
      ]);

      startPolling(tcId, task_id);
    },
    [startPolling]
  );

  const dismissTask = useCallback(
    (tcId: string) => {
      stopPolling(tcId);
      setActiveTasks((prev) => prev.filter((t) => t.tcId !== tcId));
    },
    [stopPolling]
  );

  // Call onComplete outside of any setState updater to avoid the React
  // "Cannot update a component while rendering a different component" error.
  useEffect(() => {
    activeTasks.forEach((task) => {
      if (
        (task.status === "done" || task.status === "failed") &&
        !notifiedRef.current.has(task.tcId)
      ) {
        notifiedRef.current.add(task.tcId);
        onComplete(task);
      }
    });
  }, [activeTasks, onComplete]);

  // When the user switches threads, stop all polling and clear monitor state.
  // Each thread gets its own independent monitor view.
  useEffect(() => {
    intervalRefs.current.forEach((iv) => clearInterval(iv));
    timeoutRefs.current.forEach((to) => clearTimeout(to));
    intervalRefs.current.clear();
    timeoutRefs.current.clear();
    seenTcIds.current.clear();
    notifiedRef.current.clear();
    tokenRef.current = null;
    setActiveTasks([]);
  }, [threadId]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      intervalRefs.current.forEach((iv) => clearInterval(iv));
      timeoutRefs.current.forEach((to) => clearTimeout(to));
    };
  }, []);

  return { activeTasks, addTask, dismissTask };
}
