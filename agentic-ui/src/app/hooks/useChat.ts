"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useStream } from "@langchain/langgraph-sdk/react";
import {
  type Message,
  type Assistant,
  type Checkpoint,
} from "@langchain/langgraph-sdk";
import { v4 as uuidv4 } from "uuid";
import type { UseStreamThread } from "@langchain/langgraph-sdk/react";
import type { TodoItem } from "@/app/types/types";
import { useClient } from "@/providers/ClientProvider";
import { useQueryState } from "nuqs";

export interface PipelineStep {
  step: string;
  label: string;
  status: "pending" | "running" | "completed" | "error" | "skipped";
  agent?: string;
  summary?: string;
}

export interface ModelEntry {
  model_name: string;
  auc: number;
  gini?: number;
  ks?: number;
  f1?: number;
  rank?: number;
}

export type StateType = {
  messages: Message[];
  todos: TodoItem[];
  files: Record<string, string>;
  email?: {
    id?: string;
    subject?: string;
    page_content?: string;
  };
  ui?: any;
  // Financial copilot state
  investigation_id?: string;
  dataset_name?: string;
  pipeline_type?: "credit_scoring" | "fraud_detection" | "general";
  pipeline_steps?: PipelineStep[];
  model_leaderboard?: ModelEntry[];
  best_auc?: number;
  charts_json?: string;
  charts_title?: string;
};

export function useChat({
  activeAssistant,
  onHistoryRevalidate,
  thread,
}: {
  activeAssistant: Assistant | null;
  onHistoryRevalidate?: () => void;
  thread?: UseStreamThread<StateType>;
}) {
  const [threadId, setThreadId] = useQueryState("threadId");
  const client = useClient();
  const [currentTodos, setCurrentTodos] = useState<TodoItem[]>([]);

  // ── Subagent message persistence ─────────────────────────────────────────
  const [subagentMessages, setSubagentMessages] = useState<Message[]>([]);
  const [subagentTagMap, setSubagentTagMap] = useState<Map<string, string>>(new Map());
  const streamingSnapshotRef = useRef<Message[]>([]);
  const eventCapturedRef = useRef<Message[]>([]);
  const messageSubagentTagRef = useRef<Map<string, string>>(new Map());
  const prevIsLoadingRef = useRef(false);

  // ── Live tool-call messages from updates ──────────────────────────────────
  // deepagents only writes messages to values at end of turn. We capture AI
  // messages from "model" node updates (which fire per tool call iteration) so
  // tool calls appear progressively in the UI. These are replaced when the
  // final values event arrives with the authoritative message list.
  const [liveUpdateMessages, setLiveUpdateMessages] = useState<Message[]>([]);
  const liveUpdateIdsRef = useRef<Set<string>>(new Set());

  const stream = useStream<StateType>({
    assistantId: activeAssistant?.assistant_id || "",
    client: client ?? undefined,
    reconnectOnMount: true,
    threadId: threadId ?? null,
    onThreadId: setThreadId,
    defaultHeaders: { "x-auth-scheme": "langsmith" },
    onFinish: (state) => {
      onHistoryRevalidate?.();
      setCurrentTodos(state.values.todos ?? []);
    },
    onError: onHistoryRevalidate,
    onCreated: onHistoryRevalidate,
    experimental_thread: thread,
    fetchStateHistory: true,
    onUpdateEvent: (data) => {
      for (const node of Object.keys(data)) {
        const update = data[node] as any;
        if (update?.todos) setCurrentTodos(update.todos);

        // Capture AI messages from root "model" node — these fire once per
        // tool-call iteration and contain the AI message with tool_calls.
        if (node === "model" && update?.messages && Array.isArray(update.messages)) {
          const newMsgs = (update.messages as Message[]).filter(
            (m) => m.type === "ai" && m.id && !liveUpdateIdsRef.current.has(m.id!)
          );
          if (newMsgs.length > 0) {
            newMsgs.forEach((m) => liveUpdateIdsRef.current.add(m.id!));
            setLiveUpdateMessages((prev) => [...prev, ...newMsgs]);
          }
        }

        // Capture messages from subgraph nodes (namespaced with "|")
        if (update?.messages && Array.isArray(update.messages) && node.includes("|")) {
          const subagentName = node.split("|")[0];
          const msgs = update.messages as Message[];
          const existing = new Set(eventCapturedRef.current.map((m) => m.id));
          const fresh = msgs.filter((m) => m.id && !existing.has(m.id));
          if (fresh.length > 0) {
            eventCapturedRef.current = [...eventCapturedRef.current, ...fresh];
            fresh.forEach((m) => {
              if (m.id) messageSubagentTagRef.current.set(m.id, subagentName);
            });
          }
        }
      }
    },
  });

  // ── Subagent message capture effects ────────────────────────────────────────

  // 1. Keep a rolling snapshot of stream.messages while streaming is active.
  //    We need this because when isLoading → false, stream.messages has already
  //    reverted to parent-only messages.
  useEffect(() => {
    if (stream.isLoading) {
      streamingSnapshotRef.current = [...stream.messages];
    }
  }, [stream.isLoading, stream.messages]);

  // 2. When streaming ends, diff the snapshot against the final parent messages
  //    to find orphaned (subagent) messages, then persist them in state.
  useEffect(() => {
    const wasLoading = prevIsLoadingRef.current;
    prevIsLoadingRef.current = stream.isLoading;

    if (wasLoading && !stream.isLoading) {
      const finalIds = new Set(
        (stream.values.messages ?? []).map((m) => m.id).filter(Boolean)
      );
      const snapshotOrphans = streamingSnapshotRef.current.filter(
        (m) => m.id && !finalIds.has(m.id)
      );
      const eventOrphans = eventCapturedRef.current.filter(
        (m) => m.id && !finalIds.has(m.id)
      );
      // Merge both sources, deduplicate by id
      const merged = new Map<string, Message>();
      [...snapshotOrphans, ...eventOrphans].forEach((m) => {
        if (m.id) merged.set(m.id, m);
      });

      if (merged.size > 0) {
        setSubagentMessages((prev) => {
          const prevIds = new Set(prev.map((m) => m.id).filter(Boolean));
          const truly_new = Array.from(merged.values()).filter(
            (m) => !prevIds.has(m.id!)
          );
          return truly_new.length > 0 ? [...prev, ...truly_new] : prev;
        });
        // Build and persist the tag map for newly captured messages
        const newTags = new Map<string, string>();
        merged.forEach((_, id) => {
          const tag = messageSubagentTagRef.current.get(id);
          if (tag) newTags.set(id, tag);
        });
        if (newTags.size > 0) {
          setSubagentTagMap((prev) => new Map([...prev, ...newTags]));
        }
      }
      // Reset event buffer for the next stream run
      eventCapturedRef.current = [];
    }
  }, [stream.isLoading]); // eslint-disable-line react-hooks/exhaustive-deps

  // Clear live update messages when streaming ends — stream.messages is now authoritative
  useEffect(() => {
    if (!stream.isLoading) {
      setLiveUpdateMessages([]);
      liveUpdateIdsRef.current = new Set();
    }
  }, [stream.isLoading]);

  // 3. Clear everything when the thread changes.
  useEffect(() => {
    setSubagentMessages([]);
    setSubagentTagMap(new Map());
    setLiveUpdateMessages([]);
    liveUpdateIdsRef.current = new Set();
    streamingSnapshotRef.current = [];
    eventCapturedRef.current = [];
    messageSubagentTagRef.current = new Map();
    prevIsLoadingRef.current = false;
  }, [threadId]);

  // ── Todos sync ────────────────────────────────────────────────────────────
  // Sync internal todos with stream values when not streaming or when root state changes
  useEffect(() => {
    if (!stream.isLoading && stream.values.todos) {
      setCurrentTodos(stream.values.todos);
    }
  }, [stream.isLoading, stream.values.todos]);

  const sendMessage = useCallback(
    (content: string) => {
      const newMessage: Message = { id: uuidv4(), type: "human", content };
      stream.submit(
        { messages: [newMessage] },
        {
          optimisticValues: (prev) => ({
            messages: [...(prev.messages ?? []), newMessage],
          }),
          config: {
            ...(activeAssistant?.config ?? {}),
            recursion_limit: 100,
          },
          streamSubgraphs: true,
        }
      );
      // Update thread list immediately when sending a message
      onHistoryRevalidate?.();
    },
    [stream, activeAssistant?.config, onHistoryRevalidate]
  );

  const runSingleStep = useCallback(
    (
      messages: Message[],
      checkpoint?: Checkpoint,
      isRerunningSubagent?: boolean,
      optimisticMessages?: Message[]
    ) => {
      if (checkpoint) {
        stream.submit(undefined, {
          ...(optimisticMessages
            ? { optimisticValues: { messages: optimisticMessages } }
            : {}),
          config: { ...(activeAssistant?.config ?? {}) },
          checkpoint: checkpoint,
          ...(isRerunningSubagent
            ? { interruptAfter: ["tools"] }
            : { interruptBefore: ["tools"] }),
          streamSubgraphs: true,
        });
      } else {
        stream.submit(
          { messages },
          {
            config: { ...(activeAssistant?.config ?? {}) },
            interruptBefore: ["tools"],
            streamSubgraphs: true,
          }
        );
      }
    },
    [stream, activeAssistant?.config]
  );

  const setFiles = useCallback(
    async (files: Record<string, string>) => {
      if (!threadId) return;
      // TODO: missing a way how to revalidate the internal state
      // I think we do want to have the ability to externally manage the state
      await client.threads.updateState(threadId, { values: { files } });
    },
    [client, threadId]
  );

  const continueStream = useCallback(
    (hasTaskToolCall?: boolean) => {
      stream.submit(undefined, {
        config: {
          ...(activeAssistant?.config || {}),
          recursion_limit: 100,
        },
        ...(hasTaskToolCall
          ? { interruptAfter: ["tools"] }
          : { interruptBefore: ["tools"] }),
        streamSubgraphs: true,
      });
      // Update thread list when continuing stream
      onHistoryRevalidate?.();
    },
    [stream, activeAssistant?.config, onHistoryRevalidate]
  );

  const markCurrentThreadAsResolved = useCallback(() => {
    stream.submit(null, { command: { goto: "__end__", update: null } });
    // Update thread list when marking thread as resolved
    onHistoryRevalidate?.();
  }, [stream, onHistoryRevalidate]);

  const resumeInterrupt = useCallback(
    (value: any) => {
      stream.submit(null, {
        command: { resume: value },
        config: {
          ...(activeAssistant?.config || {}),
          recursion_limit: 100,
        },
      });
      // Update thread list when resuming from interrupt
      onHistoryRevalidate?.();
    },
    [stream, activeAssistant?.config, onHistoryRevalidate]
  );

  const stopStream = useCallback(() => {
    stream.stop();
  }, [stream]);

  return {
    stream,
    todos: currentTodos,
    files: stream.values.files ?? {},
    email: stream.values.email,
    ui: stream.values.ui,
    setFiles,
    // During streaming, merge live update messages (from "model" node updates)
    // with stream.messages. These disappear once streaming ends and stream.messages
    // becomes authoritative with the final committed state.
    messages: stream.isLoading
      ? (() => {
          const streamIds = new Set(stream.messages.map((m) => m.id).filter(Boolean));
          const extra = liveUpdateMessages.filter((m) => !streamIds.has(m.id!));
          return [...stream.messages, ...extra];
        })()
      : stream.messages,
    subagentMessages,
    subagentTagMap,
    isLoading: stream.isLoading,
    isThreadLoading: stream.isThreadLoading,
    interrupt: stream.interrupt,
    getMessagesMetadata: stream.getMessagesMetadata,
    sendMessage,
    runSingleStep,
    continueStream,
    stopStream,
    markCurrentThreadAsResolved,
    resumeInterrupt,
    // Financial copilot state — read directly from stream.values
    investigationId: stream.values.investigation_id,
    datasetName: stream.values.dataset_name,
    pipelineType: stream.values.pipeline_type,
    pipelineSteps: stream.values.pipeline_steps,
    modelLeaderboard: stream.values.model_leaderboard,
    bestAuc: stream.values.best_auc,
    chartsJson: stream.values.charts_json,
    chartsTitle: stream.values.charts_title,
  };
}
