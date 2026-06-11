import { Message } from "@langchain/langgraph-sdk";
import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";
import type { ToolCall } from "@/app/types/types";

// ─── Subagent message utilities ───────────────────────────────────────────────

/** Maps tool name → the subagent that owns it. */
export const TOOL_TO_SUBAGENT: Record<string, string> = {
  create_model: "train-model-agent",
  run_model: "train-model-agent",
  get_model_report_metrics: "train-model-agent",
  run_sfa: "sfa-agent",
  get_sfa_scores: "sfa-agent",
  get_sfa_list: "sfa-agent",
  feature_selection: "sfa-agent",
  get_features: "sfa-agent",
  deploy_model: "deploy-agent",
  get_model_api_info: "deploy-agent",
  run_script: "eda-agent",
};

/**
 * Tool that marks the START of a new invocation session for each subagent.
 * When we see this tool in an orphaned AI message we begin a new session bucket.
 */
const SESSION_START_TOOL: Record<string, string> = {
  "train-model-agent": "create_model",
  "sfa-agent": "run_sfa",
  "deploy-agent": "deploy_model",
  "eda-agent": "run_script",
};

/** Extract the raw tool-call list from an AI message regardless of format. */
export function extractToolCallsFromAIMsg(
  msg: Message
): Array<{ id?: string; name: string; args: unknown }> {
  const out: Array<{ id?: string; name: string; args: unknown }> = [];

  if (
    (msg as any).additional_kwargs?.tool_calls &&
    Array.isArray((msg as any).additional_kwargs.tool_calls)
  ) {
    for (const tc of (msg as any).additional_kwargs.tool_calls) {
      out.push({
        id: tc.id,
        name: tc.function?.name ?? tc.name ?? "unknown",
        args: tc.function?.arguments ?? tc.args ?? {},
      });
    }
  } else if ((msg as any).tool_calls && Array.isArray((msg as any).tool_calls)) {
    for (const tc of (msg as any).tool_calls) {
      if (tc.name && tc.name !== "") {
        out.push({ id: tc.id, name: tc.name, args: tc.args ?? {} });
      }
    }
  } else if (Array.isArray(msg.content)) {
    for (const block of msg.content as any[]) {
      if (block?.type === "tool_use") {
        out.push({ id: block.id, name: block.name ?? "unknown", args: block.input ?? {} });
      }
    }
  }

  return out;
}

/**
 * Convert an ordered list of orphaned messages (AI + tool-result) into
 * per-subagent invocation buckets.
 *
 * Returns: Map<subagentName, Message[][]>
 *   outer array index = invocation number (0-based)
 */
export function groupOrphanedMessagesBySubagent(
  orphaned: Message[],
  tagMap: Map<string, string> = new Map()
): Map<string, Message[][]> {
  const result = new Map<string, Message[][]>();
  // Track the currently-open session bucket per subagent
  const openSession = new Map<string, Message[]>();

  for (const msg of orphaned) {
    if (msg.type === "ai") {
      const tcs = extractToolCallsFromAIMsg(msg);
      let agentName: string | null = null;
      let isStart = false;

      // Primary: use node-namespace tag (accurate, mode-agnostic)
      const tag = msg.id ? tagMap.get(msg.id) : undefined;
      if (tag) {
        agentName = tag;
        isStart = !openSession.has(agentName) ||
          tcs.some((tc) => SESSION_START_TOOL[agentName!] === tc.name);
      } else {
        // Fallback: infer from tool names
        for (const tc of tcs) {
          const agent = TOOL_TO_SUBAGENT[tc.name];
          if (agent) {
            agentName = agent;
            isStart = SESSION_START_TOOL[agent] === tc.name;
            break;
          }
        }
      }

      if (!agentName) continue; // think_tool / authenticate — skip

      if (isStart || !openSession.has(agentName)) {
        const bucket: Message[] = [msg];
        openSession.set(agentName, bucket);
        if (!result.has(agentName)) result.set(agentName, []);
        result.get(agentName)!.push(bucket);
      } else {
        openSession.get(agentName)!.push(msg);
      }
    } else if (msg.type === "tool") {
      const toolCallId = (msg as any).tool_call_id as string | undefined;
      if (!toolCallId) continue;

      // Append the result to whichever open session owns that tool_call_id
      for (const session of openSession.values()) {
        const owns = session.some(
          (m) =>
            m.type === "ai" &&
            extractToolCallsFromAIMsg(m).some((tc) => tc.id === toolCallId)
        );
        if (owns) {
          session.push(msg);
          break;
        }
      }
    }
  }

  return result;
}

/**
 * Given a session (ordered array of AI + tool-result messages), return a flat
 * ToolCall list with status and result already resolved.
 */
export function sessionToToolCalls(session: Message[]): ToolCall[] {
  const resultMap = new Map<string, string>();
  for (const m of session) {
    if (m.type === "tool" && (m as any).tool_call_id) {
      resultMap.set((m as any).tool_call_id, extractStringFromMessageContent(m));
    }
  }

  const toolCalls: ToolCall[] = [];
  for (const m of session) {
    if (m.type !== "ai") continue;
    for (const tc of extractToolCallsFromAIMsg(m)) {
      const rawArgs = tc.args;
      const parsedArgs: Record<string, unknown> =
        typeof rawArgs === "string"
          ? (() => { try { return JSON.parse(rawArgs); } catch { return { raw: rawArgs }; } })()
          : (rawArgs as Record<string, unknown>) ?? {};

      toolCalls.push({
        id: tc.id ?? `orphan-${Math.random()}`,
        name: tc.name,
        args: parsedArgs,
        result: tc.id ? resultMap.get(tc.id) : undefined,
        status: tc.id && resultMap.has(tc.id) ? "completed" : "pending",
      });
    }
  }
  return toolCalls;
}

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function extractStringFromMessageContent(message: Message): string {
  return typeof message.content === "string"
    ? message.content
    : Array.isArray(message.content)
      ? message.content
        .filter(
          (c: unknown) =>
            (typeof c === "object" &&
              c !== null &&
              "type" in c &&
              (c as { type: string }).type === "text") ||
            typeof c === "string"
        )
        .map((c: unknown) =>
          typeof c === "string"
            ? c
            : typeof c === "object" && c !== null && "text" in c
              ? (c as { text?: string }).text || ""
              : ""
        )
        .join("")
      : "";
}

export function extractSubAgentContent(data: unknown): string {
  if (typeof data === "string") {
    return data;
  }

  if (data && typeof data === "object") {
    const dataObj = data as Record<string, unknown>;

    // Try to extract description first
    if (dataObj.description && typeof dataObj.description === "string") {
      return dataObj.description;
    }

    // Then try prompt
    if (dataObj.prompt && typeof dataObj.prompt === "string") {
      return dataObj.prompt;
    }

    // For output objects, try result
    if (dataObj.result && typeof dataObj.result === "string") {
      return dataObj.result;
    }

    // Fallback to JSON stringification
    return JSON.stringify(data, null, 2);
  }

  // Fallback for any other type
  return JSON.stringify(data, null, 2);
}

export function isPreparingToCallTaskTool(messages: Message[]): boolean {
  const lastMessage = messages[messages.length - 1];
  return (
    (lastMessage.type === "ai" &&
      lastMessage.tool_calls?.some(
        (call: { name?: string }) => call.name === "task"
      )) ||
    false
  );
}

export function formatMessageForLLM(message: Message): string {
  let role: string;
  if (message.type === "human") {
    role = "Human";
  } else if (message.type === "ai") {
    role = "Assistant";
  } else if (message.type === "tool") {
    role = `Tool Result`;
  } else {
    role = message.type || "Unknown";
  }

  const timestamp = message.id ? ` (${message.id.slice(0, 8)})` : "";

  let contentText = "";

  // Extract content text
  if (typeof message.content === "string") {
    contentText = message.content;
  } else if (Array.isArray(message.content)) {
    const textParts: string[] = [];

    message.content.forEach((part: any) => {
      if (typeof part === "string") {
        textParts.push(part);
      } else if (part && typeof part === "object" && part.type === "text") {
        textParts.push(part.text || "");
      }
      // Ignore other types like tool_use in content - we handle tool calls separately
    });

    contentText = textParts.join("\n\n").trim();
  }

  // For tool messages, include additional tool metadata
  if (message.type === "tool") {
    const toolName = (message as any).name || "unknown_tool";
    const toolCallId = (message as any).tool_call_id || "";
    role = `Tool Result [${toolName}]`;
    if (toolCallId) {
      role += ` (call_id: ${toolCallId.slice(0, 8)})`;
    }
  }

  // Handle tool calls from .tool_calls property (for AI messages)
  const toolCallsText: string[] = [];
  if (
    message.type === "ai" &&
    message.tool_calls &&
    Array.isArray(message.tool_calls) &&
    message.tool_calls.length > 0
  ) {
    message.tool_calls.forEach((call: any) => {
      const toolName = call.name || "unknown_tool";
      const toolArgs = call.args ? JSON.stringify(call.args, null, 2) : "{}";
      toolCallsText.push(`[Tool Call: ${toolName}]\nArguments: ${toolArgs}`);
    });
  }

  // Combine content and tool calls
  const parts: string[] = [];
  if (contentText) {
    parts.push(contentText);
  }
  if (toolCallsText.length > 0) {
    parts.push(...toolCallsText);
  }

  if (parts.length === 0) {
    return `${role}${timestamp}: [Empty message]`;
  }

  if (parts.length === 1) {
    return `${role}${timestamp}: ${parts[0]}`;
  }

  return `${role}${timestamp}:\n${parts.join("\n\n")}`;
}

export function formatConversationForLLM(messages: Message[]): string {
  const formattedMessages = messages.map(formatMessageForLLM);
  return formattedMessages.join("\n\n---\n\n");
}
