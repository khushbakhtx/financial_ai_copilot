"use client";

import React, { useMemo, useState, useCallback } from "react";
import { SubAgentIndicator } from "@/app/components/SubAgentIndicator";
import { ToolCallBox } from "@/app/components/ToolCallBox";
import { ThinkingBlock } from "@/app/components/ThinkingBlock";
import { ScriptExecutionBlock } from "@/app/components/ScriptExecutionBlock";
import { ModelComparisonChart } from "@/app/components/ModelComparisonChart";
import { MarkdownContent } from "@/app/components/MarkdownContent";
import type {
  SubAgent,
  ToolCall,
  ActionRequest,
  ReviewConfig,
} from "@/app/types/types";
import { Message } from "@langchain/langgraph-sdk";
import {
  groupOrphanedMessagesBySubagent,
  sessionToToolCalls,
} from "@/app/utils/utils";
import {
  Square,
  ArrowUp,
  CheckCircle,
  Clock,
  Circle,
  FileIcon,
} from "lucide-react";
import {
  extractSubAgentContent,
  extractStringFromMessageContent,
} from "@/app/utils/utils";
import { cn } from "@/lib/utils";

interface ChatMessageProps {
  message: Message;
  toolCalls: ToolCall[];
  isLoading?: boolean;
  actionRequestsMap?: Map<string, ActionRequest>;
  reviewConfigsMap?: Map<string, ReviewConfig>;
  ui?: any[];
  stream?: any;
  onResumeInterrupt?: (value: any) => void;
  graphId?: string;
  subagentMessages?: Message[];
  subagentTagMap?: Map<string, string>;
}

export const ChatMessage = React.memo<ChatMessageProps>(
  ({
    message,
    toolCalls,
    isLoading,
    actionRequestsMap,
    reviewConfigsMap,
    ui,
    stream,
    onResumeInterrupt,
    graphId,
    subagentMessages,
    subagentTagMap,
  }) => {
    const isUser = message.type === "human";
    const rawContent = extractStringFromMessageContent(message);

    // System notifications are sent as human messages so the agent receives them,
    // but they should never appear as user chat bubbles in the UI.
    if (isUser && rawContent.trimStart().startsWith("[SYSTEM NOTIFICATION]")) {
      return null;
    }

    const { messageContent, fileAttachment } = useMemo(() => {
      const attachmentRegex = /\[ATTACHMENT: ({[\s\S]*?})\]/;
      const match = rawContent.match(attachmentRegex);

      if (match) {
        try {
          const metadata = JSON.parse(match[1]);
          const cleanedContent = rawContent.replace(attachmentRegex, "").trim();
          return {
            messageContent: cleanedContent,
            fileAttachment: {
              filename: metadata.filename || "Attached File",
              type: metadata.type as "blind" | "train",
            }
          };
        } catch (e) {
          console.error("Failed to parse attachment metadata:", e);
        }
      }

      return { messageContent: rawContent, fileAttachment: null };
    }, [rawContent]);

    const hasContent = messageContent && messageContent.trim() !== "";
    const hasToolCalls = toolCalls.length > 0;

    const subAgents = useMemo(() => {
      return toolCalls
        .filter((toolCall: ToolCall) => {
          return (
            toolCall.name === "task" &&
            toolCall.args["subagent_type"] &&
            toolCall.args["subagent_type"] !== "" &&
            toolCall.args["subagent_type"] !== null
          );
        })
        .map((toolCall: ToolCall) => {
          const subagentType = (toolCall.args as Record<string, unknown>)[
            "subagent_type"
          ] as string;
          return {
            id: toolCall.id,
            name: toolCall.name,
            subAgentName: subagentType,
            input: toolCall.args,
            output: toolCall.result ? { result: toolCall.result } : undefined,
            status: toolCall.status,
          } as SubAgent;
        });
    }, [toolCalls]);

    // Group orphaned subagent messages into per-subagent invocation sessions.
    // Map: subagentName → Message[][] (outer = invocation index, inner = messages)
    const subagentSessions = useMemo(
      () =>
        subagentMessages && subagentMessages.length > 0
          ? groupOrphanedMessagesBySubagent(subagentMessages, subagentTagMap)
          : new Map<string, Message[][]>(),
      [subagentMessages, subagentTagMap]
    );

    const [expandedSubAgents, setExpandedSubAgents] = useState<
      Record<string, boolean>
    >({});
    const isSubAgentExpanded = useCallback(
      (id: string) => expandedSubAgents[id] ?? true,
      [expandedSubAgents]
    );
    const toggleSubAgent = useCallback((id: string) => {
      setExpandedSubAgents((prev) => ({
        ...prev,
        [id]: prev[id] === undefined ? false : !prev[id],
      }));
    }, []);

    return (
      <div
        className={cn(
          "flex w-full max-w-full overflow-x-hidden",
          isUser && "flex-row-reverse"
        )}
      >
        <div
          className={cn(
            "min-w-0 max-w-full",
            isUser ? "max-w-[70%]" : "w-full"
          )}
        >
          {(hasContent || fileAttachment) && (
            <div className={cn("relative flex items-end gap-0")}>
              <div
                className={cn(
                  "mt-4 overflow-hidden break-words text-sm font-normal leading-[150%]",
                  isUser
                    ? "rounded-xl rounded-br-none border border-border px-3 py-2 text-foreground"
                    : "text-primary"
                )}
                style={
                  isUser
                    ? { backgroundColor: "var(--color-user-message-bg)" }
                    : undefined
                }
              >
                {isUser ? (
                  <div className="flex flex-col gap-2">
                    {hasContent && (
                      <p className="m-0 whitespace-pre-wrap break-words text-sm leading-relaxed">
                        {messageContent}
                      </p>
                    )}
                    {fileAttachment && (
                      <div className="flex items-center gap-2 rounded-md bg-foreground/5 px-2 py-1.5 text-xs">
                        <FileIcon size={14} className="text-foreground/70" />
                        <span className="max-w-[180px] truncate font-medium">
                          {fileAttachment.filename}
                        </span>
                        <span className="rounded bg-foreground/10 px-1 py-0.5 text-[10px] uppercase opacity-70">
                          {fileAttachment.type}
                        </span>
                      </div>
                    )}
                  </div>
                ) : hasContent ? (
                  <MarkdownContent content={messageContent} />
                ) : null}
              </div>
            </div>
          )}
          {hasToolCalls && (
            <div className="mt-4 flex w-full flex-col">
              {toolCalls.map((toolCall: ToolCall) => {
                if (toolCall.name === "task") return null;

                // Special rendering for think_tool
                if (toolCall.name === "think_tool") {
                  const reflection = typeof toolCall.args === 'object' && toolCall.args !== null
                    ? (toolCall.args as any).reflection || ""
                    : "";
                  return (
                    <ThinkingBlock
                      key={toolCall.id}
                      content={reflection}
                      status={toolCall.status}
                    />
                  );
                }

                // Special rendering for run_script
                if (toolCall.name === "run_script") {
                  return (
                    <ScriptExecutionBlock
                      key={toolCall.id}
                      toolCall={toolCall}
                    />
                  );
                }

                // Special rendering for plot_model_comparison
                if (toolCall.name === "plot_model_comparison") {
                  return (
                    <ModelComparisonChart
                      key={toolCall.id}
                      toolCall={toolCall}
                    />
                  );
                }

                const toolCallGenUiComponent = ui?.find(
                  (u) => u.metadata?.tool_call_id === toolCall.id
                );
                const actionRequest = actionRequestsMap?.get(toolCall.name);
                const reviewConfig = reviewConfigsMap?.get(toolCall.name);
                return (
                  <ToolCallBox
                    key={toolCall.id}
                    toolCall={toolCall}
                    uiComponent={toolCallGenUiComponent}
                    stream={stream}
                    graphId={graphId}
                    actionRequest={actionRequest}
                    reviewConfig={reviewConfig}
                    onResume={onResumeInterrupt}
                    isLoading={isLoading}
                  />
                );
              })}
            </div>
          )}
          {!isUser && subAgents.length > 0 && (
            <div className="flex w-fit max-w-full flex-col gap-4">
              {subAgents.map((subAgent) => (
                <div
                  key={subAgent.id}
                  className="flex w-full flex-col gap-2"
                >
                  <div className="flex items-end gap-2">
                    <div className="w-[calc(100%-100px)]">
                      <SubAgentIndicator
                        subAgent={subAgent}
                        onClick={() => toggleSubAgent(subAgent.id)}
                        isExpanded={isSubAgentExpanded(subAgent.id)}
                      />
                    </div>
                  </div>
                  {isSubAgentExpanded(subAgent.id) && (
                    <div className="w-full max-w-full">
                      <div className="bg-surface border-border-light rounded-md border p-4">
                        <h4 className="text-primary/70 mb-2 text-xs font-semibold uppercase tracking-wider">
                          Input
                        </h4>
                        <div className="mb-4">
                          <MarkdownContent
                            content={extractSubAgentContent(subAgent.input)}
                          />
                        </div>

                        {/* Subagent internal tool calls (persisted from streaming) */}
                        {(() => {
                          const agentName = subAgent.subAgentName;
                          const sessions = subagentSessions.get(agentName) ?? [];
                          // Which invocation of this subagent is this indicator?
                          const invocationIndex = subAgents
                            .slice(0, subAgents.findIndex((s) => s.id === subAgent.id))
                            .filter((s) => s.subAgentName === agentName).length;
                          const session = sessions[invocationIndex] ?? sessions[0];
                          if (!session || session.length === 0) return null;

                          const tcs = sessionToToolCalls(session);
                          if (tcs.length === 0) return null;

                          return (
                            <div className="mb-4">
                              <h4 className="text-primary/70 mb-2 text-xs font-semibold uppercase tracking-wider">
                                Execution
                              </h4>
                              <div className="flex flex-col gap-2">
                                {tcs.map((tc) => {
                                  if (tc.name === "think_tool") {
                                    const reflection =
                                      typeof tc.args === "object" && tc.args !== null
                                        ? (tc.args as any).reflection ?? ""
                                        : "";
                                    return (
                                      <ThinkingBlock
                                        key={tc.id}
                                        content={reflection}
                                        status={tc.status}
                                      />
                                    );
                                  }
                                  if (tc.name === "run_script") {
                                    return (
                                      <ScriptExecutionBlock
                                        key={tc.id}
                                        toolCall={tc}
                                      />
                                    );
                                  }
                                  return (
                                    <ToolCallBox
                                      key={tc.id}
                                      toolCall={tc}
                                      isLoading={false}
                                    />
                                  );
                                })}
                              </div>
                            </div>
                          );
                        })()}

                        {subAgent.output && (
                          <>
                            <h4 className="text-primary/70 mb-2 text-xs font-semibold uppercase tracking-wider">
                              Output
                            </h4>
                            <MarkdownContent
                              content={extractSubAgentContent(subAgent.output)}
                            />
                          </>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    );
  }
);

ChatMessage.displayName = "ChatMessage";
