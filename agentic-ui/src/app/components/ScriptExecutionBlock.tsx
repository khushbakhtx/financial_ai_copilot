"use client";

import React, { useState, useMemo } from "react";
import { Terminal, ChevronDown, ChevronUp, CheckCircle2, Loader2, XCircle } from "lucide-react";
import { cn } from "@/lib/utils";
import type { ToolCall } from "@/app/types/types";

interface ScriptExecutionBlockProps {
  toolCall: ToolCall;
}

/** Extract the short script filename from "data-analysis/discover" → "discover" */
function parseScriptName(scriptName: string): { skill: string; script: string } {
  const parts = scriptName.split("/");
  return parts.length >= 2
    ? { skill: parts[0], script: parts[1] }
    : { skill: "", script: scriptName };
}

/** Try to parse JSON and pretty-print with basic syntax coloring tokens */
function parseOutput(raw: string): { isJson: boolean; content: string } {
  if (!raw) return { isJson: false, content: "" };
  try {
    const parsed = JSON.parse(raw.trim());
    return { isJson: true, content: JSON.stringify(parsed, null, 2) };
  } catch {
    return { isJson: false, content: raw };
  }
}

/** Tokenize JSON string into colored spans */
function JsonView({ content }: { content: string }) {
  const tokens = useMemo(() => {
    const lines = content.split("\n");
    return lines.map((line, li) => {
      // Simple tokenizer: keys, strings, numbers, booleans, null
      const parts: { text: string; cls: string }[] = [];
      let rest = line;
      const tokenRe =
        /("(?:[^"\\]|\\.)*")(\s*:)?|(true|false|null)|(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)|([{}\[\],])/g;
      let match: RegExpExecArray | null;
      let lastIdx = 0;
      tokenRe.lastIndex = 0;
      while ((match = tokenRe.exec(rest)) !== null) {
        if (match.index > lastIdx) {
          parts.push({ text: rest.slice(lastIdx, match.index), cls: "text-zinc-400" });
        }
        if (match[1] !== undefined) {
          const isKey = match[2] !== undefined;
          parts.push({ text: match[1], cls: isKey ? "text-sky-300" : "text-emerald-300" });
          if (match[2]) parts.push({ text: match[2], cls: "text-zinc-400" });
        } else if (match[3] !== undefined) {
          parts.push({ text: match[3], cls: "text-amber-300" });
        } else if (match[4] !== undefined) {
          parts.push({ text: match[4], cls: "text-violet-300" });
        } else if (match[5] !== undefined) {
          parts.push({ text: match[5], cls: "text-zinc-500" });
        }
        lastIdx = match.index + match[0].length;
      }
      if (lastIdx < rest.length) {
        parts.push({ text: rest.slice(lastIdx), cls: "text-zinc-400" });
      }
      return (
        <div key={li} className="leading-5">
          {parts.map((p, pi) => (
            <span key={pi} className={p.cls}>{p.text}</span>
          ))}
        </div>
      );
    });
  }, [content]);

  return <>{tokens}</>;
}

export const ScriptExecutionBlock = React.memo<ScriptExecutionBlockProps>(
  ({ toolCall }) => {
    const [isExpanded, setIsExpanded] = useState(true);

    const scriptName = (toolCall.args?.script_name as string) ?? "";
    const { skill, script } = parseScriptName(scriptName);
    const isPending = toolCall.status === "pending";
    const isError = toolCall.status === "error";

    const { isJson, content } = useMemo(
      () => parseOutput(toolCall.result ?? ""),
      [toolCall.result]
    );

    const hasOutput = !!toolCall.result;

    return (
      <div className="my-2 w-full overflow-hidden rounded-lg border border-zinc-800 bg-[#181818]">
        {/* Header */}
        <button
          onClick={() => setIsExpanded((v) => !v)}
          className="flex w-full items-center justify-between px-4 py-2.5 text-left transition-colors hover:bg-zinc-900"
        >
          <div className="flex items-center gap-2.5">
            {/* Status icon */}
            {isPending ? (
              <Loader2 size={14} className="animate-spin text-emerald-400" />
            ) : isError ? (
              <XCircle size={14} className="text-red-400" />
            ) : (
              <CheckCircle2 size={14} className="text-emerald-400" />
            )}

            {/* Terminal icon */}
            <div className="flex h-5 w-5 items-center justify-center rounded bg-zinc-900">
              <Terminal size={11} className="text-emerald-400" />
            </div>

            {/* Script label */}
            <div className="flex items-center gap-1.5 font-mono text-xs">
              {skill && (
                <>
                  <span className="text-zinc-500">{skill}/</span>
                </>
              )}
              <span className="font-semibold text-emerald-300">{script || scriptName}</span>
              <span className="text-zinc-600">.py</span>
            </div>

            {/* Running badge */}
            {isPending && (
              <span className="rounded-full bg-emerald-500/10 px-2 py-0.5 text-[10px] font-medium text-emerald-400">
                running
              </span>
            )}
          </div>

          {hasOutput && (
            isExpanded
              ? <ChevronUp size={14} className="text-zinc-500" />
              : <ChevronDown size={14} className="text-zinc-500" />
          )}
        </button>

        {/* Output panel */}
        {isExpanded && hasOutput && (
          <div className="border-t border-zinc-800">
            {/* Console bar */}
            <div className="flex items-center gap-1.5 border-b border-zinc-800 bg-zinc-950 px-4 py-1.5">
              <div className="h-2 w-2 rounded-full bg-red-500/70" />
              <div className="h-2 w-2 rounded-full bg-yellow-500/70" />
              <div className="h-2 w-2 rounded-full bg-emerald-500/70" />
              <span className="ml-2 font-mono text-[10px] text-zinc-500">
                {isJson ? "json output" : "stdout"}
              </span>
              {isError && (
                <span className="ml-auto font-mono text-[10px] text-red-400">exit 1</span>
              )}
            </div>

            {/* Content */}
            <div className="max-h-96 overflow-y-auto bg-[#181818] px-4 py-3">
              <pre className={cn(
                "bg-black font-mono text-xs leading-5 whitespace-pre-wrap break-words",
                isError && !isJson ? "text-red-400" : ""
              )}>
                {isJson
                  ? <JsonView content={content} />
                  : <span className="text-zinc-300">{content}</span>
                }
              </pre>
            </div>
          </div>
        )}

        {/* Pending placeholder */}
        {isExpanded && isPending && !hasOutput && (
          <div className="border-t border-zinc-800 px-4 py-3">
            <div className="flex items-center gap-2 font-mono text-xs text-zinc-500">
              <span className="text-emerald-500">$</span>
              <span className="animate-pulse">executing...</span>
            </div>
          </div>
        )}
      </div>
    );
  }
);

ScriptExecutionBlock.displayName = "ScriptExecutionBlock";
