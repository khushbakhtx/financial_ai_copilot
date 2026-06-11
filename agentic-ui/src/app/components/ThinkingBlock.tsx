"use client";

import React, { useState } from "react";
import { Brain, ChevronDown, ChevronUp } from "lucide-react";
import { cn } from "@/lib/utils";

interface ThinkingBlockProps {
  content: string;
  timestamp?: string;
  status?: "pending" | "completed" | "error" | "interrupted";
}

export const ThinkingBlock = React.memo<ThinkingBlockProps>(
  ({ content, timestamp, status = "completed" }) => {
    const [isExpanded, setIsExpanded] = useState(true);

    const isPending = status === "pending";
    const isInterrupted = status === "interrupted";

    return (
      <div className="my-3 w-full overflow-hidden rounded-lg border border-purple-200/50 bg-purple-50/30 dark:border-purple-900/30 dark:bg-purple-950/20">
        <button
          onClick={() => setIsExpanded(!isExpanded)}
          className="flex w-full items-center justify-between px-4 py-3 text-left transition-colors hover:bg-purple-100/40 dark:hover:bg-purple-900/20"
        >
          <div className="flex items-center gap-3">
            <div className="flex h-6 w-6 items-center justify-center rounded-md bg-purple-500/10">
              <Brain
                size={14}
                className={cn(
                  "text-purple-600 dark:text-purple-400",
                  isPending && "animate-pulse"
                )}
              />
            </div>
            <span className="text-sm font-medium text-purple-900 dark:text-purple-100">
              {isPending ? "Thinking..." : isInterrupted ? "Thinking (Interrupted)" : "Thinking"}
            </span>
            {timestamp && (
              <span className="text-xs text-purple-600/60 dark:text-purple-400/60">
                {timestamp}
              </span>
            )}
          </div>
          {isExpanded ? (
            <ChevronUp size={16} className="text-purple-600/70 dark:text-purple-400/70" />
          ) : (
            <ChevronDown size={16} className="text-purple-600/70 dark:text-purple-400/70" />
          )}
        </button>

        {isExpanded && (
          <div className="border-t border-purple-200/30 dark:border-purple-900/30">
            <div className="px-4 py-3">
              <pre className="whitespace-pre-wrap font-sans text-sm leading-relaxed text-purple-900/90 dark:text-purple-100/80">
                {content || "Processing..."}
              </pre>
            </div>
          </div>
        )}
      </div>
    );
  }
);

ThinkingBlock.displayName = "ThinkingBlock";
