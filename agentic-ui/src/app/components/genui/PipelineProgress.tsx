"use client";

import { z } from "zod";
import { cn } from "@/lib/utils";
import {
  CheckCircle,
  Circle,
  Clock,
  Loader,
  XCircle,
  SkipForward,
} from "lucide-react";

export const PipelineProgressProps = z.object({
  pipeline_type: z
    .enum(["credit_scoring", "fraud_detection", "general"])
    .describe("Which pipeline is running"),
  steps: z
    .array(
      z.object({
        step: z.string(),
        label: z.string(),
        status: z.enum(["pending", "running", "completed", "error", "skipped"]),
        agent: z.string().optional(),
        summary: z.string().optional(),
      })
    )
    .describe("Ordered pipeline steps with their current status"),
  investigation_id: z.string().optional(),
  dataset_name: z.string().optional(),
});

type Props = z.infer<typeof PipelineProgressProps>;
type StepStatus = "pending" | "running" | "completed" | "error" | "skipped";

const statusConfig: Record<
  StepStatus,
  { icon: React.ElementType; color: string; line: string }
> = {
  pending: { icon: Circle, color: "text-muted-foreground/40", line: "bg-border" },
  running: { icon: Loader, color: "text-blue-500", line: "bg-border" },
  completed: { icon: CheckCircle, color: "text-green-500", line: "bg-green-500" },
  error: { icon: XCircle, color: "text-red-500", line: "bg-red-400" },
  skipped: { icon: SkipForward, color: "text-muted-foreground/50", line: "bg-border" },
};

const pipelineLabels = {
  credit_scoring: "Credit Scoring Pipeline",
  fraud_detection: "Fraud Detection Pipeline",
  general: "Financial Investigation",
};

export function PipelineProgress({
  pipeline_type,
  steps,
  investigation_id,
  dataset_name,
}: Props) {
  const completedCount = steps.filter((s) => s.status === "completed").length;
  const totalCount = steps.length;

  return (
    <div className="rounded-xl border border-border bg-card p-4 space-y-4 w-full max-w-sm">
      {/* Header */}
      <div className="space-y-1">
        <div className="flex items-center justify-between">
          <h3 className="font-semibold text-sm">
            {pipelineLabels[pipeline_type] ?? "Pipeline"}
          </h3>
          <span className="text-xs text-muted-foreground">
            {completedCount}/{totalCount}
          </span>
        </div>
        {dataset_name && (
          <p className="text-xs text-muted-foreground font-mono truncate">
            {dataset_name}
          </p>
        )}
        {/* Progress bar */}
        <div className="h-1 rounded-full bg-muted overflow-hidden">
          <div
            className="h-full rounded-full bg-green-500 transition-all duration-500"
            style={{ width: `${(completedCount / Math.max(totalCount, 1)) * 100}%` }}
          />
        </div>
      </div>

      {/* Steps */}
      <div className="space-y-0">
        {steps.map((step, idx) => {
          const cfg = statusConfig[step.status] ?? statusConfig.pending;
          const Icon = cfg.icon;
          const isLast = idx === steps.length - 1;

          return (
            <div key={step.step} className="flex gap-3">
              {/* Icon + connector line */}
              <div className="flex flex-col items-center">
                <div
                  className={cn(
                    "flex-shrink-0 mt-0.5",
                    step.status === "running" && "animate-spin"
                  )}
                >
                  <Icon size={14} className={cfg.color} />
                </div>
                {!isLast && (
                  <div className={cn("w-px flex-1 min-h-4 my-0.5", cfg.line)} />
                )}
              </div>

              {/* Label + summary */}
              <div className="pb-3 min-w-0">
                <p
                  className={cn(
                    "text-sm leading-tight",
                    step.status === "pending" || step.status === "skipped"
                      ? "text-muted-foreground/60"
                      : "text-foreground"
                  )}
                >
                  {step.label}
                </p>
                {step.summary && step.status !== "pending" && (
                  <p className="text-xs text-muted-foreground/70 mt-0.5 leading-relaxed">
                    {step.summary}
                  </p>
                )}
                {step.status === "running" && !step.summary && (
                  <p className="text-xs text-blue-500/70 mt-0.5">Running…</p>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {investigation_id && (
        <p className="text-xs text-muted-foreground/50 font-mono border-t border-border pt-2 truncate">
          {investigation_id}
        </p>
      )}
    </div>
  );
}
