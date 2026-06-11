"use client";

import { z } from "zod";
import { cn } from "@/lib/utils";

export const ModelLeaderboardProps = z.object({
  models: z
    .array(
      z.object({
        model_name: z.string(),
        auc: z.number(),
        gini: z.number().optional(),
        ks: z.number().optional(),
        f1: z.number().optional(),
        rank: z.number().optional(),
      })
    )
    .describe("Ranked list of trained models with evaluation metrics"),
  title: z.string().optional().describe("Optional heading for the leaderboard"),
  best_model: z.string().optional().describe("Name of the best-performing model"),
});

type Props = z.infer<typeof ModelLeaderboardProps>;

function MetricBadge({ label, value }: { label: string; value?: number }) {
  if (value === undefined) return null;
  return (
    <span className="inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-xs bg-muted text-muted-foreground">
      <span className="opacity-60">{label}</span>
      <span className="font-semibold text-foreground">{value.toFixed(3)}</span>
    </span>
  );
}

function AucBar({ auc }: { auc: number }) {
  const pct = Math.min(100, Math.max(0, auc * 100));
  const color =
    auc >= 0.9
      ? "bg-green-500"
      : auc >= 0.8
        ? "bg-emerald-400"
        : auc >= 0.7
          ? "bg-yellow-400"
          : "bg-red-400";
  return (
    <div className="flex items-center gap-2 w-full">
      <div className="flex-1 h-1.5 rounded-full bg-muted overflow-hidden">
        <div className={cn("h-full rounded-full", color)} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-xs font-mono font-semibold w-12 text-right">{auc.toFixed(4)}</span>
    </div>
  );
}

export function ModelLeaderboard({ models, title, best_model }: Props) {
  const sorted = [...models].sort((a, b) => (b.auc ?? 0) - (a.auc ?? 0));

  return (
    <div className="rounded-xl border border-border bg-card p-4 space-y-3 w-full max-w-xl">
      <div className="flex items-center justify-between">
        <h3 className="font-semibold text-sm">{title ?? "Model Leaderboard"}</h3>
        {best_model && (
          <span className="text-xs px-2 py-0.5 rounded-full bg-green-500/10 text-green-600 border border-green-500/20 font-medium">
            Best: {best_model}
          </span>
        )}
      </div>

      <div className="space-y-3">
        {sorted.map((model, idx) => (
          <div
            key={model.model_name}
            className={cn(
              "rounded-lg p-3 space-y-2 border transition-colors",
              idx === 0
                ? "border-green-500/30 bg-green-500/5"
                : "border-border bg-background"
            )}
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="text-xs text-muted-foreground w-5 font-mono">
                  #{idx + 1}
                </span>
                <span className="font-medium text-sm">{model.model_name}</span>
              </div>
              <div className="flex gap-1.5 flex-wrap justify-end">
                <MetricBadge label="Gini" value={model.gini} />
                <MetricBadge label="KS" value={model.ks} />
                <MetricBadge label="F1" value={model.f1} />
              </div>
            </div>
            <div className="pl-7">
              <AucBar auc={model.auc} />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
