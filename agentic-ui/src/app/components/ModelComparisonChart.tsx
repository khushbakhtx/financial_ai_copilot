"use client";

import { useMemo, useState } from "react";
import { Trophy, XCircle, Loader2 } from "lucide-react";
import { ToolCall } from "@/app/types/types";
import { cn } from "@/lib/utils";

// ── Types ────────────────────────────────────────────────────────────────────

interface ModelMetrics {
  // classification — all common key variants the agent may emit
  test_roc_auc?: number;
  test_auc?: number;
  test_pr_auc?: number;
  test_gini?: number;
  // regression
  test_r2?: number;
  test_rmse?: number;
  // clustering
  silhouette?: number;
}

interface ModelEntry {
  label: string;
  model_type: string;
  metrics: ModelMetrics;
  winner: boolean;
  status: "done" | "failed" | "training" | string;
  error?: string;
}

// ── Metric display config ────────────────────────────────────────────────────

const METRIC_CONFIG: Record<
  string,
  { label: string; higherIsBetter: boolean; color: string }
> = {
  test_roc_auc: { label: "ROC-AUC",   higherIsBetter: true,  color: "#6366f1" },
  test_auc:     { label: "ROC-AUC",   higherIsBetter: true,  color: "#6366f1" },
  test_gini:    { label: "Gini",      higherIsBetter: true,  color: "#818cf8" },
  test_pr_auc:  { label: "PR-AUC",    higherIsBetter: true,  color: "#10b981" },
  test_r2:      { label: "R²",        higherIsBetter: true,  color: "#f59e0b" },
  test_rmse:    { label: "RMSE",      higherIsBetter: false, color: "#ef4444" },
  silhouette:   { label: "Silhouette",higherIsBetter: true,  color: "#8b5cf6" },
};

// Preferred display order — whichever keys are present will be shown in this order
const METRIC_ORDER = [
  "test_roc_auc",
  "test_auc",
  "test_gini",
  "test_pr_auc",
  "test_r2",
  "test_rmse",
  "silhouette",
];

// ── Helper: short algorithm label ────────────────────────────────────────────

function shortAlgo(model_type: string): string {
  return model_type
    .replace("_classification", "")
    .replace("_regression", "")
    .replace("_clustering", "")
    .replace("catboost", "CatBoost")
    .replace("logistic_regression", "LogReg")
    .replace("random_forest", "RF")
    .replace("lstm", "LSTM")
    .replace("kmeans_cluster", "KMeans")
    .replace("agglomerative", "Agg.");
}

// ── Inline SVG bar ───────────────────────────────────────────────────────────

function Bar({
  value,
  max,
  color,
  higherIsBetter,
}: {
  value: number;
  max: number;
  color: string;
  higherIsBetter: boolean;
}) {
  const pct = max > 0 ? Math.abs(value) / max : 0;
  const width = Math.max(2, Math.round(pct * 100));
  const isGood = higherIsBetter ? value >= 0 : true;

  return (
    <div className="flex items-center gap-2">
      <div className="relative h-4 w-28 overflow-hidden rounded-sm bg-muted/40">
        <div
          className="absolute left-0 top-0 h-full rounded-sm transition-all duration-500"
          style={{ width: `${width}%`, backgroundColor: isGood ? color : "#ef4444" }}
        />
      </div>
      <span className="w-10 text-right font-mono text-xs tabular-nums text-foreground">
        {value.toFixed(3)}
      </span>
    </div>
  );
}

// ── Main component ───────────────────────────────────────────────────────────

interface ModelComparisonChartProps {
  toolCall: ToolCall;
}

export function ModelComparisonChart({ toolCall }: ModelComparisonChartProps) {
  const [expanded, setExpanded] = useState(true);

  const { models, title } = useMemo(() => {
    const args = (toolCall.args ?? {}) as { models?: ModelEntry[]; title?: string };
    return {
      models: args.models ?? [],
      title: args.title ?? "Model Comparison",
    };
  }, [toolCall.args]);

  // Collect which metrics are actually present across all models
  const activeMetrics = useMemo(() => {
    const present = new Set<string>();
    for (const m of models) {
      if (m.metrics) {
        for (const k of Object.keys(m.metrics)) {
          if ((m.metrics as Record<string, unknown>)[k] != null) present.add(k);
        }
      }
    }
    return METRIC_ORDER.filter((k) => present.has(k));
  }, [models]);

  // Per-metric max value for bar scaling
  const metricMax = useMemo(() => {
    const maxes: Record<string, number> = {};
    for (const key of activeMetrics) {
      let max = 0;
      for (const m of models) {
        if (!m.metrics) continue;
        const v = (m.metrics as Record<string, number | undefined>)[key];
        if (v != null) max = Math.max(max, Math.abs(v));
      }
      maxes[key] = max || 1;
    }
    return maxes;
  }, [activeMetrics, models]);

  // Sort: winner first, then by primary metric descending (failed models last)
  const sortedModels = useMemo(() => {
    const primaryKey = activeMetrics[0];
    return [...models].sort((a, b) => {
      if (a.winner !== b.winner) return a.winner ? -1 : 1;
      const aFailed = a.status === "failed";
      const bFailed = b.status === "failed";
      if (aFailed !== bFailed) return aFailed ? 1 : -1;
      if (!primaryKey) return 0;
      const av = a.metrics ? (a.metrics as Record<string, number | undefined>)[primaryKey] ?? -Infinity : -Infinity;
      const bv = b.metrics ? (b.metrics as Record<string, number | undefined>)[primaryKey] ?? -Infinity : -Infinity;
      const cfg = METRIC_CONFIG[primaryKey];
      return cfg?.higherIsBetter === false ? av - bv : bv - av;
    });
  }, [models, activeMetrics]);

  const isPending = toolCall.result == null;

  return (
    <div className="w-full overflow-hidden rounded-lg border border-border bg-card">
      {/* Header */}
      <button
        onClick={() => setExpanded((p) => !p)}
        className="flex w-full items-center justify-between px-4 py-3 text-left hover:bg-accent/50 transition-colors"
      >
        <div className="flex items-center gap-2">
          <Trophy size={14} className="text-amber-500" />
          <span className="text-sm font-semibold tracking-tight">{title}</span>
          <span className="text-xs text-muted-foreground">({models.length} model{models.length !== 1 ? "s" : ""})</span>
        </div>
        {isPending && <Loader2 size={13} className="animate-spin text-muted-foreground" />}
      </button>

      {expanded && (
        <div className="border-t border-border px-4 pb-4 pt-3">
          {models.length === 0 ? (
            <p className="text-xs text-muted-foreground">No model data yet.</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-border">
                    <th className="pb-2 pr-4 text-left font-semibold text-muted-foreground uppercase tracking-wider">Model</th>
                    <th className="pb-2 pr-4 text-left font-semibold text-muted-foreground uppercase tracking-wider">Algorithm</th>
                    {activeMetrics.map((k) => (
                      <th key={k} className="pb-2 pr-4 text-left font-semibold text-muted-foreground uppercase tracking-wider whitespace-nowrap">
                        {METRIC_CONFIG[k]?.label ?? k}
                      </th>
                    ))}
                    <th className="pb-2 text-left font-semibold text-muted-foreground uppercase tracking-wider">Status</th>
                  </tr>
                </thead>
                <tbody>
                  {sortedModels.map((model, i) => {
                    const isWinner = model.winner;
                    const isFailed = model.status === "failed";
                    return (
                      <tr
                        key={i}
                        className={cn(
                          "border-b border-border/50 last:border-0",
                          isWinner && "bg-amber-500/5",
                          isFailed && "opacity-50"
                        )}
                      >
                        {/* Model label */}
                        <td className="py-2 pr-4 font-medium text-foreground whitespace-nowrap">
                          <div className="flex items-center gap-1.5">
                            {isWinner && <Trophy size={11} className="text-amber-500 shrink-0" />}
                            <span className={cn("font-mono", isWinner && "font-semibold")}>{model.label}</span>
                          </div>
                        </td>

                        {/* Algorithm */}
                        <td className="py-2 pr-4 text-muted-foreground whitespace-nowrap">
                          {shortAlgo(model.model_type)}
                        </td>

                        {/* Metric bars */}
                        {activeMetrics.map((k) => {
                          const v = model.metrics ? (model.metrics as Record<string, number | undefined>)[k] : undefined;
                          const cfg = METRIC_CONFIG[k];
                          return (
                            <td key={k} className="py-2 pr-4">
                              {v != null && cfg ? (
                                <Bar
                                  value={v}
                                  max={metricMax[k]}
                                  color={cfg.color}
                                  higherIsBetter={cfg.higherIsBetter}
                                />
                              ) : (
                                <span className="text-muted-foreground/50">—</span>
                              )}
                            </td>
                          );
                        })}

                        {/* Status */}
                        <td className="py-2 whitespace-nowrap">
                          {isFailed ? (
                            <div className="flex items-center gap-1 text-destructive">
                              <XCircle size={12} />
                              <span>{model.error ?? "Failed"}</span>
                            </div>
                          ) : model.status === "training" ? (
                            <div className="flex items-center gap-1 text-muted-foreground">
                              <Loader2 size={12} className="animate-spin" />
                              <span>Training…</span>
                            </div>
                          ) : (
                            <span className={cn("text-emerald-600 font-medium", isWinner && "font-semibold")}>
                              {isWinner ? "Winner" : "Done"}
                            </span>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
