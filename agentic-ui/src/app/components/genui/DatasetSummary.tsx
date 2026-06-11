"use client";

import { z } from "zod";
import { cn } from "@/lib/utils";

export const DatasetSummaryProps = z.object({
  dataset_name: z.string().describe("Dataset filename"),
  rows: z.number().describe("Number of rows"),
  cols: z.number().describe("Number of columns"),
  target_col: z.string().optional().describe("Target column name"),
  target_rate: z.number().optional().describe("Positive class rate (0-1) e.g. fraud rate or default rate"),
  top_features: z
    .array(z.object({ name: z.string(), importance: z.number() }))
    .optional()
    .describe("Top features by importance score"),
  issues: z
    .array(z.string())
    .optional()
    .describe("Data quality issues found (leakage, missing, imbalance, etc.)"),
});

type Props = z.infer<typeof DatasetSummaryProps>;

export function DatasetSummary({
  dataset_name,
  rows,
  cols,
  target_col,
  target_rate,
  top_features,
  issues,
}: Props) {
  const hasIssues = issues && issues.length > 0;

  return (
    <div className="rounded-xl border border-border bg-card p-4 space-y-4 w-full max-w-xl">
      <div className="flex items-start justify-between gap-2">
        <div>
          <h3 className="font-semibold text-sm">{dataset_name}</h3>
          <p className="text-xs text-muted-foreground">
            {rows.toLocaleString()} rows × {cols} columns
          </p>
        </div>
        {target_rate !== undefined && (
          <div className="text-right">
            <p className="text-xs text-muted-foreground">
              {target_col ?? "Target"} rate
            </p>
            <p
              className={cn(
                "font-semibold text-sm",
                target_rate > 0.15
                  ? "text-red-500"
                  : target_rate > 0.05
                    ? "text-yellow-500"
                    : "text-green-500"
              )}
            >
              {(target_rate * 100).toFixed(2)}%
            </p>
          </div>
        )}
      </div>

      {top_features && top_features.length > 0 && (
        <div className="space-y-1.5">
          <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
            Top Features
          </p>
          <div className="space-y-1">
            {top_features.slice(0, 6).map((f) => (
              <div key={f.name} className="flex items-center gap-2">
                <span className="text-xs text-muted-foreground w-36 truncate">
                  {f.name}
                </span>
                <div className="flex-1 h-1.5 rounded-full bg-muted overflow-hidden">
                  <div
                    className="h-full rounded-full bg-blue-500"
                    style={{
                      width: `${Math.min(100, f.importance * 100)}%`,
                    }}
                  />
                </div>
                <span className="text-xs font-mono text-muted-foreground w-10 text-right">
                  {f.importance.toFixed(3)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {hasIssues && (
        <div className="space-y-1">
          <p className="text-xs font-medium text-yellow-600 uppercase tracking-wider">
            Issues Found
          </p>
          <ul className="space-y-0.5">
            {issues!.map((issue, i) => (
              <li key={i} className="text-xs text-muted-foreground flex gap-1.5">
                <span className="text-yellow-500 shrink-0">•</span>
                {issue}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
