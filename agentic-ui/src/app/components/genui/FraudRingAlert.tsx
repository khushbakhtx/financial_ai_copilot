"use client";

import { z } from "zod";
import { cn } from "@/lib/utils";
import { AlertTriangle, Link, Users, Smartphone } from "lucide-react";

export const FraudRingAlertProps = z.object({
  rings: z
    .array(
      z.object({
        ring_id: z.string(),
        account_count: z.number(),
        fraud_rate: z.number().optional(),
        shared_entities: z.array(z.string()).optional(),
        risk_score: z.number().optional(),
      })
    )
    .describe("Detected fraud rings"),
  total_accounts_flagged: z.number().optional(),
  cross_investigation_matches: z.number().optional().describe("Entities seen in prior investigations"),
  investigation_id: z.string().optional(),
});

type Props = z.infer<typeof FraudRingAlertProps>;

function RiskBar({ score }: { score: number }) {
  const pct = Math.min(100, Math.max(0, score * 100));
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-1.5 rounded-full bg-muted overflow-hidden">
        <div
          className={cn(
            "h-full rounded-full",
            score >= 0.8 ? "bg-red-500" : score >= 0.5 ? "bg-orange-400" : "bg-yellow-400"
          )}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="text-xs font-mono w-8 text-right">{(score * 100).toFixed(0)}%</span>
    </div>
  );
}

export function FraudRingAlert({
  rings,
  total_accounts_flagged,
  cross_investigation_matches,
  investigation_id,
}: Props) {
  const highRisk = rings.filter((r) => (r.risk_score ?? 0) >= 0.7);

  return (
    <div className="rounded-xl border border-red-500/30 bg-red-500/5 p-4 space-y-4 w-full max-w-xl">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <AlertTriangle size={16} className="text-red-500" />
          <h3 className="font-semibold text-sm">Fraud Ring Detection</h3>
        </div>
        <div className="flex gap-2">
          <span className="text-xs px-2 py-0.5 rounded-full bg-red-500/10 text-red-600 border border-red-500/20 font-medium">
            {rings.length} ring{rings.length !== 1 ? "s" : ""} detected
          </span>
        </div>
      </div>

      {/* Stats row */}
      <div className="grid grid-cols-3 gap-2">
        <div className="rounded-lg border border-border bg-background p-2 text-center">
          <Users size={14} className="mx-auto text-muted-foreground mb-1" />
          <p className="text-xs text-muted-foreground">Accounts</p>
          <p className="font-semibold text-sm">
            {total_accounts_flagged?.toLocaleString() ?? rings.reduce((s, r) => s + r.account_count, 0).toLocaleString()}
          </p>
        </div>
        <div className="rounded-lg border border-border bg-background p-2 text-center">
          <AlertTriangle size={14} className="mx-auto text-red-500 mb-1" />
          <p className="text-xs text-muted-foreground">High Risk</p>
          <p className="font-semibold text-sm text-red-500">{highRisk.length}</p>
        </div>
        <div className="rounded-lg border border-border bg-background p-2 text-center">
          <Link size={14} className="mx-auto text-orange-500 mb-1" />
          <p className="text-xs text-muted-foreground">Cross-inv.</p>
          <p className="font-semibold text-sm text-orange-500">
            {cross_investigation_matches ?? 0}
          </p>
        </div>
      </div>

      {/* Ring list */}
      <div className="space-y-2">
        {rings.slice(0, 5).map((ring) => (
          <div
            key={ring.ring_id}
            className="rounded-lg border border-border bg-background p-3 space-y-2"
          >
            <div className="flex items-center justify-between">
              <span className="text-xs font-mono text-muted-foreground">
                {ring.ring_id}
              </span>
              <div className="flex items-center gap-2 text-xs text-muted-foreground">
                <Users size={12} />
                <span>{ring.account_count} accounts</span>
                {ring.fraud_rate !== undefined && (
                  <span className="text-red-500 font-semibold">
                    {(ring.fraud_rate * 100).toFixed(0)}% fraud
                  </span>
                )}
              </div>
            </div>
            {ring.risk_score !== undefined && (
              <div className="space-y-0.5">
                <p className="text-xs text-muted-foreground">Risk score</p>
                <RiskBar score={ring.risk_score} />
              </div>
            )}
            {ring.shared_entities && ring.shared_entities.length > 0 && (
              <div className="flex gap-1 flex-wrap">
                {ring.shared_entities.slice(0, 4).map((e, i) => (
                  <span
                    key={i}
                    className="inline-flex items-center gap-1 text-xs px-1.5 py-0.5 rounded bg-muted text-muted-foreground"
                  >
                    <Smartphone size={10} />
                    {e}
                  </span>
                ))}
                {ring.shared_entities.length > 4 && (
                  <span className="text-xs text-muted-foreground/60">
                    +{ring.shared_entities.length - 4} more
                  </span>
                )}
              </div>
            )}
          </div>
        ))}
        {rings.length > 5 && (
          <p className="text-xs text-muted-foreground text-center">
            +{rings.length - 5} more rings
          </p>
        )}
      </div>

      {investigation_id && (
        <p className="text-xs text-muted-foreground/50 font-mono border-t border-border pt-2 truncate">
          {investigation_id}
        </p>
      )}
    </div>
  );
}
