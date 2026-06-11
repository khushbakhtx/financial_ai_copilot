"use client";

import { z } from "zod";
import { cn } from "@/lib/utils";
import { AlertTriangle, CheckCircle, Info, XCircle } from "lucide-react";

export const FindingCardProps = z.object({
  severity: z
    .enum(["CRITICAL", "WARNING", "INFO", "POSITIVE"])
    .describe("Severity level of the finding"),
  title: z.string().describe("Short finding title"),
  content: z.string().describe("Full finding description"),
  agent: z.string().optional().describe("Agent that produced this finding"),
  metric: z.string().optional().describe("Optional key metric e.g. 'AUC: 0.923'"),
});

type Props = z.infer<typeof FindingCardProps>;

const severityConfig = {
  CRITICAL: {
    icon: XCircle,
    border: "border-red-500/30",
    bg: "bg-red-500/5",
    badge: "bg-red-500/10 text-red-600 border-red-500/20",
    icon_class: "text-red-500",
    label: "Critical",
  },
  WARNING: {
    icon: AlertTriangle,
    border: "border-yellow-500/30",
    bg: "bg-yellow-500/5",
    badge: "bg-yellow-500/10 text-yellow-600 border-yellow-500/20",
    icon_class: "text-yellow-500",
    label: "Warning",
  },
  INFO: {
    icon: Info,
    border: "border-blue-500/30",
    bg: "bg-blue-500/5",
    badge: "bg-blue-500/10 text-blue-600 border-blue-500/20",
    icon_class: "text-blue-500",
    label: "Info",
  },
  POSITIVE: {
    icon: CheckCircle,
    border: "border-green-500/30",
    bg: "bg-green-500/5",
    badge: "bg-green-500/10 text-green-600 border-green-500/20",
    icon_class: "text-green-500",
    label: "Positive",
  },
};

export function FindingCard({ severity, title, content, agent, metric }: Props) {
  const cfg = severityConfig[severity] ?? severityConfig.INFO;
  const Icon = cfg.icon;

  return (
    <div
      className={cn(
        "rounded-xl border p-4 space-y-2 w-full max-w-xl",
        cfg.border,
        cfg.bg
      )}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="flex items-center gap-2">
          <Icon size={16} className={cfg.icon_class} />
          <span className="font-semibold text-sm">{title}</span>
        </div>
        <span
          className={cn(
            "text-xs px-2 py-0.5 rounded-full border font-medium shrink-0",
            cfg.badge
          )}
        >
          {cfg.label}
        </span>
      </div>

      <p className="text-sm text-muted-foreground leading-relaxed pl-6">{content}</p>

      {(agent || metric) && (
        <div className="flex items-center gap-3 pl-6 pt-1">
          {agent && (
            <span className="text-xs text-muted-foreground/70 font-mono">{agent}</span>
          )}
          {metric && (
            <span className="text-xs font-semibold text-foreground">{metric}</span>
          )}
        </div>
      )}
    </div>
  );
}
