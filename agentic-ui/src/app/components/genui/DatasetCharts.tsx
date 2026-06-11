"use client";

import { z } from "zod";
import dynamic from "next/dynamic";
import { useState } from "react";
import { cn } from "@/lib/utils";

// Factory pattern: bind react-plotly.js to plotly.js-dist-min (smaller build)
const Plot = dynamic(
  () =>
    Promise.all([
      import("react-plotly.js/factory"),
      import("plotly.js-dist-min"),
    ]).then(([{ default: createPlotlyComponent }, { default: Plotly }]) =>
      createPlotlyComponent(Plotly)
    ),
  { ssr: false }
);

export const DatasetChartsProps = z.object({
  title: z.string().optional().describe("Section heading e.g. 'Credit Risk Dataset — EDA Charts'"),
  charts: z
    .array(
      z.object({
        chart_title: z.string().describe("Chart heading shown above the plot"),
        data: z.array(z.any()).describe("Plotly data array (traces)"),
        layout: z.any().optional().describe("Plotly layout object (optional — defaults applied)"),
      })
    )
    .describe("Array of Plotly chart specs to render"),
});

type Props = z.infer<typeof DatasetChartsProps>;

const BASE_LAYOUT = {
  paper_bgcolor: "transparent",
  plot_bgcolor: "transparent",
  font: { color: "#e5e7eb", size: 12, family: "Inter, sans-serif" },
  margin: { t: 32, r: 16, b: 48, l: 48 },
  legend: { bgcolor: "transparent", bordercolor: "transparent" },
  xaxis: {
    gridcolor: "#374151",
    linecolor: "#374151",
    zerolinecolor: "#374151",
    tickfont: { color: "#9ca3af" },
  },
  yaxis: {
    gridcolor: "#374151",
    linecolor: "#374151",
    zerolinecolor: "#374151",
    tickfont: { color: "#9ca3af" },
  },
  colorway: [
    "#2F6868", "#34d399", "#60a5fa", "#f59e0b",
    "#a78bfa", "#f87171", "#fb923c", "#e879f9",
  ],
};

export function DatasetCharts({ title, charts }: Props) {
  const [activeIdx, setActiveIdx] = useState(0);

  if (!charts?.length) return <></>;

  const active = charts[activeIdx];

  return (
    <div className="rounded-xl border border-border bg-card w-full max-w-2xl overflow-hidden">
      {/* Header */}
      {title && (
        <div className="px-4 pt-3 pb-2 border-b border-border">
          <h3 className="font-semibold text-sm">{title}</h3>
        </div>
      )}

      {/* Tab bar — one tab per chart */}
      {charts.length > 1 && (
        <div className="flex overflow-x-auto border-b border-border bg-sidebar px-2 pt-2 gap-1">
          {charts.map((chart, idx) => (
            <button
              key={idx}
              type="button"
              onClick={() => setActiveIdx(idx)}
              className={cn(
                "px-3 py-1.5 text-xs rounded-t-md whitespace-nowrap transition-colors shrink-0",
                idx === activeIdx
                  ? "bg-background text-foreground font-medium border border-border border-b-background -mb-px"
                  : "text-muted-foreground hover:text-foreground"
              )}
            >
              {chart.chart_title}
            </button>
          ))}
        </div>
      )}

      {/* Active chart */}
      <div className="p-3">
        {charts.length === 1 && (
          <p className="text-xs font-medium text-muted-foreground mb-2">
            {active.chart_title}
          </p>
        )}
        <Plot
          data={active.data as any}
          layout={{
            ...BASE_LAYOUT,
            ...(active.layout ?? {}),
            paper_bgcolor: "transparent",
            plot_bgcolor: "transparent",
            font: { ...(active.layout?.font ?? {}), color: "#e5e7eb", size: 12 },
            height: 300,
          }}
          config={{
            displayModeBar: true,
            displaylogo: false,
            modeBarButtonsToRemove: [
              "sendDataToCloud",
              "autoScale2d",
              "resetScale2d",
            ] as any,
            responsive: true,
          }}
          style={{ width: "100%", height: "300px" }}
          useResizeHandler
        />
      </div>
    </div>
  );
}
