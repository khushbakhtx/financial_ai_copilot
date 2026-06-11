"use client";

import { useChatContext } from "@/providers/ChatProvider";
import {
  ModelLeaderboard,
  PipelineProgress,
  DatasetCharts,
} from "@/app/components/genui";

export function useFinancialCopilot() {
  const {
    pipelineSteps,
    pipelineType,
    investigationId,
    datasetName,
    modelLeaderboard,
    chartsJson,
    chartsTitle,
  } = useChatContext();

  const pipelinePanel =
    pipelineSteps && pipelineSteps.length > 0 ? (
      <PipelineProgress
        pipeline_type={pipelineType ?? "general"}
        steps={pipelineSteps}
        investigation_id={investigationId}
        dataset_name={datasetName}
      />
    ) : null;

  const leaderboardPanel =
    modelLeaderboard && modelLeaderboard.length > 0 ? (
      <ModelLeaderboard
        models={modelLeaderboard}
        best_model={
          modelLeaderboard.reduce((a, b) => (a.auc > b.auc ? a : b)).model_name
        }
      />
    ) : null;

  const chartsPanel = (() => {
    if (!chartsJson) return null;
    try {
      return <DatasetCharts charts={JSON.parse(chartsJson)} title={chartsTitle} />;
    } catch {
      return null;
    }
  })();

  return { pipelinePanel, leaderboardPanel, chartsPanel };
}
