"use client";

import React, { useState } from "react";
import { Download, FileCode, FileText, Database, Box, File } from "lucide-react";
import { cn } from "@/lib/utils";

const TERMINAL_SERVER =
  process.env.NEXT_PUBLIC_TERMINAL_SERVER_URL ?? "http://localhost:8001";

export interface ArtifactMeta {
  gridfs_id: string;
  investigation_id?: string;
  name: string;
  title?: string;
  kind?: string; // "image" | "model" | "report" | "data" | "code" | "file"
  step?: string;
  content_type?: string;
  size_bytes?: number;
  created_at?: string;
}

function formatSize(bytes?: number): string {
  if (!bytes) return "";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(2)} MB`;
}

function kindIcon(kind?: string) {
  switch (kind) {
    case "model":
      return <Box size={15} className="text-muted-foreground" />;
    case "report":
      return <FileText size={15} className="text-muted-foreground" />;
    case "code":
      return <FileCode size={15} className="text-muted-foreground" />;
    case "data":
      return <Database size={15} className="text-muted-foreground" />;
    default:
      return <File size={15} className="text-muted-foreground" />;
  }
}

function downloadUrl(a: ArtifactMeta): string {
  return `${TERMINAL_SERVER}/artifacts/${a.gridfs_id}/download`;
}

function rawUrl(a: ArtifactMeta): string {
  return `${TERMINAL_SERVER}/artifacts/${a.gridfs_id}/raw`;
}

function ImageArtifact({ artifact }: { artifact: ArtifactMeta }) {
  const [expanded, setExpanded] = useState(false);
  return (
    <div className="overflow-hidden rounded-lg border border-border bg-card">
      <button
        type="button"
        onClick={() => setExpanded((v) => !v)}
        className="block w-full cursor-zoom-in"
        title={expanded ? "Collapse" : "Expand"}
      >
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={rawUrl(artifact)}
          alt={artifact.title || artifact.name}
          className={cn(
            "w-full object-contain bg-white",
            expanded ? "max-h-[640px]" : "max-h-56"
          )}
        />
      </button>
      <div className="flex items-center justify-between gap-2 border-t border-border px-3 py-2">
        <div className="min-w-0">
          <p className="truncate text-xs font-medium">{artifact.title || artifact.name}</p>
          {artifact.step && (
            <p className="text-[10px] text-muted-foreground">{artifact.step}</p>
          )}
        </div>
        <a
          href={downloadUrl(artifact)}
          className="shrink-0 rounded-md p-1.5 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
          title={`Download ${artifact.name}`}
        >
          <Download size={14} />
        </a>
      </div>
    </div>
  );
}

function FileArtifact({ artifact }: { artifact: ArtifactMeta }) {
  return (
    <div className="flex items-center justify-between gap-3 rounded-lg border border-border bg-card px-3 py-2.5">
      <div className="flex min-w-0 items-center gap-2.5">
        {kindIcon(artifact.kind)}
        <div className="min-w-0">
          <p className="truncate text-xs font-medium">{artifact.title || artifact.name}</p>
          <p className="text-[10px] text-muted-foreground">
            {artifact.name}
            {artifact.size_bytes ? ` · ${formatSize(artifact.size_bytes)}` : ""}
            {artifact.step ? ` · ${artifact.step}` : ""}
          </p>
        </div>
      </div>
      <a
        href={downloadUrl(artifact)}
        className="flex shrink-0 items-center gap-1.5 rounded-md border border-border px-2.5 py-1.5 text-xs font-medium text-foreground transition-colors hover:bg-muted"
      >
        <Download size={13} />
        Download
      </a>
    </div>
  );
}

export function ArtifactsPanel({ artifacts }: { artifacts: ArtifactMeta[] }) {
  if (!artifacts || artifacts.length === 0) {
    return (
      <p className="p-4 text-sm text-muted-foreground">
        No artifacts yet — charts and model files appear here as the pipeline runs.
      </p>
    );
  }

  const images = artifacts.filter((a) => a.kind === "image");
  const files = artifacts.filter((a) => a.kind !== "image");

  return (
    <div className="flex flex-col gap-4 p-4">
      {images.length > 0 && (
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          {images.map((a) => (
            <ImageArtifact key={a.gridfs_id} artifact={a} />
          ))}
        </div>
      )}
      {files.length > 0 && (
        <div className="flex flex-col gap-2">
          {files.map((a) => (
            <FileArtifact key={a.gridfs_id} artifact={a} />
          ))}
        </div>
      )}
    </div>
  );
}
