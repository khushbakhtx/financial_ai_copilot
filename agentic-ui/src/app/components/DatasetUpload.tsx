"use client";

import React, { useRef, useState } from "react";
import { Paperclip, X } from "lucide-react";

const TERMINAL_SERVER =
  process.env.NEXT_PUBLIC_TERMINAL_SERVER_URL ?? "http://localhost:8001";

interface DatasetUploadProps {
  onUploaded: (filename: string) => void;
  onAutoFill?: (message: string) => void;
  disabled?: boolean;
}

export function DatasetUpload({ onUploaded, onAutoFill, disabled }: DatasetUploadProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setSelectedFile(file);

    setIsUploading(true);
    try {
      const formData = new FormData();
      formData.append("file", file);

      const res = await fetch(`${TERMINAL_SERVER}/datasets/upload`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        throw new Error(`Upload failed: ${res.statusText}`);
      }

      onUploaded(file.name);
      onAutoFill?.(
        `I've uploaded '${file.name}'. Please run a full financial investigation.`
      );
    } catch (err) {
      console.error("Dataset upload error:", err);
      alert("Failed to upload dataset. Make sure the terminal server is running.");
    } finally {
      setIsUploading(false);
      // Reset input so the same file can be re-selected
      if (inputRef.current) inputRef.current.value = "";
    }
  };

  const handleClear = () => {
    setSelectedFile(null);
    if (inputRef.current) inputRef.current.value = "";
  };

  return (
    <div className="flex items-center gap-1">
      <input
        ref={inputRef}
        type="file"
        accept=".csv,.xlsx,.xls,.parquet"
        className="hidden"
        onChange={handleFileChange}
        disabled={disabled || isUploading}
      />

      {selectedFile ? (
        <div className="flex items-center gap-1.5 rounded-md border border-border bg-sidebar px-2 py-1 text-xs text-muted-foreground">
          <span className="max-w-32 truncate">{selectedFile.name}</span>
          <button
            type="button"
            onClick={handleClear}
            className="flex-shrink-0 text-muted-foreground hover:text-foreground"
          >
            <X size={12} />
          </button>
        </div>
      ) : (
        <button
          type="button"
          onClick={() => inputRef.current?.click()}
          disabled={disabled || isUploading}
          title="Upload dataset (CSV, Excel, Parquet)"
          className="flex h-8 w-8 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-accent hover:text-foreground disabled:opacity-40"
        >
          {isUploading ? (
            <span className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-current border-t-transparent" />
          ) : (
            <Paperclip size={16} />
          )}
        </button>
      )}
    </div>
  );
}
