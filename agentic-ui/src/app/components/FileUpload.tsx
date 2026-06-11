"use client";

import React, { useRef, useState, useEffect, useCallback } from "react";
import { Paperclip, X, FileSpreadsheet, Eye } from "lucide-react";

interface FileUploadProps {
    onFileSelect: (file: File | null) => void;
    selectedFile: File | null;
    disabled?: boolean;
    uploadType: "train" | "blind";
    onUploadTypeChange: (type: "train" | "blind") => void;
}

export const FileUpload: React.FC<FileUploadProps> = ({
    onFileSelect,
    selectedFile,
    disabled,
    uploadType,
    onUploadTypeChange,
}) => {
    const fileInputRef = useRef<HTMLInputElement>(null);
    const buttonRef = useRef<HTMLButtonElement>(null);
    const popoverRef = useRef<HTMLDivElement>(null);
    const [showPopover, setShowPopover] = useState(false);
    const [popoverPos, setPopoverPos] = useState({ top: 0, left: 0 });

    const updatePosition = useCallback(() => {
        if (buttonRef.current) {
            const rect = buttonRef.current.getBoundingClientRect();
            setPopoverPos({
                top: rect.top - 8,
                left: rect.left,
            });
        }
    }, []);

    useEffect(() => {
        const handleClickOutside = (e: MouseEvent) => {
            if (
                popoverRef.current && !popoverRef.current.contains(e.target as Node) &&
                buttonRef.current && !buttonRef.current.contains(e.target as Node)
            ) {
                setShowPopover(false);
            }
        };
        if (showPopover) {
            document.addEventListener("mousedown", handleClickOutside);
        }
        return () => document.removeEventListener("mousedown", handleClickOutside);
    }, [showPopover]);

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files.length > 0) {
            onFileSelect(e.target.files[0]);
        }
    };

    const removeFile = () => {
        onFileSelect(null);
        if (fileInputRef.current) {
            fileInputRef.current.value = "";
        }
    };

    const handleTypeSelect = (type: "train" | "blind") => {
        onUploadTypeChange(type);
        setShowPopover(false);
        fileInputRef.current?.click();
    };

    const togglePopover = () => {
        if (!showPopover) {
            updatePosition();
        }
        setShowPopover((prev) => !prev);
    };

    return (
        <div className="flex items-center gap-2">
            <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileChange}
                className="hidden"
                accept=".csv,.xlsx,.xls"
                disabled={disabled}
            />

            {!selectedFile ? (
                <>
                    <button
                        ref={buttonRef}
                        type="button"
                        onClick={togglePopover}
                        disabled={disabled}
                        className="flex h-8 w-8 items-center justify-center rounded-md text-tertiary transition-colors hover:bg-sidebar hover:text-primary disabled:opacity-50"
                        title="Attach file"
                    >
                        <Paperclip size={18} />
                    </button>

                    {showPopover && (
                        <div
                            ref={popoverRef}
                            className="fixed z-50 w-44 -translate-y-full overflow-hidden rounded-lg border border-border bg-popover shadow-lg"
                            style={{ top: popoverPos.top, left: popoverPos.left }}
                        >
                            <button
                                type="button"
                                onClick={() => handleTypeSelect("train")}
                                className="flex w-full items-center gap-2.5 px-3 py-2.5 text-sm text-popover-foreground transition-colors hover:bg-accent"
                            >
                                <FileSpreadsheet size={15} className="text-[#2F6868]" />
                                Train File
                            </button>
                            <div className="mx-2 h-px bg-border" />
                            <button
                                type="button"
                                onClick={() => handleTypeSelect("blind")}
                                className="flex w-full items-center gap-2.5 px-3 py-2.5 text-sm text-popover-foreground transition-colors hover:bg-accent"
                            >
                                <Eye size={15} className="text-blue-500" />
                                Blind File
                            </button>
                        </div>
                    )}
                </>
            ) : (
                <div className="flex items-center gap-1.5 rounded-md border border-border bg-sidebar px-2 py-1 text-xs">
                    <Paperclip size={12} className="text-primary" />
                    <span className="text-[10px] font-medium uppercase text-muted-foreground">
                        {uploadType}
                    </span>
                    <span className="max-w-[120px] truncate text-primary">
                        {selectedFile.name}
                    </span>
                    <button
                        type="button"
                        onClick={removeFile}
                        className="ml-0.5 text-tertiary hover:text-destructive"
                        disabled={disabled}
                    >
                        <X size={12} />
                    </button>
                </div>
            )}
        </div>
    );
};
