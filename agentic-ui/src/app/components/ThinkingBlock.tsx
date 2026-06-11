"use client";

import React, { useRef, useState } from "react";
import gsap from "gsap";
import { useGSAP } from "@gsap/react";
import { SplitText } from "gsap/SplitText";
import { MarkdownContent } from "@/app/components/MarkdownContent";

gsap.registerPlugin(useGSAP, SplitText);

interface ThinkingBlockProps {
  content: string;
  timestamp?: string;
  status?: "pending" | "completed" | "error" | "interrupted";
}

// Reveal speed for live reasoning, in seconds per character.
const SECONDS_PER_CHAR = 0.008;
const MAX_CATCHUP_SECONDS = 2.0;

/**
 * Reasoning rendered as flowing text — no box, no chrome.
 * Visually distinct from normal output: smaller, muted grey markdown.
 *
 * Liveness model:
 * - Live blocks (seen in a pending state) reveal text with a GSAP typewriter:
 *   a tween advances a character cursor toward the latest streamed content,
 *   so the reasoning always types out steadily even when the model delivers
 *   the whole reflection in one chunk.
 * - Blocks that mount with finished content (thread reload / history) play a
 *   one-time word-by-word blur reveal instead — no slow re-typing of old text.
 */
export const ThinkingBlock = React.memo<ThinkingBlockProps>(
  ({ content, status = "completed" }) => {
    const scopeRef = useRef<HTMLDivElement>(null);
    const labelRef = useRef<HTMLSpanElement>(null);
    const isPending = status === "pending";

    // True if this block ever rendered in a streaming state.
    const sawStreamingRef = useRef(isPending);
    if (isPending) sawStreamingRef.current = true;
    const isLive = sawStreamingRef.current;

    // ── Typewriter reveal for live blocks ──────────────────────────────────
    const [revealedChars, setRevealedChars] = useState(() =>
      isPending ? 0 : content.length
    );
    const cursorRef = useRef({ chars: isPending ? 0 : content.length });
    const tweenRef = useRef<gsap.core.Tween | null>(null);

    useGSAP(
      () => {
        if (!isLive) return;
        const target = content.length;
        if (cursorRef.current.chars >= target) return;
        tweenRef.current?.kill();
        tweenRef.current = gsap.to(cursorRef.current, {
          chars: target,
          duration: Math.min(
            MAX_CATCHUP_SECONDS,
            (target - cursorRef.current.chars) * SECONDS_PER_CHAR
          ),
          ease: "none",
          onUpdate: () => setRevealedChars(Math.floor(cursorRef.current.chars)),
        });
      },
      { dependencies: [content], scope: scopeRef }
    );

    // ── One-time word reveal for blocks that mount already complete ────────
    useGSAP(
      () => {
        if (isLive || !content) return;
        const target = scopeRef.current?.querySelector("[data-thinking-body]");
        if (!target) return;
        const split = new SplitText(target, { type: "words" });
        gsap.from(split.words, {
          opacity: 0,
          y: 4,
          filter: "blur(4px)",
          duration: 0.4,
          stagger: 0.012,
          ease: "power2.out",
          onComplete: () => split.revert(),
        });
        return () => split.revert();
      },
      { scope: scopeRef }
    );

    // ── Soft breathing shimmer on the label while reasoning is live ────────
    const stillTyping = isLive && revealedChars < content.length;
    const labelActive = isPending || stillTyping;
    useGSAP(
      () => {
        if (!labelRef.current) return;
        if (labelActive) {
          gsap.to(labelRef.current, {
            opacity: 0.3,
            duration: 0.8,
            yoyo: true,
            repeat: -1,
            ease: "sine.inOut",
          });
        } else {
          gsap.to(labelRef.current, { opacity: 1, duration: 0.3 });
        }
      },
      { dependencies: [labelActive], revertOnUpdate: true, scope: scopeRef }
    );

    const visibleContent = isLive ? content.slice(0, revealedChars) : content;

    return (
      <div ref={scopeRef} className="my-3 w-full">
        <span
          ref={labelRef}
          className="block select-none text-[11px] font-medium uppercase tracking-[0.16em] text-muted-foreground/60"
        >
          Reasoning
        </span>
        {visibleContent ? (
          <div
            data-thinking-body
            className="mt-1.5 w-full text-[13px] leading-relaxed text-muted-foreground/80"
          >
            <MarkdownContent content={visibleContent} />
          </div>
        ) : null}
      </div>
    );
  }
);

ThinkingBlock.displayName = "ThinkingBlock";
