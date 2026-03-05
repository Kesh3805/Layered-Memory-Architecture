/**
 * AIMessage — premium message component with framer-motion animations.
 *
 * Surfaces backend intelligence:
 *  - Streaming phase indicators (classifying → retrieving → generating)
 *  - AI Status Bar with event chips
 *  - Intent badge with confidence dot
 *  - Expandable retrieval panel
 *  - Debug panel (when debug mode active)
 *  - Token estimation meter
 *  - Profile injection badge
 */

import { useState, useCallback, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Copy, Check, Bot, UserIcon, ChevronDown } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import rehypeHighlight from 'rehype-highlight';
import remarkGfm from 'remark-gfm';
import type { StreamMessage } from '../hooks/use-chat-stream';

import { AIStatusBar, AIIntentBadge, AIRetrievalPanel, AITokenMeter, AIDebugPanel } from './ai';
import type { StageEvent } from './ai';
import { useChatStore } from '../store';

interface Props {
  message: StreamMessage;
  isStreaming?: boolean;
}

/** Rough token estimate: ~4 chars per token */
function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

const messageVariants = {
  hidden: { opacity: 0, y: 12 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.35, ease: [0.16, 1, 0.3, 1] } },
};

export default function AIMessage({ message, isStreaming }: Props) {
  const [copied, setCopied] = useState(false);
  const [showRetrieval, setShowRetrieval] = useState(false);
  const { memoryPanelOpen, toggleMemoryPanel, debugMode } = useChatStore();
  const isUser = message.role === 'user';

  const copyText = useCallback(() => {
    navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }, [message.content]);

  // ── Parse annotations into stages + final metadata ──────────────────
  const { stages, metadata } = useMemo(() => {
    const annotations: Record<string, any>[] = message.annotations ?? [];
    const stageEvents: StageEvent[] = [];
    let meta: Record<string, any> = {};

    for (const ann of annotations) {
      if (ann.stage) {
        stageEvents.push(ann as StageEvent);
      } else if (ann.intent) {
        // Final metadata annotation
        meta = ann;
      }
    }
    return { stages: stageEvents, metadata: meta };
  }, [message.annotations]);

  const hasMetadata = Object.keys(metadata).length > 0;
  const ri = metadata.retrieval_info || stages.find(s => s.stage === 'retrieved')?.retrieval_info;
  const intent = metadata.intent || stages.find(s => s.stage === 'classified')?.intent;
  const confidence = metadata.confidence ?? stages.find(s => s.stage === 'classified')?.confidence ?? 0;

  // ── Streaming phase indicator (before content arrives) ──────────────
  const streamingPhase = useMemo(() => {
    if (!isStreaming || message.content) return null;
    if (stages.length === 0) return 'Thinking…';
    const last = stages[stages.length - 1];
    switch (last.stage) {
      case 'classified': return 'Retrieving context…';
      case 'retrieved': return 'Preparing response…';
      case 'generating': return 'Generating…';
      default: return 'Thinking…';
    }
  }, [isStreaming, message.content, stages]);

  if (isUser) {
    return (
      <motion.div
        variants={messageVariants}
        initial="hidden"
        animate="visible"
        className="group flex gap-3.5 py-5"
      >
        <div className="flex-shrink-0 w-7 h-7 rounded-lg flex items-center justify-center
                        bg-accent/10 text-accent ring-1 ring-accent/20">
          <UserIcon size={13} />
        </div>
        <div className="flex-1 min-w-0">
          <div className="text-2xs text-zinc-500 mb-1.5 font-semibold uppercase tracking-wider">You</div>
          <p className="text-zinc-200 whitespace-pre-wrap leading-relaxed">{message.content}</p>
        </div>
      </motion.div>
    );
  }

  return (
    <motion.div
      variants={messageVariants}
      initial="hidden"
      animate="visible"
      className="group flex gap-3.5 py-5"
    >
      {/* Avatar */}
      <div className="flex-shrink-0 w-7 h-7 rounded-lg flex items-center justify-center
                      bg-surface-1 text-zinc-400 ring-1 ring-surface-2">
        <Bot size={13} />
      </div>

      {/* Content column */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-1.5">
          <span className="text-2xs text-zinc-500 font-semibold uppercase tracking-wider">Assistant</span>
          {intent && <AIIntentBadge intent={intent} confidence={confidence} />}
        </div>

        {/* Status bar (stage timeline) — only for assistant messages with stages */}
        {stages.length > 0 && (
          <AIStatusBar
            stages={stages}
            isStreaming={!!isStreaming}
            onChipClick={ri ? () => setShowRetrieval(v => !v) : undefined}
          />
        )}

        {/* Streaming phase (before content) */}
        {streamingPhase && (
          <div className="flex items-center gap-2.5 py-3 text-sm text-zinc-500">
            <div className="flex gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-accent typing-dot" />
              <span className="w-1.5 h-1.5 rounded-full bg-accent typing-dot" />
              <span className="w-1.5 h-1.5 rounded-full bg-accent typing-dot" />
            </div>
            <span className="text-[13px]">{streamingPhase}</span>
          </div>
        )}

        {/* Message body (markdown) */}
        {message.content && (
          <div className="prose max-w-none">
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              rehypePlugins={[rehypeHighlight]}
              components={{
                pre: ({ children, ...props }) => (
                  <CodeBlock {...props}>{children}</CodeBlock>
                ),
              }}
            >
              {message.content}
            </ReactMarkdown>
            {isStreaming && (
              <span className="inline-block w-2 h-4 bg-accent streaming-cursor rounded-sm ml-0.5" />
            )}
          </div>
        )}

        {/* Actions bar */}
        {!isStreaming && message.content && (
          <div className="flex items-center gap-1.5 mt-3 opacity-0 group-hover:opacity-100
                          transition-opacity duration-200">
            <ActionButton onClick={copyText}>
              {copied ? <Check size={12} /> : <Copy size={12} />}
              {copied ? 'Copied' : 'Copy'}
            </ActionButton>

            {ri && (
              <ActionButton onClick={() => toggleMemoryPanel(message.id)}>
                <ChevronDown
                  size={12}
                  className={`transition-transform ${memoryPanelOpen[message.id] ? 'rotate-180' : ''}`}
                />
                Memory
              </ActionButton>
            )}

            {/* Token estimate */}
            <AITokenMeter used={estimateTokens(message.content)} />
          </div>
        )}

        {/* Retrieval panel (toggle from status bar "Details" button) */}
        <AnimatePresence>
          {showRetrieval && ri && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.25, ease: [0.16, 1, 0.3, 1] }}
            >
              <AIRetrievalPanel info={{ ...ri, intent, confidence }} />
            </motion.div>
          )}
        </AnimatePresence>

        {/* Full retrieval panel (toggle from "Memory" action) */}
        <AnimatePresence>
          {memoryPanelOpen[message.id] && ri && !showRetrieval && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.25, ease: [0.16, 1, 0.3, 1] }}
            >
              <AIRetrievalPanel info={{ ...ri, intent, confidence }} />
            </motion.div>
          )}
        </AnimatePresence>

        {/* Debug panel — only in debug mode */}
        <AnimatePresence>
          {debugMode && hasMetadata && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.2 }}
            >
              <AIDebugPanel metadata={metadata} />
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </motion.div>
  );
}

/* ── Action button ──────────────────────────────────────────────────────── */

function ActionButton({ onClick, children }: { onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      onClick={onClick}
      className="flex items-center gap-1 px-2.5 py-1 text-xs text-zinc-500 rounded-lg
                 hover:bg-surface-1 hover:text-zinc-300 transition-all duration-200"
    >
      {children}
    </button>
  );
}

/* ── Code block with copy & language label ──────────────────────────────── */

function CodeBlock({ children, ...props }: React.HTMLAttributes<HTMLPreElement>) {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    const text = (children as any)?.props?.children || '';
    navigator.clipboard.writeText(typeof text === 'string' ? text : String(text));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const childClass = (children as any)?.props?.className || '';
  const lang = childClass.replace('hljs language-', '').replace('language-', '') || 'code';

  return (
    <div className="relative rounded-xl overflow-hidden my-4 border border-surface-2/50 shadow-elevated">
      <div className="flex items-center justify-between px-4 py-2 bg-surface-0 text-xs text-zinc-500
                      border-b border-surface-2/30">
        <span className="font-mono text-2xs uppercase tracking-wider">{lang}</span>
        <button
          onClick={handleCopy}
          className="flex items-center gap-1.5 hover:text-zinc-300 transition-colors duration-200"
        >
          {copied ? <Check size={12} /> : <Copy size={12} />}
          <span>{copied ? 'Copied' : 'Copy'}</span>
        </button>
      </div>
      <pre {...props} className="!mt-0 !rounded-t-none !border-0">
        {children}
      </pre>
    </div>
  );
}
