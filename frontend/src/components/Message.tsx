import { useState, useCallback } from 'react';
import { Copy, Check, RefreshCw, ChevronDown, Bot, UserIcon } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import rehypeHighlight from 'rehype-highlight';
import remarkGfm from 'remark-gfm';
import type { Message as AIMessage } from 'ai';
import MemoryPanel from './MemoryPanel';
import { useChatStore } from '../store';

interface Props {
  message: AIMessage;
  isStreaming?: boolean;
}

export default function Message({ message, isStreaming }: Props) {
  const [copied, setCopied] = useState(false);
  const { memoryPanelOpen, toggleMemoryPanel } = useChatStore();
  const isUser = message.role === 'user';

  const copyText = useCallback(() => {
    navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }, [message.content]);

  // Extract metadata annotation if present
  const annotation = (message as any).annotations?.[0] as Record<string, any> | undefined;
  const showMemory = !isUser && annotation?.retrieval_info;

  return (
    <div className={`group flex gap-3 py-4 fade-in ${isUser ? '' : ''}`}>
      {/* Avatar */}
      <div
        className={`flex-shrink-0 w-7 h-7 rounded-full flex items-center justify-center text-xs
          ${isUser ? 'bg-accent/20 text-accent' : 'bg-sidebar-hover text-gray-300'}`}
      >
        {isUser ? <UserIcon size={14} /> : <Bot size={14} />}
      </div>

      {/* Content */}
      <div className="flex-1 min-w-0">
        <div className="text-xs text-sidebar-muted mb-1 font-medium">
          {isUser ? 'You' : 'Assistant'}
        </div>

        {isUser ? (
          <p className="text-gray-200 whitespace-pre-wrap">{message.content}</p>
        ) : (
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
              <span className="inline-block w-2 h-4 bg-accent animate-pulse rounded-sm ml-0.5" />
            )}
          </div>
        )}

        {/* Actions bar */}
        {!isUser && !isStreaming && message.content && (
          <div className="flex items-center gap-1 mt-2 opacity-0 group-hover:opacity-100 transition-opacity">
            <button
              onClick={copyText}
              className="flex items-center gap-1 px-2 py-1 text-xs text-sidebar-muted
                         rounded hover:bg-sidebar-hover hover:text-white transition-colors"
            >
              {copied ? <Check size={12} /> : <Copy size={12} />}
              {copied ? 'Copied' : 'Copy'}
            </button>

            {/* Intent badge */}
            {annotation?.intent && (
              <span className="px-2 py-0.5 text-[10px] rounded-full bg-sidebar-hover text-sidebar-muted">
                {annotation.intent}
                {annotation.confidence != null && (
                  <span className="ml-1 opacity-60">
                    {Math.round(annotation.confidence * 100)}%
                  </span>
                )}
              </span>
            )}

            {/* Memory panel toggle */}
            {showMemory && (
              <button
                onClick={() => toggleMemoryPanel(message.id)}
                className="flex items-center gap-1 px-2 py-1 text-xs text-sidebar-muted
                           rounded hover:bg-sidebar-hover hover:text-white transition-colors"
              >
                <ChevronDown
                  size={12}
                  className={`transition-transform ${memoryPanelOpen[message.id] ? 'rotate-180' : ''}`}
                />
                Memory
              </button>
            )}
          </div>
        )}

        {/* Expandable memory panel */}
        {showMemory && memoryPanelOpen[message.id] && (
          <MemoryPanel info={annotation!.retrieval_info as Record<string, any>} />
        )}
      </div>
    </div>
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

  // Try to extract language from child className
  const childClass = (children as any)?.props?.className || '';
  const lang = childClass.replace('hljs language-', '').replace('language-', '') || 'code';

  return (
    <div className="relative rounded-lg overflow-hidden my-3 border border-sidebar-border">
      <div className="flex items-center justify-between px-4 py-1.5 bg-sidebar-bg text-xs text-sidebar-muted">
        <span>{lang}</span>
        <button
          onClick={handleCopy}
          className="flex items-center gap-1 hover:text-white transition-colors"
        >
          {copied ? <Check size={12} /> : <Copy size={12} />}
          {copied ? 'Copied' : 'Copy'}
        </button>
      </div>
      <pre {...props} className="!mt-0 !rounded-t-none">
        {children}
      </pre>
    </div>
  );
}
