import { useRef, useEffect, useState } from 'react';
import { PanelLeftOpen, ArrowDown, Bug, GitBranch, Brain } from 'lucide-react';
import { useChatStore } from '../store';
import type { useChatStream } from '../hooks/use-chat-stream';
import AIMessage from './AIMessage';
import InputArea from './InputArea';
import WelcomeScreen from './WelcomeScreen';
import { AIThreadPanel } from './ai';

interface Props {
  chat: ReturnType<typeof useChatStream>;
}

export default function ChatArea({ chat }: Props) {
  const {
    conversationId,
    sidebarOpen,
    toggleSidebar,
    debugMode,
    toggleDebugMode,
    threadPanelOpen,
    toggleThreadPanel,
    toggleResearchDash,
  } = useChatStore();
  const bottomRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [showScrollBtn, setShowScrollBtn] = useState(false);

  // Auto-scroll while streaming
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chat.messages]);

  // Scroll-to-bottom button visibility
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const handler = () => {
      const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 100;
      setShowScrollBtn(!atBottom);
    };
    el.addEventListener('scroll', handler, { passive: true });
    return () => el.removeEventListener('scroll', handler);
  }, []);

  const scrollToBottom = () =>
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });

  const hasMessages = chat.messages.length > 0;

  return (
    <div className="flex flex-col flex-1 min-w-0 relative">
      {/* Header bar */}
      <div className="flex items-center gap-3 px-4 py-3 border-b border-sidebar-border/50">
        {!sidebarOpen && (
          <button
            onClick={toggleSidebar}
            title="Open sidebar"
            className="p-2 text-sidebar-muted hover:text-white rounded-lg
                       hover:bg-sidebar-hover transition-colors"
          >
            <PanelLeftOpen size={18} />
          </button>
        )}
        <span className="text-sm text-sidebar-muted">
          {conversationId ? 'Chat' : 'New Conversation'}
        </span>
        <div className="ml-auto flex items-center gap-2">
          {conversationId && (
            <>
              <button
                onClick={toggleResearchDash}
                className="p-1.5 rounded-lg transition-colors text-xs flex items-center gap-1.5
                  text-sidebar-muted hover:text-white hover:bg-sidebar-hover"
                title="Research dashboard"
              >
                <Brain size={14} />
              </button>
              <button
                onClick={toggleThreadPanel}
                className={`p-1.5 rounded-lg transition-colors text-xs flex items-center gap-1.5
                  ${threadPanelOpen
                    ? 'bg-accent/15 text-accent border border-accent/30'
                    : 'text-sidebar-muted hover:text-white hover:bg-sidebar-hover'}`}
                title="Toggle thread panel"
              >
                <GitBranch size={14} />
                {threadPanelOpen && <span className="text-[10px]">Threads</span>}
              </button>
            </>
          )}
          <button
            onClick={toggleDebugMode}
            className={`p-1.5 rounded-lg transition-colors text-xs flex items-center gap-1.5
              ${debugMode
                ? 'bg-yellow-500/15 text-yellow-400 border border-yellow-500/30'
                : 'text-sidebar-muted hover:text-white hover:bg-sidebar-hover'}`}
            title="Toggle debug mode"
          >
            <Bug size={14} />
            {debugMode && <span className="text-[10px]">Debug</span>}
          </button>
        </div>
      </div>

      {/* Messages + Thread Panel */}
      <div className="flex flex-1 min-h-0">
        {/* Messages area */}
        <div className="flex flex-col flex-1 min-w-0 relative">
          <div ref={containerRef} className="flex-1 overflow-y-auto">
            {!hasMessages ? (
              <WelcomeScreen onSuggestion={chat.send} />
            ) : (
              <div className="max-w-3xl mx-auto px-4 py-6 space-y-1">
                {chat.messages.map((m) => (
                  <AIMessage
                    key={m.id}
                    message={m}
                    isStreaming={chat.streamingId === m.id}
                  />
                ))}
                <div ref={bottomRef} />
              </div>
            )}
          </div>

          {/* Scroll-to-bottom FAB */}
          {showScrollBtn && (
            <button
              onClick={scrollToBottom}
              title="Scroll to bottom"
              className="absolute bottom-28 left-1/2 -translate-x-1/2 p-2 rounded-full
                         bg-sidebar-hover/90 text-white shadow-lg hover:bg-sidebar-active
                         transition-all z-10"
            >
              <ArrowDown size={18} />
            </button>
          )}

          {/* Input area */}
          <InputArea chat={chat} />
        </div>

        {/* Thread panel (right sidebar) */}
        <div
          className={`${
            threadPanelOpen ? 'w-72' : 'w-0'
          } transition-all duration-200 flex-shrink-0 overflow-hidden border-l border-sidebar-border/50 bg-sidebar-bg`}
        >
          <AIThreadPanel />
        </div>
      </div>
    </div>
  );
}
