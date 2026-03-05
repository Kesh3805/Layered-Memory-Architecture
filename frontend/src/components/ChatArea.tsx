import { useRef, useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
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
      <div className="flex items-center gap-3 px-5 py-3 border-b border-surface-2/30
                      bg-gradient-to-r from-chat-bg via-chat-bg to-chat-bg">
        {!sidebarOpen && (
          <motion.button
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            onClick={toggleSidebar}
            title="Open sidebar"
            className="p-2 text-sidebar-muted hover:text-white rounded-xl
                       hover:bg-surface-1 transition-all duration-200"
          >
            <PanelLeftOpen size={18} />
          </motion.button>
        )}
        <span className="text-sm text-zinc-500 font-medium">
          {conversationId ? 'Chat' : 'New Conversation'}
        </span>
        <div className="ml-auto flex items-center gap-1.5">
          {conversationId && (
            <>
              <HeaderButton
                icon={<Brain size={14} />}
                onClick={toggleResearchDash}
                tooltip="Research dashboard"
              />
              <HeaderButton
                icon={<GitBranch size={14} />}
                onClick={toggleThreadPanel}
                tooltip="Toggle thread panel"
                active={threadPanelOpen}
                activeLabel="Threads"
              />
            </>
          )}
          <HeaderButton
            icon={<Bug size={14} />}
            onClick={toggleDebugMode}
            tooltip="Toggle debug mode"
            active={debugMode}
            activeLabel="Debug"
            activeColor="bg-yellow-500/10 text-yellow-400 border-yellow-500/20"
          />
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
          <AnimatePresence>
            {showScrollBtn && (
              <motion.button
                initial={{ opacity: 0, y: 10, scale: 0.9 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: 10, scale: 0.9 }}
                transition={{ duration: 0.2, ease: [0.16, 1, 0.3, 1] }}
                onClick={scrollToBottom}
                title="Scroll to bottom"
                className="absolute bottom-28 left-1/2 -translate-x-1/2 p-2.5 rounded-full
                           glass text-zinc-300 shadow-elevated hover:text-white
                           hover:shadow-glow-sm transition-all z-10"
              >
                <ArrowDown size={16} />
              </motion.button>
            )}
          </AnimatePresence>

          {/* Input area */}
          <InputArea chat={chat} />
        </div>

        {/* Thread panel (right sidebar) */}
        <motion.div
          animate={{ width: threadPanelOpen ? 288 : 0 }}
          transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
          className="flex-shrink-0 overflow-hidden border-l border-surface-2/30 bg-sidebar-bg"
        >
          <AIThreadPanel />
        </motion.div>
      </div>
    </div>
  );
}

/* ── Header button ─────────────────────────────────────────────────────── */

function HeaderButton({
  icon, onClick, tooltip, active, activeLabel, activeColor,
}: {
  icon: React.ReactNode;
  onClick: () => void;
  tooltip: string;
  active?: boolean;
  activeLabel?: string;
  activeColor?: string;
}) {
  const defaultActiveColor = 'bg-accent/10 text-accent border-accent/20';
  return (
    <button
      onClick={onClick}
      className={`p-1.5 rounded-xl transition-all duration-200 text-xs flex items-center gap-1.5
        ${active
          ? `${activeColor || defaultActiveColor} border shadow-glow-sm`
          : 'text-zinc-500 hover:text-white hover:bg-surface-1'}`}
      title={tooltip}
    >
      {icon}
      {active && activeLabel && (
        <span className="text-2xs font-medium">{activeLabel}</span>
      )}
    </button>
  );
}
