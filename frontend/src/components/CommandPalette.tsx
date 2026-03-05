/**
 * Command Palette — Ctrl+K / Cmd+K trigger.
 *
 * Quick-access to: new chat, search, profile, debug toggle, clear memory, etc.
 */

import { useState, useEffect, useRef, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Plus,
  User,
  Bug,
  MessageSquare,
  Command,
  X,
} from 'lucide-react';
import { useChatStore } from '../store';

interface PaletteCommand {
  id: string;
  icon: React.ReactNode;
  label: string;
  description?: string;
  action: () => void;
}

const overlayVariants = {
  hidden: { opacity: 0 },
  visible: { opacity: 1, transition: { duration: 0.15 } },
  exit: { opacity: 0, transition: { duration: 0.1 } },
};

const panelVariants = {
  hidden: { opacity: 0, scale: 0.97, y: -8 },
  visible: {
    opacity: 1,
    scale: 1,
    y: 0,
    transition: { duration: 0.2, ease: [0.16, 1, 0.3, 1] },
  },
  exit: {
    opacity: 0,
    scale: 0.97,
    y: -8,
    transition: { duration: 0.12 },
  },
};

export default function CommandPalette() {
  const {
    commandPaletteOpen,
    setCommandPaletteOpen,
    setConversationId,
    setProfileModalOpen,
    debugMode,
    toggleDebugMode,
    conversations,
  } = useChatStore();

  const [query, setQuery] = useState('');
  const [selectedIdx, setSelectedIdx] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  // Focus input when opened
  useEffect(() => {
    if (commandPaletteOpen) {
      setQuery('');
      setSelectedIdx(0);
      setTimeout(() => inputRef.current?.focus(), 50);
    }
  }, [commandPaletteOpen]);

  // Commands list
  const commands: PaletteCommand[] = useMemo(() => {
    const cmds: PaletteCommand[] = [
      {
        id: 'new-chat',
        icon: <Plus size={15} />,
        label: 'New Chat',
        description: 'Start a new conversation',
        action: () => { setConversationId(null); close(); },
      },
      {
        id: 'profile',
        icon: <User size={15} />,
        label: 'View Profile',
        description: 'Open profile & memory manager',
        action: () => { setProfileModalOpen(true); close(); },
      },
      {
        id: 'debug',
        icon: <Bug size={15} />,
        label: debugMode ? 'Disable Debug Mode' : 'Enable Debug Mode',
        description: 'Show raw AI internals on messages',
        action: () => { toggleDebugMode(); close(); },
      },
    ];

    // Add recent conversations as jump targets
    conversations.slice(0, 8).forEach(c => {
      cmds.push({
        id: `conv-${c.id}`,
        icon: <MessageSquare size={15} />,
        label: c.title || 'Untitled',
        description: 'Switch to conversation',
        action: () => { setConversationId(c.id); close(); },
      });
    });

    return cmds;
  }, [conversations, debugMode, setConversationId, setProfileModalOpen, toggleDebugMode]);

  // Filter by query
  const filtered = useMemo(() => {
    if (!query.trim()) return commands;
    const q = query.toLowerCase();
    return commands.filter(
      c => c.label.toLowerCase().includes(q) || c.description?.toLowerCase().includes(q)
    );
  }, [commands, query]);

  // Clamp selection
  useEffect(() => {
    setSelectedIdx(i => Math.min(i, Math.max(0, filtered.length - 1)));
  }, [filtered]);

  const close = () => setCommandPaletteOpen(false);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault();
        setSelectedIdx(i => Math.min(i + 1, filtered.length - 1));
        break;
      case 'ArrowUp':
        e.preventDefault();
        setSelectedIdx(i => Math.max(i - 1, 0));
        break;
      case 'Enter':
        e.preventDefault();
        filtered[selectedIdx]?.action();
        break;
      case 'Escape':
        close();
        break;
    }
  };

  return (
    <AnimatePresence>
      {commandPaletteOpen && (
        <motion.div
          key="palette-overlay"
          variants={overlayVariants}
          initial="hidden"
          animate="visible"
          exit="exit"
          className="fixed inset-0 z-50 flex items-start justify-center pt-[15vh]"
          onClick={close}
        >
          <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" />
          <motion.div
            variants={panelVariants}
            initial="hidden"
            animate="visible"
            exit="exit"
            className="relative w-full max-w-lg glass border border-surface-2/40
                        rounded-2xl shadow-elevated-lg overflow-hidden"
            onClick={e => e.stopPropagation()}
          >
            {/* Search input */}
            <div className="flex items-center gap-3 px-4 py-3.5 border-b border-surface-2/30">
              <Command size={14} className="text-zinc-500" />
              <input
                ref={inputRef}
                value={query}
                onChange={e => setQuery(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Type a command…"
                className="flex-1 bg-transparent text-sm text-white placeholder:text-zinc-600 outline-none"
              />
              <button
                onClick={close}
                title="Close"
                className="p-1 rounded-lg text-zinc-600 hover:text-white hover:bg-surface-1
                           transition-all duration-200"
              >
                <X size={13} />
              </button>
            </div>

            {/* Results */}
            <div className="max-h-80 overflow-y-auto py-1.5">
              {filtered.length === 0 && (
                <p className="text-center text-zinc-600 text-xs py-6">No matching commands</p>
              )}
              {filtered.map((cmd, i) => (
                <button
                  key={cmd.id}
                  onClick={cmd.action}
                  onMouseEnter={() => setSelectedIdx(i)}
                  className={`w-full flex items-center gap-3 px-4 py-2.5 text-left transition-all duration-150
                    ${i === selectedIdx
                      ? 'bg-surface-1/60 text-white'
                      : 'text-zinc-400 hover:bg-surface-1/30'}`}
                >
                  <span className="text-zinc-500">{cmd.icon}</span>
                  <div className="flex-1 min-w-0">
                    <div className="text-[13px] truncate">{cmd.label}</div>
                    {cmd.description && (
                      <div className="text-2xs text-zinc-600 truncate">{cmd.description}</div>
                    )}
                  </div>
                  {i === selectedIdx && (
                    <span className="text-2xs text-zinc-500 px-1.5 py-0.5 rounded-md bg-surface-0
                                     border border-surface-2/30 font-mono">
                      ↵
                    </span>
                  )}
                </button>
              ))}
            </div>

            {/* Footer hint */}
            <div className="border-t border-surface-2/30 px-4 py-2.5 flex items-center gap-4 text-2xs text-zinc-600">
              <span><kbd className="px-1 py-0.5 rounded-md bg-surface-1 border border-surface-2/30 font-mono">↑↓</kbd> Navigate</span>
              <span><kbd className="px-1 py-0.5 rounded-md bg-surface-1 border border-surface-2/30 font-mono">↵</kbd> Select</span>
              <span><kbd className="px-1 py-0.5 rounded-md bg-surface-1 border border-surface-2/30 font-mono">Esc</kbd> Close</span>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
